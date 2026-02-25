-- ============================================================
-- C2: Postgres 네이티브 파티셔닝 + 시계열 롤업 아키텍처
-- ============================================================
-- CTO 지적: 3개월 아카이빙 시 CausalImpact/GSC의
-- 장기 시계열 베이스라인이 소멸하는 문제 해결.
--
-- 아키텍처:
-- [원시 테이블] → 주간 파티션 → 3개월 후 DROP
--       ↓ (pg_cron 트리거)
-- [롤업 테이블] → 일별 집계 → 영구 보존 (인과 감사용)
--
-- 인과 감사 엔진 라우팅:
-- - 단기 분석 (< 90일): 원시 파티션 테이블
-- - 장기 분석 (≥ 90일): 롤업 테이블
-- ============================================================

-- ── 1. 파티셔닝된 원시 테이블 ──

CREATE TABLE IF NOT EXISTS audit_decisions (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    decision_id     UUID        NOT NULL,
    agent_id        TEXT        NOT NULL,
    action_type     TEXT        NOT NULL,
    treatment_value FLOAT       DEFAULT 0.0,
    context         JSONB       DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

CREATE TABLE IF NOT EXISTS audit_outcomes (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    decision_id     UUID        NOT NULL REFERENCES audit_decisions(id),
    outcome_metric  TEXT        NOT NULL,  -- 'conversion_rate', 'revenue', etc.
    outcome_value   FLOAT       DEFAULT 0.0,
    pre_value       FLOAT       DEFAULT 0.0,
    post_value      FLOAT       DEFAULT 0.0,
    measured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (measured_at);

-- ── 2. 주간(Weekly) 파티션 자동 생성 함수 ──
-- pg_partman 대체: Postgres 네이티브 파티셔닝 + pg_cron

CREATE OR REPLACE FUNCTION create_weekly_partition(
    parent_table TEXT,
    ts_column    TEXT DEFAULT 'created_at'
)
RETURNS TEXT AS $$
DECLARE
    week_start DATE := date_trunc('week', NOW())::DATE;
    week_end   DATE := week_start + INTERVAL '7 days';
    part_name  TEXT := parent_table || '_w' ||
                       to_char(week_start, 'IYYY_IW');
BEGIN
    -- 이미 존재하면 스킵
    IF EXISTS (
        SELECT 1 FROM pg_class WHERE relname = part_name
    ) THEN
        RETURN part_name || ' (already exists)';
    END IF;

    EXECUTE format(
        'CREATE TABLE %I PARTITION OF %I
         FOR VALUES FROM (%L) TO (%L)',
        part_name, parent_table,
        week_start, week_end
    );

    -- 인덱스: 에이전트별 + 시간별 조회 최적화
    EXECUTE format(
        'CREATE INDEX %I ON %I (agent_id, %I)',
        part_name || '_agent_idx', part_name, ts_column
    );

    RETURN part_name || ' created';
END;
$$ LANGUAGE plpgsql;

-- ── 3. 롤업 테이블 (영구 보존) ──

CREATE TABLE IF NOT EXISTS daily_agent_rollup (
    rollup_date     DATE        NOT NULL,
    agent_id        TEXT        NOT NULL,
    action_type     TEXT        NOT NULL,
    decision_count  INT         DEFAULT 0,
    avg_treatment   FLOAT       DEFAULT 0.0,
    avg_outcome     FLOAT       DEFAULT 0.0,
    min_outcome     FLOAT       DEFAULT 0.0,
    max_outcome     FLOAT       DEFAULT 0.0,
    std_outcome     FLOAT       DEFAULT 0.0,
    success_rate    FLOAT       DEFAULT 0.0,  -- 긍정 결과 비율
    total_revenue   FLOAT       DEFAULT 0.0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (rollup_date, agent_id, action_type)
);

-- 인과 감사용 장기 트렌드 인덱스
CREATE INDEX IF NOT EXISTS idx_rollup_agent_date
    ON daily_agent_rollup (agent_id, rollup_date);
CREATE INDEX IF NOT EXISTS idx_rollup_action_date
    ON daily_agent_rollup (action_type, rollup_date);

-- ── 4. 롤업 프로시저 ──
-- 원시 데이터를 일별 집계하여 롤업 테이블로 이관

CREATE OR REPLACE FUNCTION rollup_daily_stats(target_date DATE DEFAULT CURRENT_DATE - 1)
RETURNS INT AS $$
DECLARE
    rows_inserted INT;
BEGIN
    INSERT INTO daily_agent_rollup (
        rollup_date, agent_id, action_type,
        decision_count, avg_treatment, avg_outcome,
        min_outcome, max_outcome, std_outcome,
        success_rate, total_revenue
    )
    SELECT
        target_date,
        d.agent_id,
        d.action_type,
        COUNT(*)::INT,
        AVG(d.treatment_value),
        AVG(o.outcome_value),
        MIN(o.outcome_value),
        MAX(o.outcome_value),
        COALESCE(STDDEV(o.outcome_value), 0),
        AVG(CASE WHEN o.outcome_value > o.pre_value THEN 1.0 ELSE 0.0 END),
        SUM(o.outcome_value)
    FROM audit_decisions d
    LEFT JOIN audit_outcomes o ON d.decision_id = o.decision_id
    WHERE d.created_at >= target_date
      AND d.created_at < target_date + INTERVAL '1 day'
    GROUP BY d.agent_id, d.action_type
    ON CONFLICT (rollup_date, agent_id, action_type)
    DO UPDATE SET
        decision_count = EXCLUDED.decision_count,
        avg_treatment  = EXCLUDED.avg_treatment,
        avg_outcome    = EXCLUDED.avg_outcome,
        min_outcome    = EXCLUDED.min_outcome,
        max_outcome    = EXCLUDED.max_outcome,
        std_outcome    = EXCLUDED.std_outcome,
        success_rate   = EXCLUDED.success_rate,
        total_revenue  = EXCLUDED.total_revenue,
        created_at     = NOW();

    GET DIAGNOSTICS rows_inserted = ROW_COUNT;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

-- ── 5. 파티션 아카이빙 (3개월 초과 Drop) ──
-- 실행 전 반드시 rollup_daily_stats() 완료 확인

CREATE OR REPLACE FUNCTION archive_old_partitions(
    parent_table TEXT,
    retention_months INT DEFAULT 3
)
RETURNS TEXT AS $$
DECLARE
    cutoff_date DATE := (NOW() - (retention_months || ' months')::INTERVAL)::DATE;
    part_record RECORD;
    dropped     INT := 0;
BEGIN
    FOR part_record IN
        SELECT inhrelid::regclass::TEXT AS part_name
        FROM pg_inherits
        WHERE inhparent = parent_table::regclass
    LOOP
        -- 파티션 이름에서 주차 추출하여 cutoff과 비교
        -- 형식: parent_table_wYYYY_WW
        IF part_record.part_name < parent_table || '_w' ||
           to_char(cutoff_date, 'IYYY_IW') THEN
            EXECUTE 'DROP TABLE IF EXISTS ' || part_record.part_name;
            dropped := dropped + 1;
        END IF;
    END LOOP;

    RETURN dropped || ' partitions archived';
END;
$$ LANGUAGE plpgsql;

-- ── 6. pg_cron 스케줄 (Supabase 환경) ──
-- 매일 02:00 UTC: 롤업 실행
-- SELECT cron.schedule('daily-rollup', '0 2 * * *',
--     $$SELECT rollup_daily_stats()$$);
--
-- 매주 일요일 03:00 UTC: 파티션 생성
-- SELECT cron.schedule('weekly-partition-decisions', '0 3 * * 0',
--     $$SELECT create_weekly_partition('audit_decisions')$$);
-- SELECT cron.schedule('weekly-partition-outcomes', '0 3 * * 0',
--     $$SELECT create_weekly_partition('audit_outcomes', 'measured_at')$$);
--
-- 매월 1일 04:00 UTC: 3개월 초과 파티션 아카이빙
-- SELECT cron.schedule('monthly-archive', '0 4 1 * *',
--     $$SELECT archive_old_partitions('audit_decisions')$$);
