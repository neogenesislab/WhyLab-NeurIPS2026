-- WhyLab v2.0 Causal Audit Engine — Supabase 마이그레이션
-- 리서치 기반 시계열 최적화 스키마

-- ========================================
-- 1. decisions 테이블
-- ========================================
CREATE TABLE IF NOT EXISTS audit_decisions (
    id                      UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    decision_id             TEXT UNIQUE NOT NULL,
    agent_type              TEXT NOT NULL,
    agent_name              TEXT NOT NULL,
    decision_type           TEXT NOT NULL,
    treatment               TEXT NOT NULL,
    target_sbu              TEXT NOT NULL,
    target_metric           TEXT NOT NULL,
    treatment_value         JSONB,
    context                 JSONB DEFAULT '{}',
    expected_effect         TEXT DEFAULT 'positive',
    observation_window_days INT DEFAULT 7,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- 복합 인덱스: 에이전트 유형 + 결정 유형 (빈번 필터링)
CREATE INDEX IF NOT EXISTS idx_decisions_agent
    ON audit_decisions(agent_type, decision_type);

-- SBU별 조회
CREATE INDEX IF NOT EXISTS idx_decisions_sbu
    ON audit_decisions(target_sbu);

-- 시간순 조회
CREATE INDEX IF NOT EXISTS idx_decisions_created
    ON audit_decisions(created_at DESC);

-- ========================================
-- 2. outcomes 테이블 (시계열 최적화)
-- ========================================
CREATE TABLE IF NOT EXISTS audit_outcomes (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    outcome_id  TEXT UNIQUE NOT NULL,
    metric      TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    sbu         TEXT NOT NULL,
    source      TEXT DEFAULT 'ga4',
    period      TEXT DEFAULT 'daily',
    metadata    JSONB DEFAULT '{}',
    observed_at TIMESTAMPTZ NOT NULL
);

-- 시계열 인덱스 (최신순, 가장 빈번한 조회 패턴)
CREATE INDEX IF NOT EXISTS idx_outcomes_ts
    ON audit_outcomes(observed_at DESC);

-- SBU + 지표 복합 인덱스 (매칭 쿼리 최적화)
CREATE INDEX IF NOT EXISTS idx_outcomes_sbu_metric
    ON audit_outcomes(sbu, metric);

-- ========================================
-- 3. audit_results 테이블
-- ========================================
CREATE TABLE IF NOT EXISTS audit_results (
    id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    audit_id            TEXT UNIQUE NOT NULL,
    decision_id         TEXT REFERENCES audit_decisions(decision_id),
    verdict             TEXT NOT NULL,
    confidence          DOUBLE PRECISION NOT NULL,
    ate                 DOUBLE PRECISION DEFAULT 0,
    ate_ci              DOUBLE PRECISION[] DEFAULT '{0,0}',
    p_value             DOUBLE PRECISION,
    method              TEXT DEFAULT 'lightweight_t_test',
    refutation_passed   BOOLEAN DEFAULT FALSE,
    recommendation      TEXT,
    pipeline_results    JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 판결별 조회
CREATE INDEX IF NOT EXISTS idx_results_verdict
    ON audit_results(verdict);

-- 결정 ID로 감사 결과 조회
CREATE INDEX IF NOT EXISTS idx_results_decision
    ON audit_results(decision_id);

-- ========================================
-- 4. feedback_signals 테이블
-- ========================================
CREATE TABLE IF NOT EXISTS audit_feedback_signals (
    id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    decision_id       TEXT NOT NULL,
    agent_name        TEXT NOT NULL,
    verdict           TEXT NOT NULL,
    confidence        DOUBLE PRECISION NOT NULL,
    damping_factor    DOUBLE PRECISION NOT NULL,
    effective_weight  DOUBLE PRECISION NOT NULL,
    action            TEXT NOT NULL,  -- reinforce | suppress | hold
    memo              TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_agent
    ON audit_feedback_signals(agent_name);

CREATE INDEX IF NOT EXISTS idx_feedback_action
    ON audit_feedback_signals(action);

-- ========================================
-- RLS 정책 (보안)
-- ========================================
ALTER TABLE audit_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_outcomes ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_feedback_signals ENABLE ROW LEVEL SECURITY;

-- 서비스 키로 모든 작업 허용 (에이전트 백엔드용)
CREATE POLICY "Service role full access" ON audit_decisions
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON audit_outcomes
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON audit_results
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON audit_feedback_signals
    FOR ALL USING (auth.role() = 'service_role');
