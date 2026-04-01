# -*- coding: utf-8 -*-
"""WhyLab Engine Entrypoint (CLI v3).

사용법:
  # 합성 데이터 파이프라인
  python -m engine.cli --scenario A

  # 외부 데이터 (CSV/Parquet/SQL/BigQuery 자동 감지)
  python -m engine.cli --data data.csv --treatment coupon --outcome purchase
  python -m engine.cli --data "postgresql://user:pass@host/db" --db-query "SELECT * FROM users"

  # RAG 질의
  python -m engine.cli --query "왜 연체율이 줄었어?"

  # 모니터링 (1회 드리프트 체크)
  python -m engine.cli --monitor
"""

import argparse
import sys
import io
from pathlib import Path

# Windows 콘솔 인코딩 호환성 확보
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from engine.config import DEFAULT_CONFIG
from engine.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(
        description="WhyLab Causal Decision Intelligence Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  %(prog)s --scenario A                             # 합성 데이터 시나리오 A
  %(prog)s --data sales.csv --treatment ad --outcome buy  # CSV 분석
  %(prog)s --query "쿠폰 효과가 있어?"                  # RAG 질의
  %(prog)s --monitor --interval 30                   # 30분 간격 모니터링
        """,
    )
    
    # 데이터 관련 인자
    data_group = parser.add_argument_group("데이터 설정")
    data_group.add_argument("--data", type=str, help="데이터 경로 또는 DB 연결 URI")
    data_group.add_argument("--treatment", type=str, default="treatment", help="처치 변수 컬럼명")
    data_group.add_argument("--outcome", type=str, default="outcome", help="결과 변수 컬럼명")
    data_group.add_argument("--features", type=str, help="피처 컬럼 (쉼표 구분)")
    data_group.add_argument("--source-type", type=str, help="데이터 소스 타입 (csv/sql/bigquery 등, 미지정 시 자동 감지)")
    data_group.add_argument("--db-query", type=str, help="SQL/BigQuery 쿼리")
    data_group.add_argument("--db-table", type=str, help="테이블명 (쿼리 대신 전체 로드)")
    
    # 실행 옵션
    run_group = parser.add_argument_group("실행 옵션")
    run_group.add_argument("--scenario", type=str, default="A", help="시나리오 (A/B, 기본: A)")
    run_group.add_argument("--query", type=str, help="RAG 질의 모드: 분석 결과에 대해 질문")
    run_group.add_argument("--persona", type=str, default="product_owner",
                           choices=["growth_hacker", "risk_manager", "product_owner"],
                           help="RAG 답변 페르소나 (기본: product_owner)")
    
    # 모니터링 옵션
    monitor_group = parser.add_argument_group("모니터링")
    monitor_group.add_argument("--monitor", action="store_true", help="드리프트 모니터링 모드")
    monitor_group.add_argument("--interval", type=int, default=60, help="모니터링 주기 (분, 기본: 60)")
    monitor_group.add_argument("--max-runs", type=int, help="최대 모니터링 실행 횟수")
    monitor_group.add_argument("--slack-webhook", type=str, help="Slack 알림 웹훅 URL")
    
    args = parser.parse_args()
    config = DEFAULT_CONFIG
    
    print("WhyLab Engine CLI (v3) Initializing...")

    # ── 모드 1: 모니터링 ──
    if args.monitor:
        _run_monitor(args, config)
        return

    # ── 모드 2: RAG 질의 ──
    if args.query:
        _run_rag_query(args, config)
        return

    # ── 모드 3: 파이프라인 실행 ──
    _run_pipeline(args, config)


def _run_monitor(args, config):
    """드리프트 모니터링을 실행합니다."""
    print(f"🔄 모니터링 모드 (주기: {args.interval}분)")
    try:
        from engine.monitoring import MonitoringScheduler, Alerter
        
        alerter = Alerter(
            slack_webhook_url=args.slack_webhook,
            log_alerts=True,
        )
        scheduler = MonitoringScheduler(
            config=config,
            alerter=alerter,
            interval_minutes=args.interval,
            scenario=args.scenario,
        )
        
        if args.max_runs:
            print(f"최대 {args.max_runs}회 실행 후 종료")
            scheduler.start(max_runs=args.max_runs)
        else:
            # 1회 실행 후 결과 출력
            result = scheduler.run_once()
            if result.drifted:
                print(f"⚠️ 드리프트 감지! metric={result.metric}, score={result.score:.4f}")
            else:
                print(f"✅ 안정 (metric={result.metric})")
    except ImportError:
        print("모니터링 모듈 로드 실패. engine.monitoring 패키지를 확인하세요.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n모니터링 중단.")


def _run_rag_query(args, config):
    """RAG 기반 질의응답을 실행합니다."""
    print(f"💬 RAG 질의: \"{args.query}\" (페르소나: {args.persona})")
    try:
        from engine.rag.agent import RAGAgent
        agent = RAGAgent(config)
        
        print("인덱싱 중...")
        agent.index_knowledge()
        
        print("답변 생성 중...")
        answer = agent.ask(args.query, persona=args.persona)
        print(f"\n{'='*50}")
        print(answer)
        print(f"{'='*50}\n")
    except ImportError:
        print("RAG 모듈 오류: pip install chromadb sentence-transformers")
        sys.exit(1)
    except Exception as e:
        print(f"RAG 오류: {e}")
        sys.exit(1)


def _run_pipeline(args, config):
    """인과추론 파이프라인을 실행합니다."""
    # 외부 데이터 설정
    if args.data:
        print(f"📂 외부 데이터: {args.data}")
        config.data.input_path = args.data
        config.data.treatment_col = args.treatment
        config.data.outcome_col = args.outcome
        
        if args.features:
            config.data.feature_cols = [f.strip() for f in args.features.split(",")]
            print(f"피처: {config.data.feature_cols}")
        
        # DB 쿼리/테이블 설정
        if args.db_query:
            config.data.query = args.db_query
        if args.db_table:
            config.data.table = args.db_table
    else:
        print(f"🧪 합성 데이터 모드: 시나리오 {args.scenario}")

    # 파이프라인 실행
    orchestrator = Orchestrator(config)
    
    try:
        orchestrator.run_pipeline(scenario=args.scenario)
        print("✅ 파이프라인 완료!")
    except Exception as e:
        print(f"❌ 파이프라인 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
