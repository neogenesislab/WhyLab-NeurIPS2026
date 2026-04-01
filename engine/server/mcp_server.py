# -*- coding: utf-8 -*-
"""WhyLab Membrane Server — MCP Protocol Interface (v2).

이 서버는 WhyLab 엔진을 외부 에이전트(Claude Desktop 등)와 연결하는
표준 인터페이스(Membrane) 역할을 합니다.

v2 변경점:
- simulate_intervention: Mock → 실제 로직 구현
- 신규 tool: get_debate_verdict, compare_scenarios, ask_rag
- 신규 resource: whylab://report/latest, whylab://benchmark/summary
"""

from mcp.server.fastmcp import FastMCP
import json
from pathlib import Path
from typing import Any, Dict

# WhyLab 엔진 모듈 임포트
from engine.pipeline import run_pipeline
from engine.config import WhyLabConfig

# MCP 서버 초기화 (서버 이름: WhyLab)
mcp = FastMCP("WhyLab")

# 전역 설정 로드
config = WhyLabConfig()


# ──────────────────────────────────────────────
# Resources
# ──────────────────────────────────────────────
@mcp.resource("whylab://data/latest")
def get_latest_data() -> str:
    """최신 분석 결과 JSON 데이터를 반환합니다."""
    json_path = config.paths.dashboard_data_dir / "latest.json"
    if not json_path.exists():
        return json.dumps(
            {"error": "No data found. Run pipeline first."}, ensure_ascii=False
        )
    with open(json_path, "r", encoding="utf-8") as f:
        return f.read()


@mcp.resource("whylab://report/latest")
def get_latest_report() -> str:
    """최신 분석 리포트(마크다운)를 반환합니다."""
    report_dir = config.paths.reports_dir
    reports = sorted(report_dir.glob("whylab_report_*.md"))
    if not reports:
        return "아직 생성된 리포트가 없습니다. `run_analysis`를 먼저 실행하세요."
    with open(reports[-1], "r", encoding="utf-8") as f:
        return f.read()


@mcp.resource("whylab://benchmark/summary")
def get_benchmark_summary() -> str:
    """벤치마크 결과 요약을 반환합니다."""
    results_dir = config.paths.project_root / "results"
    if not results_dir.exists():
        return "벤치마크 결과가 없습니다."

    summaries = []
    for json_file in sorted(results_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = json_file.stem
            if isinstance(data, dict) and "ate" in data:
                summaries.append(f"- {name}: ATE={data['ate']}")
        except Exception:
            continue

    return "\n".join(summaries) if summaries else "벤치마크 결과 파싱 실패."


# ──────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────
@mcp.tool()
def run_analysis(scenario: str = "A") -> str:
    """인과추론 파이프라인을 실행합니다.

    Args:
        scenario: "A" (Credit Limit → Default) 또는 "B" (Marketing → Signup).
    """
    try:
        result = run_pipeline(scenario=scenario)
        summary = {
            "ate": result.get("ate"),
            "model_type": result.get("model_type"),
            "sensitivity": result.get("sensitivity_results", {}).get("status"),
            "json_path": result.get("json_path"),
        }
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error running analysis: {str(e)}"


@mcp.tool()
def get_debate_verdict() -> str:
    """최신 AI Debate 결과 (Growth Hacker vs Risk Manager) 를 반환합니다.

    Returns:
        판결(CAUSAL/NOT_CAUSAL/UNCERTAIN), 확신도, 비즈니스 권고사항.
    """
    json_path = config.paths.dashboard_data_dir / "latest.json"
    if not json_path.exists():
        return "분석 결과가 없습니다. run_analysis를 먼저 실행하세요."

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    debate = data.get("debate", {})
    if not debate:
        return "Debate 결과가 포함되지 않았습니다."

    result = {
        "verdict": debate.get("verdict", "UNKNOWN"),
        "confidence": debate.get("confidence", 0),
        "recommendation": debate.get("recommendation", "N/A"),
        "pro_evidence_count": debate.get("pro_count", 0),
        "con_evidence_count": debate.get("con_count", 0),
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def simulate_intervention(
    treatment_intensity: float = 100.0,
    target_ratio: float = 0.5,
) -> str:
    """정책 개입 시뮬레이션 — 처치 강도와 타겟 비율에 따른 비즈니스 결과 예측.

    Args:
        treatment_intensity: 처치 강도 (예: 신용한도 상향액, 단위: 만원). 기본: 100.
        target_ratio: 타겟 유저 비율 (0.0~1.0). 기본: 0.5.
    """
    json_path = config.paths.dashboard_data_dir / "latest.json"
    if not json_path.exists():
        return "분석 결과가 없습니다. run_analysis를 먼저 실행하세요."

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ate = data.get("ate", 0)
    if isinstance(ate, dict):
        ate = ate.get("point_estimate", 0)

    # 비즈니스 로직 (PolicySimulator.tsx와 동일)
    base_revenue_per_user = 50  # 만원
    total_users = 10000
    target_users = int(total_users * target_ratio)
    intensity_factor = treatment_intensity / 100.0

    revenue = target_users * base_revenue_per_user * intensity_factor * (1 + abs(ate))
    cost = target_users * treatment_intensity * 0.3
    risk_factor = 1.0 + (target_ratio ** 2) * 0.5
    net_profit = revenue - (cost * risk_factor)
    roi = ((net_profit / max(cost * risk_factor, 1)) * 100)

    result = {
        "treatment_intensity": treatment_intensity,
        "target_ratio": target_ratio,
        "target_users": target_users,
        "expected_revenue": round(revenue, 0),
        "expected_cost": round(cost * risk_factor, 0),
        "net_profit": round(net_profit, 0),
        "roi_percent": round(roi, 1),
        "recommendation": (
            "🚀 배포 권장" if roi > 20
            else "⚖️ A/B 테스트 권장" if roi > 0
            else "🛑 보류 권장"
        ),
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def ask_rag(query: str, persona: str = "product_owner") -> str:
    """RAG 기반 자연어 질의응답 — 분석 결과에 대해 질문합니다.

    Args:
        query: 질문 (예: "왜 연체율이 줄었어?").
        persona: 답변 페르소나 ("growth_hacker"|"risk_manager"|"product_owner").
    """
    try:
        from engine.rag.agent import RAGAgent

        agent = RAGAgent(config)
        agent.index_knowledge()
        return agent.ask(query, persona=persona)
    except Exception as e:
        return f"RAG 질의 실패: {str(e)}"


@mcp.tool()
def compare_scenarios() -> str:
    """시나리오 A(신용한도)와 B(마케팅 쿠폰)를 비교 분석합니다."""
    results = {}
    for scenario in ["A", "B"]:
        try:
            result = run_pipeline(scenario=scenario)
            ate = result.get("ate", 0)
            if isinstance(ate, dict):
                ate = ate.get("point_estimate", 0)

            debate = result.get("debate", {})
            results[scenario] = {
                "ate": ate,
                "verdict": debate.get("verdict", "UNKNOWN"),
                "confidence": debate.get("confidence", 0),
            }
        except Exception as e:
            results[scenario] = {"error": str(e)}

    return json.dumps(
        {"comparison": results, "note": "시나리오별 ATE 및 판결 비교"},
        indent=2,
        ensure_ascii=False,
    )

@mcp.tool()
def run_drift_check() -> str:
    """Causal Drift 탐지를 1회 실행합니다.

    파이프라인을 실행하고, 이전 결과 대비 ATE/CATE 변동을 감지합니다.
    """
    try:
        from engine.monitoring import MonitoringScheduler

        scheduler = MonitoringScheduler(config=config, scenario="A")
        result = scheduler.run_once()

        output = {
            "drifted": result.drifted,
            "metric": result.metric,
            "score": round(result.score, 4),
            "threshold": result.threshold,
            "recommendation": (
                "🚨 드리프트 감지! 원인 분석 필요." if result.drifted
                else "✅ 안정 상태."
            ),
        }
        return json.dumps(output, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"드리프트 체크 실패: {str(e)}"


@mcp.tool()
def get_monitoring_status() -> str:
    """현재 모니터링 시스템 상태를 반환합니다."""
    try:
        from engine.monitoring import MonitoringScheduler

        scheduler = MonitoringScheduler(config=config)
        return json.dumps(scheduler.status, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"상태 조회 실패: {str(e)}"


if __name__ == "__main__":
    # stdio 모드로 서버 실행
    mcp.run()
