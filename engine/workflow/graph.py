# -*- coding: utf-8 -*-
"""WhyLab Cytoplasm — Agents Workflow using LangGraph.

세포질(Cytoplasm) 역할을 하는 상태 관리 워크플로우입니다.
에이전트들이 협력하여 가설 수립 -> 추정 -> 반증의 순환 과정을 수행합니다.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Union
import operator
from langgraph.graph import StateGraph, START, END

# 타입 정의 (Agent State)
class AgentState(TypedDict):
    scenario: str
    data_summary: str
    dag_structure: List[str]  # e.g. ["A->B", "B->C"]
    causal_effect: float      # Estimated ATE
    refutation_result: bool   # Pass/Fail
    history: Annotated[List[str], operator.add]

# ─────────────────────────────────────────────────────────────
# Nodes (세포 소기관 역할)
# ─────────────────────────────────────────────────────────────

def discovery_node(state: AgentState) -> Dict[str, Any]:
    """Discovery Agent: 데이터를 분석하여 인과 구조(DAG)를 발견합니다."""
    print("   [Discovery] Analyzing data & building causal graph...")
    
    from engine.config import WhyLabConfig
    from engine.cells.data_cell import DataCell
    from engine.agents.discovery import DiscoveryAgent

    config = WhyLabConfig()
    data_out = DataCell(config).execute({"scenario": state["scenario"]})
    agent = DiscoveryAgent(config)
    graph = agent.discover(data_out["dataframe"], data_out)

    dag = [f"{u} -> {v}" for u, v in graph.edges()]

    return {
        "dag_structure": dag,
        "history": ["Discovery Completed"]
    }

def estimation_node(state: AgentState) -> Dict[str, Any]:
    """Estimation Agent: 발견된 DAG를 바탕으로 인과 효과를 추정합니다."""
    print(f"   [Estimation] Estimating effect based on DAG: {state['dag_structure']}")
    
    from engine.config import WhyLabConfig
    from engine.cells.data_cell import DataCell
    from engine.cells.causal_cell import CausalCell

    config = WhyLabConfig()
    data_out = DataCell(config).execute({"scenario": state["scenario"]})
    causal_out = CausalCell(config).execute(data_out)
    effect = causal_out["ate"]
    
    return {
        "causal_effect": effect,
        "history": ["Estimation Completed"]
    }

def refutation_node(state: AgentState) -> Dict[str, Any]:
    """Refutation Agent: 추정된 효과를 실제 반증합니다.

    RefutationCell을 호출하여 진짜 Placebo Test, Bootstrap CI,
    Leave-One-Out Confounder, Subset Validation을 수행합니다.
    """
    print(f"   [Refutation] Testing robustness of effect: {state['causal_effect']}")

    from engine.config import WhyLabConfig
    from engine.cells.data_cell import DataCell
    from engine.cells.causal_cell import CausalCell
    from engine.cells.refutation_cell import RefutationCell

    config = WhyLabConfig()
    data_out = DataCell(config).execute({"scenario": state["scenario"]})
    causal_out = CausalCell(config).execute(data_out)
    refutation_out = RefutationCell(config).execute(causal_out)

    refutation_results = refutation_out.get("refutation_results", {})
    overall = refutation_results.get("overall", {})
    is_robust = overall.get("status", "Fail") == "Pass"

    return {
        "refutation_result": is_robust,
        "history": [
            f"Refutation {'Passed' if is_robust else 'Failed'}: "
            f"{overall.get('pass_count', 0)}/{overall.get('total', 0)} tests"
        ],
    }


# ─────────────────────────────────────────────────────────────
# Conditional Edges (항상성 유지)
# ─────────────────────────────────────────────────────────────

def check_robustness(state: AgentState) -> str:
    """반증 결과에 따라 다음 단계 결정."""
    if state["refutation_result"]:
        print("   [Homeostasis] Robustness Verified.")
        return "end"
    else:
        print("   [Homeostasis] Robustness Failed! Retrying Discovery.")
        return "retry"

# ─────────────────────────────────────────────────────────────
# Graph Construction (세포질 형성)
# ─────────────────────────────────────────────────────────────

def build_graph():
    """LangGraph 워크플로우를 생성합니다."""
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("discovery", discovery_node)
    workflow.add_node("estimation", estimation_node)
    workflow.add_node("refutation", refutation_node)

    # 엣지 연결
    workflow.add_edge(START, "discovery")
    workflow.add_edge("discovery", "estimation")
    workflow.add_edge("estimation", "refutation")
    
    # 조건부 엣지 (Feedback Loop)
    workflow.add_conditional_edges(
        "refutation",
        check_robustness,
        {
            "end": END,
            "retry": "discovery"
        }
    )

    return workflow.compile()
