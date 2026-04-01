# -*- coding: utf-8 -*-
"""DiscoveryCell — 인과 구조 자동 발견 파이프라인 셀.

DiscoveryAgent를 활용하여 데이터로부터 인과 그래프(DAG)를 자동으로 발견합니다.
Orchestrator 파이프라인의 최전방에 위치하여, DataCell 다음에 실행됩니다.

Phase 9-2: PC 알고리즘 + LLM 하이브리드 인과 발견.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from engine.cells.base_cell import BaseCell
from engine.agents.discovery import DiscoveryAgent
from engine.config import WhyLabConfig

logger = logging.getLogger(__name__)


class DiscoveryCell(BaseCell):
    """인과 구조(DAG) 자동 발견 셀.

    DataCell의 출력을 받아 treatment/outcome/confounder/DAG를
    자동으로 탐색합니다. 이미 사용자가 지정한 경우 재발견 없이
    DAG만 보강합니다.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="discovery_cell", config=config)
        self.agent = DiscoveryAgent(config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """인과 구조 발견 실행.

        Args:
            inputs: DataCell 출력 (dataframe, feature_names 등).

        Returns:
            기존 inputs + dag_edges, discovered_roles 추가.
        """
        df = inputs.get("dataframe")
        if df is None:
            self.logger.warning("데이터프레임 없음 → Discovery 건너뜀")
            return inputs

        treatment_col = inputs.get("treatment_col")
        outcome_col = inputs.get("outcome_col")
        feature_names = inputs.get("feature_names", [])

        # 이미 treatment/outcome이 지정된 경우 → DAG만 발견
        if treatment_col and outcome_col:
            self.logger.info(
                "🔍 DAG 발견 모드 (T=%s, Y=%s 고정)", treatment_col, outcome_col
            )
            metadata = {
                "feature_names": feature_names,
                "treatment_col": treatment_col,
                "outcome_col": outcome_col,
            }
            dag = self.agent.discover(df, metadata)
            dag_edges = list(dag.edges())

            return {
                **inputs,
                "dag_edges": dag_edges,
                "discovery_mode": "dag_only",
                "dag_nodes": list(dag.nodes()),
                "dag_edge_count": len(dag_edges),
            }

        # treatment/outcome 미지정 → 전체 자동 발견
        self.logger.info("🔍 전체 자동 발견 모드 (Auto-Discovery)")
        roles = self.agent.auto_discover(df)

        discovered_treatment = roles.get("treatment", treatment_col)
        discovered_outcome = roles.get("outcome", outcome_col)
        discovered_confounders = roles.get("confounders", feature_names)
        dag_edges = roles.get("dag", [])

        self.logger.info(
            "✅ Auto-Discovery 완료: T=%s, Y=%s, 교란변수 %d개, DAG 엣지 %d개",
            discovered_treatment, discovered_outcome,
            len(discovered_confounders), len(dag_edges),
        )

        return {
            **inputs,
            "treatment_col": discovered_treatment,
            "outcome_col": discovered_outcome,
            "feature_names": discovered_confounders,
            "dag_edges": dag_edges,
            "discovery_mode": "auto",
            "discovered_roles": {
                "treatment": discovered_treatment,
                "outcome": discovered_outcome,
                "confounders": discovered_confounders,
                "reasoning": roles.get("reasoning", ""),
            },
        }
