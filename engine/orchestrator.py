# -*- coding: utf-8 -*-
"""Orchestrator — 셀 간 데이터 흐름 조율자.

모든 셀을 초기화하고, DAG 순서에 따라 순차적으로 실행하며,
전체 파이프라인의 상태를 관리합니다.

책임:
    1. 전역 설정(Config) 로드 및 주입
    2. 셀 인스턴스화
    3. 실행 순서 제어 (Data → Causal → Explain → Viz → Export)
    4. 예외 처리 및 로깅
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from engine.config import WhyLabConfig
from engine.cells.data_cell import DataCell
from engine.cells.discovery_cell import DiscoveryCell
from engine.cells.auto_causal_cell import AutoCausalCell
from engine.cells.causal_cell import CausalCell
from engine.cells.explain_cell import ExplainCell
from engine.cells.viz_cell import VizCell
from engine.cells.export_cell import ExportCell
from engine.cells.report_cell import ReportCell
from engine.cells.sensitivity_cell import SensitivityCell
from engine.cells.refutation_cell import RefutationCell
from engine.cells.meta_learner_cell import MetaLearnerCell
from engine.cells.conformal_cell import ConformalCell
from engine.cells.debate_cell import DebateCell
from engine.cells.quasi_experimental_cell import QuasiExperimentalCell
from engine.cells.temporal_causal_cell import TemporalCausalCell
from engine.cells.counterfactual_cell import CounterfactualCell

class Orchestrator:
    """WhyLab 엔진 오케스트레이터."""

    def __init__(self, config: WhyLabConfig = None) -> None:
        self.config = config if config else WhyLabConfig()
        self.logger = logging.getLogger("whylab.orchestrator")
        
        # 디렉토리 초기화
        self.config.paths.ensure_dirs()

        # 셀 초기화 (의존성 주입) — 16셀 파이프라인
        self.cells = {
            "data": DataCell(self.config),
            "discovery": DiscoveryCell(self.config),            # 인과 구조 자동 발견
            "auto_causal": AutoCausalCell(self.config),         # 방법론 자동 추천
            "causal": CausalCell(self.config),
            "meta_learner": MetaLearnerCell(self.config),       # 5종 메타러너
            "conformal": ConformalCell(self.config),             # 분포무가정 CI
            "explain": ExplainCell(self.config),
            "refutation": RefutationCell(self.config),           # 진짜 반증
            "sensitivity": SensitivityCell(self.config),
            "quasi_experimental": QuasiExperimentalCell(self.config),  # IV/DiD/RDD
            "temporal_causal": TemporalCausalCell(self.config),       # 시계열 인과
            "counterfactual": CounterfactualCell(self.config),        # 구조적 반사실
            "viz": VizCell(self.config),
            "debate": DebateCell(self.config),                   # 3-에이전트 LLM 판결
            "export": ExportCell(self.config),
            "report": ReportCell(self.config),
        }

    def run_pipeline(self, scenario: str = "A") -> Dict[str, Any]:
        """시나리오에 따라 전체 파이프라인을 실행합니다.

        Args:
            scenario: "A" (신용한도) 또는 "B" (마케팅).

        Returns:
            최종 실행 결과 딕셔너리.
        """
        self.logger.info("[PIPELINE] Starting (scenario: %s)", scenario)
        
        context: Dict[str, Any] = {"scenario_name": f"Scenario {scenario}", "scenario": scenario}

        # 실행 순서 (DAG) — 16셀 파이프라인
        # Data → Discovery → AutoCausal → Causal → MetaLearner → Conformal →
        # Explain → Refutation → Sensitivity → QE → Temporal → Counterfactual →
        # Viz → Debate → Export → Report
        pipeline_sequence = [
            "data", "discovery", "auto_causal",
            "causal", "meta_learner", "conformal", "explain",
            "refutation", "sensitivity",
            "quasi_experimental", "temporal_causal", "counterfactual",
            "viz", "debate", "export", "report",
        ]

        try:
            for cell_name in pipeline_sequence:
                cell = self.cells[cell_name]
                self.logger.info("[CELL] Running: %s", cell.name)
                
                # 이전 단계의 출력을 다음 단계의 입력으로 병합
                output = cell.run(context)
                if output is not None:
                    context.update(output)

            self.logger.info("[PIPELINE] Completed successfully")
            return context

        except Exception as e:
            self.logger.error("[PIPELINE] Aborted: %s", e)
            raise e

    def get_status(self) -> Dict[str, Any]:
        """모든 셀의 현재 상태를 반환합니다."""
        return {
            name: {"state": cell.state, "name": cell.name}
            for name, cell in self.cells.items()
        }
