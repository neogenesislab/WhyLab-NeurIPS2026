# -*- coding: utf-8 -*-
"""WhyLab 간편 API — 3줄 코드로 인과 분석.

사용법:
    import whylab

    # CSV 파일로 분석
    result = whylab.analyze("data.csv", treatment="T", outcome="Y")
    result.summary()       # 텍스트 요약
    result.verdict          # AI Debate 판결

    # pandas DataFrame으로 분석
    result = whylab.analyze(df, treatment="T", outcome="Y")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("whylab")


@dataclass
class CausalResult:
    """인과 분석 결과 컨테이너.

    Attributes:
        ate: 평균 처치 효과 (Average Treatment Effect).
        ate_ci: ATE 95% 신뢰구간 (lower, upper).
        cate: 조건부 평균 처치 효과 벡터 (CATE).
        verdict: AI Debate 판결 ("CAUSAL" | "NOT_CAUSAL" | "UNCERTAIN").
        confidence: 판결 확신도 (0.0 ~ 1.0).
        meta_learners: 메타러너별 결과.
        sensitivity: 민감도 분석 결과.
        raw: 전체 파이프라인 원시 결과.
    """

    ate: float = 0.0
    ate_ci: tuple = (0.0, 0.0)
    cate: Optional[np.ndarray] = None
    verdict: str = "UNCERTAIN"
    confidence: float = 0.0
    meta_learners: Dict[str, Any] = field(default_factory=dict)
    sensitivity: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """핵심 결과를 한눈에 보여주는 텍스트 요약."""
        lines = [
            "=" * 60,
            "  WhyLab Causal Analysis Result",
            "=" * 60,
            f"  ATE (Average Treatment Effect): {self.ate:.4f}",
            f"  95% CI: [{self.ate_ci[0]:.4f}, {self.ate_ci[1]:.4f}]",
            f"  AI Verdict: {self.verdict} (confidence: {self.confidence:.1%})",
            "-" * 60,
        ]

        # 메타러너 결과
        if self.meta_learners:
            lines.append("  Meta-Learner Results:")
            for name, result in self.meta_learners.items():
                ate = result.get("ate", "N/A")
                lines.append(f"    {name}: ATE = {ate:.4f}" if isinstance(ate, float) else f"    {name}: {ate}")
            lines.append("-" * 60)

        # 민감도
        if self.sensitivity:
            e_val = self.sensitivity.get("e_value", {}).get("e_value", "N/A")
            placebo = self.sensitivity.get("placebo_test", {}).get("status", "N/A")
            lines.append(f"  E-value: {e_val}")
            lines.append(f"  Placebo Test: {placebo}")
            lines.append("-" * 60)

        lines.append("=" * 60)
        text = "\n".join(lines)
        print(text)
        return text


def analyze(
    data: Union[str, Path, pd.DataFrame],
    treatment: str = "treatment",
    outcome: str = "outcome",
    features: Optional[List[str]] = None,
    scenario: Optional[str] = None,
    debate: bool = True,
) -> CausalResult:
    """인과 분석을 수행합니다.

    Args:
        data: CSV 파일 경로 또는 pandas DataFrame. None이면 합성 데이터 사용.
        treatment: 처치 변수 컬럼명.
        outcome: 결과 변수 컬럼명.
        features: 공변량 컬럼 리스트 (None이면 자동 추론).
        scenario: 합성 데이터 시나리오 ("A" 또는 "B"). data가 None일 때만 사용.
        debate: AI Debate 활성화 여부.

    Returns:
        CausalResult: 인과 분석 결과.

    Examples:
        >>> import whylab
        >>> result = whylab.analyze("data.csv", treatment="T", outcome="Y")
        >>> print(result.verdict)
        'CAUSAL'
    """
    from engine.config import WhyLabConfig
    from engine.orchestrator import Orchestrator

    config = WhyLabConfig()

    # 데이터 소스 설정
    if data is not None:
        if isinstance(data, (str, Path)):
            config.data.input_path = str(data)
        elif isinstance(data, pd.DataFrame):
            # DataFrame을 임시 CSV로 저장
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
            data.to_csv(tmp.name, index=False)
            config.data.input_path = tmp.name

        config.data.treatment_col = treatment
        config.data.outcome_col = outcome
        if features:
            config.data.feature_cols = features

    # 파이프라인 실행
    orchestrator = Orchestrator(config=config)
    sc = scenario or "A"

    try:
        raw = orchestrator.run_pipeline(scenario=sc)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        return CausalResult(verdict="ERROR", raw={"error": str(e)})

    # 결과 추출
    result = CausalResult(raw=raw)

    # ATE
    ate_info = raw.get("ate", {})
    if isinstance(ate_info, dict):
        result.ate = ate_info.get("point_estimate", 0.0)
        result.ate_ci = (
            ate_info.get("ci_lower", 0.0),
            ate_info.get("ci_upper", 0.0),
        )
    elif isinstance(ate_info, (int, float)):
        result.ate = float(ate_info)

    # CATE
    result.cate = raw.get("cate_predictions")

    # 메타러너 결과
    ml = raw.get("meta_learners", {})
    if ml:
        result.meta_learners = {
            name: {"ate": info.get("ate", 0.0)}
            for name, info in ml.items()
            if isinstance(info, dict) and "ate" in info
        }

    # 민감도
    result.sensitivity = raw.get("sensitivity", {})

    # Debate 결과
    debate_result = raw.get("debate", {})
    if debate_result:
        result.verdict = debate_result.get("verdict", "UNCERTAIN")
        result.confidence = debate_result.get("confidence", 0.0)

    return result
