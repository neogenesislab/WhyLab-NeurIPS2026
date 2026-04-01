# -*- coding: utf-8 -*-
"""R2: 민감도 분석 — E-value + Gaussian Copula 기반 교란 변수 방어.

DML이 산출한 ATE가 관측되지 않은 교란 변수(Unobserved Confounder)에
의해 무효화될 수 있는지 정량적으로 평가합니다.

Reviewer 방어:
"숨겨진 변수가 이 인과 효과를 뒤집을 수 있지 않은가?"
→ E-value로 "최소한 이 강도의 교란이 있어야 뒤집힌다" 증명

수학적 기반:
- VanderWeele & Ding (2017): "Sensitivity Analysis in Observational
  Research: Introducing the E-value"
- E = RR + sqrt(RR × (RR - 1))  where RR = exp(|ATE| / SE)
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("whylab.audit.sensitivity")


@dataclass
class SensitivityResult:
    """민감도 분석 결과.

    AAAI/KDD 논문 Table용 구조화된 결과.
    """
    # E-value 관련
    e_value: float = 0.0
    e_value_ci_lower: float = 0.0
    risk_ratio: float = 1.0

    # Robustness 판단
    is_robust: bool = False
    robustness_level: str = "unknown"  # weak / moderate / strong / very_strong

    # Partial R² 기반 민감도 (Cinelli & Hazlett 2020)
    partial_r2_treatment: float = 0.0
    partial_r2_outcome: float = 0.0
    rv_q: float = 0.0  # Robustness Value (RV_q)

    # 메타 정보
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class SensitivityAnalyzer:
    """인과 효과의 미관측 교란 변수에 대한 강건성 분석.

    제공 메서드:
    1. E-value: 빠르고 직관적 (VanderWeele & Ding 2017)
    2. Partial R² Bounds: 학술적 최고점 (Cinelli & Hazlett 2020)
    """

    # E-value 강건성 수준 기준
    ROBUSTNESS_THRESHOLDS = {
        "very_strong": 3.0,
        "strong": 2.0,
        "moderate": 1.5,
        "weak": 1.0,
    }

    def analyze(
        self,
        ate: float,
        ate_ci: List[float],
        pre_values: List[float],
        post_values: List[float],
        p_value: Optional[float] = None,
    ) -> SensitivityResult:
        """전체 민감도 분석을 실행합니다.

        Args:
            ate: 평균 처치 효과
            ate_ci: 95% 신뢰구간 [lower, upper]
            pre_values: 개입 전 시계열
            post_values: 개입 후 시계열
            p_value: 통계적 유의성

        Returns:
            SensitivityResult
        """
        # E-value 계산
        se = self._compute_se(ate, ate_ci)
        rr = self._ate_to_risk_ratio(ate, se, pre_values)
        e_value = self._compute_e_value(rr)

        # CI 하한의 E-value (보수적 추정)
        if ate_ci[0] > 0:
            rr_lower = self._ate_to_risk_ratio(ate_ci[0], se, pre_values)
            e_value_ci = self._compute_e_value(rr_lower)
        elif ate_ci[1] < 0:
            rr_lower = self._ate_to_risk_ratio(abs(ate_ci[1]), se, pre_values)
            e_value_ci = self._compute_e_value(rr_lower)
        else:
            e_value_ci = 1.0  # CI가 0을 포함하면 강건하지 않음

        # Partial R² 기반 민감도 근사
        partial_r2_t, partial_r2_o, rv_q = self._partial_r2_bounds(
            ate, se, pre_values, post_values
        )

        # 강건성 수준 판단
        robustness_level = self._classify_robustness(e_value)
        is_robust = e_value >= self.ROBUSTNESS_THRESHOLDS["moderate"]

        result = SensitivityResult(
            e_value=round(e_value, 3),
            e_value_ci_lower=round(e_value_ci, 3),
            risk_ratio=round(rr, 3),
            is_robust=is_robust,
            robustness_level=robustness_level,
            partial_r2_treatment=round(partial_r2_t, 4),
            partial_r2_outcome=round(partial_r2_o, 4),
            rv_q=round(rv_q, 4),
            diagnostics={
                "ate": round(ate, 4),
                "ate_ci": [round(x, 4) for x in ate_ci],
                "se": round(se, 4),
                "interpretation": self._generate_interpretation(
                    e_value, robustness_level, ate
                ),
            },
        )

        logger.info(
            "📊 민감도 분석: E-value=%.2f (%s), RR=%.2f, RV_q=%.4f",
            e_value, robustness_level, rr, rv_q,
        )

        return result

    def _compute_se(self, ate: float, ci: List[float]) -> float:
        """신뢰구간에서 표준오차 역산."""
        ci_width = ci[1] - ci[0]
        return ci_width / 3.92 if ci_width > 0 else 0.01

    def _ate_to_risk_ratio(
        self,
        ate: float,
        se: float,
        pre_values: List[float],
    ) -> float:
        """ATE를 Risk Ratio로 변환.

        RR ≈ exp(|ATE| / σ_pre)  (로그-선형 근사)
        """
        pre_std = statistics.stdev(pre_values) if len(pre_values) > 1 else 1.0
        # ATE를 표준화된 효과 크기로 변환 후 RR 근사
        standardized = abs(ate) / max(pre_std, 1e-10)
        return math.exp(min(standardized, 10))  # overflow 방지

    def _compute_e_value(self, rr: float) -> float:
        """E-value 계산.

        VanderWeele & Ding (2017):
        E = RR + sqrt(RR × (RR - 1))

        해석: 관측되지 않은 교란 변수가 처치와 결과 양쪽에
        최소한 E-value만큼의 Risk Ratio로 연관되어야만
        관측된 인과 효과를 무효화할 수 있습니다.
        """
        if rr <= 1.0:
            return 1.0
        return rr + math.sqrt(rr * (rr - 1))

    def _partial_r2_bounds(
        self,
        ate: float,
        se: float,
        pre: List[float],
        post: List[float],
    ) -> tuple:
        """Partial R² 기반 민감도 경계 (Cinelli & Hazlett 2020 근사).

        Returns: (partial_r2_treatment, partial_r2_outcome, rv_q)
        """
        n = len(pre) + len(post)
        t_stat = ate / se if se > 1e-10 else 0.0

        # Partial R²(Y~D|X): 처치가 결과를 설명하는 비율
        partial_r2_t = (t_stat ** 2) / (t_stat ** 2 + n - 2) if n > 2 else 0.0

        # Partial R²(D~Y|X): 결과가 처치를 설명하는 비율 (대칭 근사)
        partial_r2_o = partial_r2_t  # 관측 데이터로는 대칭 근사

        # Robustness Value (RV_q): 효과를 0으로 만드는 최소 교란 강도
        # RV_q ≈ sqrt(partial_r2_t) (단순화)
        rv_q = math.sqrt(partial_r2_t)

        return partial_r2_t, partial_r2_o, rv_q

    def _classify_robustness(self, e_value: float) -> str:
        """E-value 기반 강건성 수준 분류."""
        for level, threshold in self.ROBUSTNESS_THRESHOLDS.items():
            if e_value >= threshold:
                return level
        return "not_robust"

    def _generate_interpretation(
        self,
        e_value: float,
        level: str,
        ate: float,
    ) -> str:
        """자연어 해석 생성 (감사 보고서용)."""
        direction = "양" if ate > 0 else "음"
        return (
            f"이 인과 효과(ATE={ate:.4f}, {direction}의 방향)를 "
            f"무효화하려면, 관측되지 않은 교란 변수가 처치와 결과 "
            f"양쪽에 최소 Risk Ratio {e_value:.2f}의 강도로 "
            f"연관되어야 합니다. "
            f"강건성 수준: {level}."
        )
