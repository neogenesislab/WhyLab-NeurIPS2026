# -*- coding: utf-8 -*-
"""CounterfactualCell — 구조적 반사실 추론 (SCM 기반 What-if).

Pearl 인과 사다리 3단계 "만약 ~했다면?"에 대응하는
구조적 반사실(Structural Counterfactual) 분석을 제공합니다.

- **개별 반사실**: 특정 관측치에 대해 "처치를 받았/안 받았다면?"
- **분포 반사실**: 처치 분포 변경 시 결과 분포 변화 추정
- **경계 분석**: 반사실 효과의 Sharp 경계 산출

Phase 10-3: 구조적 반사실.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig

logger = logging.getLogger(__name__)


@dataclass
class IndividualCounterfactual:
    """개별 반사실 결과."""
    index: int
    factual_treatment: float
    factual_outcome: float
    counterfactual_treatment: float
    counterfactual_outcome: float
    individual_effect: float  # Y(1) - Y(0) 또는 Y(0) - Y(1)


@dataclass
class CounterfactualSummary:
    """반사실 분석 요약."""
    method: str = "structural_counterfactual"
    n_individuals: int = 0
    mean_ite: float = 0.0  # 개별 처치 효과 평균
    median_ite: float = 0.0
    std_ite: float = 0.0
    positive_effect_ratio: float = 0.0  # 양의 효과를 받은 비율
    top_beneficiaries: List[Dict[str, Any]] = field(default_factory=list)
    top_harmed: List[Dict[str, Any]] = field(default_factory=list)
    distribution_shift: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""


class CounterfactualCell(BaseCell):
    """구조적 반사실 추론 셀.

    CATE 추정 결과를 기반으로 개별 반사실(Individual Counterfactual)을
    계산하고, what-if 시나리오 분석을 수행합니다.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="counterfactual_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """반사실 분석 실행.

        Args:
            inputs: 파이프라인 컨텍스트 (CATE 추정 결과 포함 필요).

        Returns:
            반사실 분석 결과 추가.
        """
        df = inputs.get("dataframe")
        if df is None:
            self.logger.warning("데이터프레임 없음 → Counterfactual 건너뜀")
            return inputs

        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")
        feature_names = inputs.get("feature_names", [])

        # CATE 추정치 확인 (MetaLearnerCell 또는 CausalCell에서 생성)
        cate_values = inputs.get("cate_values")
        ate_value = inputs.get("ate")

        # ATE 값 정규화
        if isinstance(ate_value, dict):
            ate = float(ate_value.get("value", 0.0))
        elif isinstance(ate_value, (int, float)):
            ate = float(ate_value)
        else:
            ate = 0.0

        # CATE가 없으면 ATE 기반 간이 반사실
        if cate_values is None:
            self.logger.info("📐 CATE 없음 → ATE 기반 간이 반사실 분석")
            cate_values = np.full(len(df), ate)

        cate_array = np.array(cate_values)

        # 1. 개별 반사실 계산
        self.logger.info("🔄 개별 반사실 계산 (n=%d)", len(df))
        cf_results = self._compute_individual_counterfactuals(
            df, treatment_col, outcome_col, cate_array,
        )

        # 2. 분포 반사실
        dist_shift = self._distribution_counterfactual(
            df, treatment_col, outcome_col, cate_array,
        )

        # 3. 경계 분석 (Sharp bounds)
        bounds = self._compute_bounds(
            df, treatment_col, outcome_col, cate_array,
        )

        # 4. 요약 생성
        summary = self._build_summary(cf_results, dist_shift, bounds)

        self.logger.info(
            "✅ 반사실 분석 완료: 평균 ITE=%.4f, 양의 효과 비율=%.1f%%",
            summary.mean_ite, summary.positive_effect_ratio * 100,
        )

        return {
            **inputs,
            "counterfactual": {
                "summary": self._serialize_summary(summary),
                "individual_effects": cate_array.tolist(),
            },
        }

    def _compute_individual_counterfactuals(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        cate: np.ndarray,
    ) -> List[IndividualCounterfactual]:
        """각 관측치에 대한 반사실 결과(Outcome)를 계산합니다.

        - 처치를 받은 개체: Y(0) = Y_observed - CATE
        - 처치를 안 받은 개체: Y(1) = Y_observed + CATE
        """
        results = []

        if treatment_col not in df.columns or outcome_col not in df.columns:
            return results

        treatments = df[treatment_col].values
        outcomes = df[outcome_col].values

        for i in range(min(len(df), len(cate))):
            t = float(treatments[i])
            y = float(outcomes[i])
            ite = float(cate[i])

            if t == 1:
                # 처치를 받음 → "안 받았다면?" 반사실
                cf_outcome = y - ite
                cf_treatment = 0.0
            else:
                # 처치를 안 받음 → "받았다면?" 반사실
                cf_outcome = y + ite
                cf_treatment = 1.0

            results.append(IndividualCounterfactual(
                index=i,
                factual_treatment=t,
                factual_outcome=y,
                counterfactual_treatment=cf_treatment,
                counterfactual_outcome=cf_outcome,
                individual_effect=ite,
            ))

        return results

    def _distribution_counterfactual(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        cate: np.ndarray,
    ) -> Dict[str, float]:
        """처치 분포 변경 시 결과 분포 변화 추정.

        "모두 처치" vs "모두 미처치" 시나리오 비교.
        """
        if outcome_col not in df.columns:
            return {}

        outcomes = df[outcome_col].values
        n = min(len(outcomes), len(cate))

        # 현재 평균
        current_mean = float(np.mean(outcomes[:n]))

        # 모두 처치 시나리오: Y(1) = Y + CATE * (1 - T)
        treatments = df[treatment_col].values[:n] if treatment_col in df.columns else np.zeros(n)
        all_treated = outcomes[:n] + cate[:n] * (1 - treatments)
        all_treated_mean = float(np.mean(all_treated))

        # 모두 미처치 시나리오: Y(0) = Y - CATE * T
        all_control = outcomes[:n] - cate[:n] * treatments
        all_control_mean = float(np.mean(all_control))

        return {
            "current_mean": current_mean,
            "all_treated_mean": all_treated_mean,
            "all_control_mean": all_control_mean,
            "gain_from_universal_treatment": all_treated_mean - current_mean,
            "loss_from_no_treatment": current_mean - all_control_mean,
        }

    def _compute_bounds(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        cate: np.ndarray,
    ) -> Dict[str, float]:
        """반사실 효과의 Sharp 경계 (Manski bounds) 산출."""
        if outcome_col not in df.columns:
            return {}

        outcomes = df[outcome_col].values
        n = min(len(outcomes), len(cate))

        y_min = float(np.min(outcomes[:n]))
        y_max = float(np.max(outcomes[:n]))

        # Manski bounds: 결과 변수의 범위를 기반
        if treatment_col in df.columns:
            treatments = df[treatment_col].values[:n]
            p_treated = float(np.mean(treatments))
            p_control = 1 - p_treated

            if p_treated > 0 and p_control > 0:
                e_y1 = float(np.mean(outcomes[:n][treatments == 1])) if np.any(treatments == 1) else 0
                e_y0 = float(np.mean(outcomes[:n][treatments == 0])) if np.any(treatments == 0) else 0

                # Manski worst case bounds
                lower_bound = e_y1 - y_max  # E[Y(1)] - Y_max(0)
                upper_bound = e_y1 - y_min  # E[Y(1)] - Y_min(0)

                return {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "point_estimate": float(np.mean(cate[:n])),
                    "width": upper_bound - lower_bound,
                    "informative": (upper_bound - lower_bound) < (y_max - y_min),
                }

        return {
            "lower_bound": float(np.percentile(cate[:n], 2.5)),
            "upper_bound": float(np.percentile(cate[:n], 97.5)),
            "point_estimate": float(np.mean(cate[:n])),
        }

    def _build_summary(
        self,
        cf_results: List[IndividualCounterfactual],
        dist_shift: Dict[str, float],
        bounds: Dict[str, float],
    ) -> CounterfactualSummary:
        """반사실 분석 결과를 요약합니다."""
        if not cf_results:
            return CounterfactualSummary(interpretation="반사실 계산 결과 없음")

        effects = [r.individual_effect for r in cf_results]
        effects_arr = np.array(effects)

        positive_ratio = float(np.mean(effects_arr > 0))

        # 상위 수혜자 / 피해자
        sorted_results = sorted(cf_results, key=lambda r: r.individual_effect, reverse=True)
        top_beneficiaries = [
            {"index": r.index, "effect": r.individual_effect,
             "factual_y": r.factual_outcome, "cf_y": r.counterfactual_outcome}
            for r in sorted_results[:5]
        ]
        top_harmed = [
            {"index": r.index, "effect": r.individual_effect,
             "factual_y": r.factual_outcome, "cf_y": r.counterfactual_outcome}
            for r in sorted_results[-5:]
        ]

        mean_ite = float(np.mean(effects_arr))
        std_ite = float(np.std(effects_arr))
        interp = (
            f"반사실 분석 완료 (n={len(cf_results)}). "
            f"평균 ITE={mean_ite:.4f} (±{std_ite:.4f}). "
            f"양의 효과 비율={positive_ratio:.1%}."
        )

        return CounterfactualSummary(
            n_individuals=len(cf_results),
            mean_ite=mean_ite,
            median_ite=float(np.median(effects_arr)),
            std_ite=std_ite,
            positive_effect_ratio=positive_ratio,
            top_beneficiaries=top_beneficiaries,
            top_harmed=top_harmed,
            distribution_shift=dist_shift,
            bounds=bounds,
            interpretation=interp,
        )

    def _serialize_summary(self, summary: CounterfactualSummary) -> Dict[str, Any]:
        """요약을 JSON 직렬화."""
        d = {}
        for k, v in summary.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, (np.integer, np.floating)):
                d[k] = v.item()
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d
