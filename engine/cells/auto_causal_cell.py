# -*- coding: utf-8 -*-
"""AutoCausalCell — 데이터 특성 기반 자동 방법론 선택.

데이터를 프로파일링하여 최적의 인과추론 방법론을 자동 추천합니다.
Orchestrator에서 DataCell → DiscoveryCell → **AutoCausalCell** → CausalCell
순서로 실행되어, CausalCell에 최적 설정을 주입합니다.

Phase 9-3: AutoCausal 파이프라인.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 데이터 프로파일
# ──────────────────────────────────────────────

@dataclass
class DataProfile:
    """데이터셋 자동 프로파일 결과."""
    n_samples: int
    n_features: int
    treatment_type: str  # "binary" | "continuous" | "multi_level"
    outcome_type: str    # "binary" | "continuous"
    has_missing: bool
    missing_ratio: float
    treatment_balance: float  # 처치/통제 비율 (이진) 또는 분산 (연속)
    overlap_risk: str    # "low" | "medium" | "high"
    linearity_score: float  # 선형성 정도 (0~1)
    confounders_count: int
    warnings: List[str]


class AutoCausalCell(BaseCell):
    """데이터 프로파일링 → 방법론 자동 추천 셀.

    데이터 특성을 분석하여:
    1. 최적 메타러너 추천
    2. 추정 방법(DML/Forest/IV) 추천
    3. 위험 요소 경고 (샘플 크기, Overlap, SUTVA 등)
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="auto_causal_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 프로파일링 + 방법론 추천.

        Args:
            inputs: DataCell/DiscoveryCell 출력.

        Returns:
            기존 inputs + data_profile, recommended_method 추가.
        """
        df = inputs.get("dataframe")
        if df is None:
            self.logger.warning("데이터프레임 없음 → AutoCausal 건너뜀")
            return inputs

        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")
        feature_names = inputs.get("feature_names", [])

        # 1단계: 데이터 프로파일링
        profile = self._profile_data(df, treatment_col, outcome_col, feature_names)
        self.logger.info(
            "📊 데이터 프로파일: n=%d, T=%s(%s), Y=%s(%s), 교란=%d",
            profile.n_samples, treatment_col, profile.treatment_type,
            outcome_col, profile.outcome_type, profile.confounders_count,
        )

        # 2단계: 방법론 추천
        recommendation = self._recommend_method(profile)
        self.logger.info(
            "🎯 추천 방법론: %s (모델: %s, 이유: %s)",
            recommendation["primary_method"],
            recommendation["nuisance_model"],
            recommendation["reasoning"],
        )

        # 3단계: 경고 출력
        for warning in profile.warnings:
            self.logger.warning("⚠️ %s", warning)

        return {
            **inputs,
            "data_profile": {
                "n_samples": profile.n_samples,
                "n_features": profile.n_features,
                "treatment_type": profile.treatment_type,
                "outcome_type": profile.outcome_type,
                "has_missing": profile.has_missing,
                "missing_ratio": profile.missing_ratio,
                "treatment_balance": profile.treatment_balance,
                "overlap_risk": profile.overlap_risk,
                "linearity_score": profile.linearity_score,
                "warnings": profile.warnings,
            },
            "auto_recommendation": recommendation,
        }

    def _profile_data(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_names: List[str],
    ) -> DataProfile:
        """데이터셋을 자동으로 프로파일링합니다."""
        n_samples = len(df)
        n_features = len(feature_names)
        warnings = []

        # Treatment 유형 판별
        if treatment_col in df.columns:
            t_nunique = df[treatment_col].nunique()
            if t_nunique <= 2:
                treatment_type = "binary"
            elif t_nunique <= 10:
                treatment_type = "multi_level"
            else:
                treatment_type = "continuous"

            # 처치 균형
            if treatment_type == "binary":
                t_ratio = df[treatment_col].mean()
                treatment_balance = min(t_ratio, 1 - t_ratio) / max(t_ratio, 1 - t_ratio)
                if treatment_balance < 0.1:
                    warnings.append(f"처치 불균형 심각: 처치 비율 {t_ratio:.1%}")
            else:
                treatment_balance = float(df[treatment_col].std() / (df[treatment_col].mean() + 1e-10))
        else:
            treatment_type = "unknown"
            treatment_balance = 0.0

        # Outcome 유형 판별
        if outcome_col in df.columns:
            o_nunique = df[outcome_col].nunique()
            outcome_type = "binary" if o_nunique <= 2 else "continuous"
        else:
            outcome_type = "unknown"

        # 결측치
        missing_ratio = df[feature_names].isnull().sum().sum() / max(df[feature_names].size, 1)
        has_missing = missing_ratio > 0
        if missing_ratio > 0.1:
            warnings.append(f"결측치 비율 {missing_ratio:.1%} (10% 초과)")

        # 표본 크기 경고
        if n_samples < 500:
            warnings.append(f"소표본 주의 (n={n_samples})")
        if n_samples < 100:
            warnings.append("표본 크기 매우 부족 → 결과 신뢰도 낮음")

        # Overlap 위험도 (Propensity Score 분포 기반 간이 추정)
        overlap_risk = "low"
        if treatment_type == "binary" and treatment_col in df.columns and feature_names:
            try:
                from sklearn.linear_model import LogisticRegression
                X = df[feature_names].fillna(0).values
                t = df[treatment_col].values
                lr = LogisticRegression(max_iter=200, solver="lbfgs")
                lr.fit(X, t)
                ps = lr.predict_proba(X)[:, 1]
                ps_std = np.std(ps)
                if ps_std > 0.3:
                    overlap_risk = "high"
                    warnings.append("Propensity Score 분산 높음 → Overlap 위험")
                elif ps_std > 0.15:
                    overlap_risk = "medium"
            except Exception:
                overlap_risk = "unknown"

        # 선형성 점수 (Treatment-Outcome 상관 기준 간이 추정)
        linearity_score = 0.5
        if treatment_col in df.columns and outcome_col in df.columns:
            try:
                corr = abs(df[treatment_col].corr(df[outcome_col]))
                linearity_score = float(corr) if not np.isnan(corr) else 0.5
            except Exception:
                pass

        return DataProfile(
            n_samples=n_samples,
            n_features=n_features,
            treatment_type=treatment_type,
            outcome_type=outcome_type,
            has_missing=has_missing,
            missing_ratio=missing_ratio,
            treatment_balance=treatment_balance,
            overlap_risk=overlap_risk,
            linearity_score=linearity_score,
            confounders_count=n_features,
            warnings=warnings,
        )

    def _recommend_method(self, profile: DataProfile) -> Dict[str, Any]:
        """프로파일 기반 최적 방법론을 추천합니다."""

        # 기본 추천
        primary_method = "linear_dml"
        nuisance_model = "lightgbm"
        meta_learners = ["S-Learner", "T-Learner", "X-Learner", "DR-Learner"]
        reasoning_parts = []

        # Treatment 유형별 분기
        if profile.treatment_type == "binary":
            meta_learners = ["T-Learner", "X-Learner", "DR-Learner", "S-Learner"]
            reasoning_parts.append("이진 처치 → T/X-Learner 우선")
        elif profile.treatment_type == "continuous":
            primary_method = "linear_dml"
            reasoning_parts.append("연속 처치 → DML 최적")
        elif profile.treatment_type == "multi_level":
            meta_learners = ["S-Learner", "DR-Learner"]
            reasoning_parts.append("다수준 처치 → S/DR-Learner 안정적")

        # 표본 크기별 분기
        if profile.n_samples < 500:
            nuisance_model = "linear"
            meta_learners = ["S-Learner", "T-Learner"]
            reasoning_parts.append("소표본 → 선형 모델 + 단순 러너")
        elif profile.n_samples > 50000:
            reasoning_parts.append("대규모 표본 → 비모수적 방법 유리")

        # 선형성에 따른 분기
        if profile.linearity_score > 0.7:
            primary_method = "linear_dml"
            reasoning_parts.append("높은 선형성 → LinearDML 최적")
        elif profile.linearity_score < 0.3:
            primary_method = "causal_forest"
            nuisance_model = "lightgbm"
            reasoning_parts.append("비선형 관계 → Causal Forest 추천")

        # Overlap 위험 대응
        if profile.overlap_risk == "high":
            meta_learners = ["DR-Learner", "X-Learner"]
            reasoning_parts.append("Overlap 위험 → DR/X-Learner (robustness)")

        # R-Learner는 기본 제외 (벤치마크에서 일관적으로 저성능)
        if "R-Learner" in meta_learners:
            meta_learners.remove("R-Learner")

        return {
            "primary_method": primary_method,
            "nuisance_model": nuisance_model,
            "recommended_learners": meta_learners,
            "reasoning": " | ".join(reasoning_parts),
            "confidence": "high" if len(profile.warnings) == 0 else
                         "medium" if len(profile.warnings) <= 2 else "low",
            "warnings": profile.warnings,
        }
