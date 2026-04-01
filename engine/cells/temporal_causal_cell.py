# -*- coding: utf-8 -*-
"""TemporalCausalCell — 시계열 인과추론.

시계열 데이터에서 인과 관계를 분석하는 세 가지 방법론:

- **Granger Causality**: 시차(Lag) 기반 인과 방향 검정
- **CausalImpact (합성 통제)**: 개입 전후 반사실 시나리오 추정
- **Lag Correlation Analysis**: 최적 시차 자동 탐색

Phase 10-2: 시계열 인과추론.
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
class GrangerResult:
    """Granger 인과 검정 결과."""
    cause_col: str = ""
    effect_col: str = ""
    max_lag: int = 5
    best_lag: int = 1
    f_stat: float = 0.0
    p_value: float = 1.0
    is_causal: bool = False
    interpretation: str = ""


@dataclass
class CausalImpactResult:
    """인과 영향(CausalImpact) 추정 결과."""
    pre_mean: float = 0.0
    post_mean: float = 0.0
    predicted_post_mean: float = 0.0
    absolute_effect: float = 0.0
    relative_effect: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    intervention_point: int = 0
    interpretation: str = ""


@dataclass
class LagCorrelationResult:
    """시차 상관 분석 결과."""
    optimal_lag: int = 0
    max_correlation: float = 0.0
    lag_correlations: Dict[int, float] = field(default_factory=dict)
    interpretation: str = ""


class TemporalCausalCell(BaseCell):
    """시계열 인과추론 셀.

    시계열 구조를 가진 데이터에서 인과 관계를 분석합니다.
    자동으로 시계열 여부를 판단하고, 적용 가능한 방법론을 실행합니다.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="temporal_causal_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """시계열 인과 분석 실행."""
        df = inputs.get("dataframe")
        if df is None:
            self.logger.warning("데이터프레임 없음 → TemporalCausal 건너뜀")
            return inputs

        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")
        time_col = inputs.get("time_col")
        feature_names = inputs.get("feature_names", [])

        # 시계열 여부 자동 감지
        is_temporal = self._detect_temporal(df, time_col)
        if not is_temporal and time_col is None:
            self.logger.info("ℹ️ 시계열 구조 미감지 → TemporalCausal 건너뜀")
            return inputs

        results = {}

        # Granger 인과 검정
        if treatment_col in df.columns and outcome_col in df.columns:
            self.logger.info("⏱️ Granger 인과 검정 시작")
            granger = self._granger_test(df, treatment_col, outcome_col)
            results["granger"] = granger

            # 역방향도 체크 (양방향 Granger)
            granger_reverse = self._granger_test(df, outcome_col, treatment_col)
            results["granger_reverse"] = granger_reverse

        # 시차 상관 분석
        if treatment_col in df.columns and outcome_col in df.columns:
            self.logger.info("📈 시차 상관 분석 시작")
            lag_corr = self._lag_correlation(df, treatment_col, outcome_col)
            results["lag_correlation"] = lag_corr

        # CausalImpact (개입 시점이 있는 경우)
        intervention_point = inputs.get("intervention_point")
        if intervention_point is not None and outcome_col in df.columns:
            self.logger.info("📊 CausalImpact 분석 시작 (개입: %s)", intervention_point)
            impact = self._causal_impact(df, outcome_col, intervention_point, feature_names)
            results["causal_impact"] = impact
        elif treatment_col in df.columns and df[treatment_col].nunique() == 2:
            # 개입 시점 자동 추정 (처치 변수가 0→1로 바뀌는 시점)
            auto_point = self._detect_intervention_point(df, treatment_col)
            if auto_point is not None:
                self.logger.info("🔍 개입 시점 자동 감지: index=%d", auto_point)
                impact = self._causal_impact(df, outcome_col, auto_point, feature_names)
                results["causal_impact"] = impact

        methods_used = list(results.keys())
        if methods_used:
            self.logger.info("✅ 시계열 인과 분석 완료: %s", ", ".join(methods_used))

        return {
            **inputs,
            "temporal_causal": self._serialize(results),
        }

    def _detect_temporal(self, df: pd.DataFrame, time_col: Optional[str]) -> bool:
        """데이터의 시계열 구조를 자동 감지합니다."""
        if time_col and time_col in df.columns:
            return True

        # datetime 인덱스 체크
        if isinstance(df.index, pd.DatetimeIndex):
            return True

        # datetime 컬럼 체크
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return True

        # 순서가 있는 정수 인덱스 (시계열 가능성)
        if df.index.is_monotonic_increasing and len(df) > 50:
            return True

        return False

    def _granger_test(
        self, df: pd.DataFrame, cause_col: str, effect_col: str, max_lag: int = 5,
    ) -> GrangerResult:
        """Granger 인과 검정.

        제한된 모델(과거 Y만)과 비제한 모델(과거 Y + X)의 비교.
        """
        from sklearn.linear_model import LinearRegression

        y = df[effect_col].values
        x = df[cause_col].values
        n = len(y)

        best_lag = 1
        best_f = 0.0
        best_p = 1.0

        for lag in range(1, min(max_lag + 1, n // 4)):
            # 시차 변수 생성
            Y = y[lag:]
            Y_lags = np.column_stack([y[lag - i - 1:n - i - 1] for i in range(lag)])
            X_lags = np.column_stack([x[lag - i - 1:n - i - 1] for i in range(lag)])

            # 제한 모델: Y ~ Y_lags
            restricted = LinearRegression().fit(Y_lags, Y)
            rss_r = np.sum((Y - restricted.predict(Y_lags)) ** 2)

            # 비제한 모델: Y ~ Y_lags + X_lags
            unrestricted_X = np.hstack([Y_lags, X_lags])
            unrestricted = LinearRegression().fit(unrestricted_X, Y)
            rss_u = np.sum((Y - unrestricted.predict(unrestricted_X)) ** 2)

            # F-검정
            n_obs = len(Y)
            k = lag
            f_stat = ((rss_r - rss_u) / k) / (rss_u / max(1, n_obs - 2 * k - 1) + 1e-10)

            from scipy import stats as scipy_stats
            p_value = float(1 - scipy_stats.f.cdf(f_stat, k, max(1, n_obs - 2 * k - 1)))

            if f_stat > best_f:
                best_f = float(f_stat)
                best_p = p_value
                best_lag = lag

        is_causal = best_p < 0.05

        return GrangerResult(
            cause_col=cause_col, effect_col=effect_col,
            max_lag=max_lag, best_lag=best_lag,
            f_stat=best_f, p_value=best_p, is_causal=is_causal,
            interpretation=(
                f"Granger 검정: {cause_col} → {effect_col} "
                f"{'✅ 유의 (인과)' if is_causal else '❌ 비유의'} "
                f"(F={best_f:.2f}, p={best_p:.4f}, lag={best_lag})"
            ),
        )

    def _lag_correlation(
        self, df: pd.DataFrame, x_col: str, y_col: str, max_lag: int = 10,
    ) -> LagCorrelationResult:
        """시차 상관 분석: 최적 시차를 자동으로 찾습니다."""
        x = df[x_col].values
        y = df[y_col].values
        n = len(x)

        correlations = {}
        max_corr = 0.0
        optimal_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                corr = float(np.corrcoef(x[:n - lag], y[lag:])[0, 1]) if lag < n else 0.0
            else:
                corr = float(np.corrcoef(x[-lag:], y[:n + lag])[0, 1]) if -lag < n else 0.0

            if np.isnan(corr):
                corr = 0.0
            correlations[lag] = corr

            if abs(corr) > abs(max_corr):
                max_corr = corr
                optimal_lag = lag

        return LagCorrelationResult(
            optimal_lag=optimal_lag,
            max_correlation=max_corr,
            lag_correlations=correlations,
            interpretation=(
                f"최적 시차={optimal_lag} (상관={max_corr:.3f}). "
                f"{'양(+)' if optimal_lag > 0 else '음(-)' if optimal_lag < 0 else '동시'} 시차에서 최대 상관."
            ),
        )

    def _causal_impact(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        intervention_point: int,
        covariates: List[str],
    ) -> CausalImpactResult:
        """간이 CausalImpact 분석 (합성 통제 기반).

        개입 전 데이터로 모델을 학습하고, 개입 후 반사실을 예측합니다.
        """
        from sklearn.linear_model import BayesianRidge

        y = df[outcome_col].values
        pre_y = y[:intervention_point]
        post_y = y[intervention_point:]

        if len(pre_y) < 10 or len(post_y) < 3:
            return CausalImpactResult(
                intervention_point=intervention_point,
                interpretation="개입 전/후 데이터 부족",
            )

        # 공변량이 있으면 합성 통제, 없으면 시계열 추세 기반
        cov_cols = [c for c in covariates if c in df.columns and c != outcome_col]

        if cov_cols:
            X = df[cov_cols].fillna(0).values
            X_pre = X[:intervention_point]
            X_post = X[intervention_point:]

            model = BayesianRidge()
            model.fit(X_pre, pre_y)
            predicted = model.predict(X_post)
        else:
            # 추세 기반 예측 (시간 인덱스를 특성으로)
            t_pre = np.arange(len(pre_y)).reshape(-1, 1)
            t_post = np.arange(len(pre_y), len(pre_y) + len(post_y)).reshape(-1, 1)

            model = BayesianRidge()
            model.fit(t_pre, pre_y)
            predicted = model.predict(t_post)

        # 효과 추정
        absolute_effect = float(np.mean(post_y) - np.mean(predicted))
        predicted_mean = float(np.mean(predicted))
        relative_effect = absolute_effect / (abs(predicted_mean) + 1e-10)

        # p-value 추정 (부트스트랩 간이)
        residuals = pre_y - model.predict(
            (df[cov_cols].fillna(0).values[:intervention_point] if cov_cols
             else np.arange(len(pre_y)).reshape(-1, 1))
        )
        se = float(np.std(residuals))
        z = abs(absolute_effect) / (se + 1e-10)
        from scipy import stats as scipy_stats
        p_value = float(2 * (1 - scipy_stats.norm.cdf(z)))

        significant = p_value < 0.05

        return CausalImpactResult(
            pre_mean=float(np.mean(pre_y)),
            post_mean=float(np.mean(post_y)),
            predicted_post_mean=predicted_mean,
            absolute_effect=absolute_effect,
            relative_effect=relative_effect,
            p_value=p_value,
            significant=significant,
            intervention_point=intervention_point,
            interpretation=(
                f"CausalImpact: 절대 효과={absolute_effect:.4f}, "
                f"상대 효과={relative_effect:.1%}, "
                f"{'✅ 유의' if significant else '❌ 비유의'} (p={p_value:.4f})"
            ),
        )

    def _detect_intervention_point(
        self, df: pd.DataFrame, treatment_col: str,
    ) -> Optional[int]:
        """처치 변수가 0→1로 변하는 시점을 자동 감지합니다."""
        t = df[treatment_col].values
        for i in range(1, len(t)):
            if t[i] == 1 and t[i - 1] == 0:
                return i
        return None

    def _serialize(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """결과 직렬화."""
        serialized = {}
        for key, val in results.items():
            if hasattr(val, '__dict__'):
                d = {k: v for k, v in val.__dict__.items() if not k.startswith('_')}
                # numpy 타입 변환
                for k, v in d.items():
                    if isinstance(v, (np.integer, np.floating)):
                        d[k] = v.item()
                serialized[key] = d
            else:
                serialized[key] = val
        return serialized
