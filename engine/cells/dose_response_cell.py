# -*- coding: utf-8 -*-
"""DoseResponseCell — 연속 처치의 비선형 용량-반응 곡선 추정.

연속 처치(가격, 할인율, 투약량 등)의 비선형 효과를 추정합니다.
GPS(Generalized Propensity Score)를 활용한 이중 견고(Doubly Robust) 추정:

1. GPS 추정: 처치 T가 공변량 X의 조건부 분포를 모델링
2. 결과 표면(Outcome Surface) 추정: E[Y | T, GPS(T, X)] 모델링
3. 용량-반응 곡선 도출: E[Y(t)] = ∫ E[Y | T=t, GPS(t, x)] dF(x)
4. 한계 효과(Marginal Effect): dE[Y(t)]/dt

학술 참조:
  - Hirano & Imbens (2004). "The Propensity Score with Continuous Treatments."
  - Schwab et al. (2020). "Learning Counterfactual Representations for
    Estimating Individual Dose-Response Curves."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

@dataclass
class DoseResponseConfig:
    """용량-반응 곡선 추정 설정."""

    n_grid_points: int = 50          # 용량 격자 점 수
    gps_method: str = "gaussian"     # GPS 추정 방식: "gaussian" | "kernel"
    outcome_method: str = "kernel"   # 결과 표면 방식: "kernel" | "polynomial"
    polynomial_degree: int = 3       # 다항 회귀 차수
    kernel_bandwidth: float = 0.0    # 0이면 자동 설정 (Silverman's rule)
    bootstrap_ci: bool = True        # Bootstrap 신뢰구간
    n_bootstrap: int = 100           # Bootstrap 반복 수
    ci_alpha: float = 0.05           # 신뢰구간 유의수준 (95%)
    trim_quantile: float = 0.01      # 극단값 제거 분위수


# ──────────────────────────────────────────────
# GPS 추정
# ──────────────────────────────────────────────

def estimate_gps_gaussian(
    T: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """가우시안 GPS: T | X ~ N(μ(X), σ²).

    Returns:
        (gps_values, sigma, residuals)
    """
    lr = LinearRegression()
    lr.fit(X, T)
    mu = lr.predict(X)
    residuals = T - mu
    sigma = np.std(residuals)

    # GPS = φ((T - μ(X)) / σ)
    gps = sp_stats.norm.pdf(T, loc=mu, scale=sigma)
    return gps, sigma, residuals


# ──────────────────────────────────────────────
# 결과 표면 추정
# ──────────────────────────────────────────────

def estimate_outcome_surface_polynomial(
    T: np.ndarray,
    GPS: np.ndarray,
    Y: np.ndarray,
    degree: int = 3,
) -> Any:
    """다항 회귀 기반 결과 표면: E[Y | T, GPS].

    Returns:
        (poly, scaler, model) — 예측용 객체들.
    """
    Z = np.column_stack([T, GPS])
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    Z_poly = poly.fit_transform(scaler.fit_transform(Z))

    model = LinearRegression()
    model.fit(Z_poly, Y)

    return poly, scaler, model


def predict_outcome_surface(
    poly, scaler, model,
    t_grid: np.ndarray,
    gps_grid: np.ndarray,
) -> np.ndarray:
    """결과 표면 예측."""
    Z = np.column_stack([t_grid, gps_grid])
    Z_poly = poly.transform(scaler.transform(Z))
    return model.predict(Z_poly)


# ──────────────────────────────────────────────
# 커널 회귀 기반 용량-반응 곡선
# ──────────────────────────────────────────────

def kernel_dose_response(
    T: np.ndarray,
    Y: np.ndarray,
    X: np.ndarray,
    t_grid: np.ndarray,
    bandwidth: float = 0.0,
) -> np.ndarray:
    """Nadaraya-Watson 커널 회귀 기반 용량-반응 곡선.

    E[Y(t)] ≈ Σ K((T_i - t)/h) · Y_i / Σ K((T_i - t)/h)
    """
    if bandwidth <= 0:
        # Silverman's rule of thumb
        bandwidth = 1.06 * np.std(T) * len(T) ** (-1 / 5)

    dr_curve = np.zeros(len(t_grid))
    for j, t in enumerate(t_grid):
        weights = sp_stats.norm.pdf((T - t) / bandwidth)
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            dr_curve[j] = np.sum(weights * Y) / weight_sum
        else:
            dr_curve[j] = np.nan

    return dr_curve


# ──────────────────────────────────────────────
# DoseResponseCell 메인
# ──────────────────────────────────────────────

class DoseResponseCell:
    """연속 처치의 용량-반응 곡선 추정 셀.

    파이프라인 인터페이스:
        cell = DoseResponseCell(config)
        result = cell.execute(inputs)

    직접 호출:
        result = cell.estimate(X, T, Y)
    """

    name = "DoseResponse"

    def __init__(self, config=None, dr_config: Optional[DoseResponseConfig] = None):
        self.config = config
        self.dr_config = dr_config or DoseResponseConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.response_model = None  # 예측용 모델 (LGBM)

    def estimate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, Any]:
        """용량-반응 곡선을 추정합니다.

        Args:
            X: (n, p) 공변량.
            T: (n,) 연속 처치.
            Y: (n,) 결과.

        Returns:
            Dict with: t_grid, dr_curve, marginal_effect, ci_lower, ci_upper, optimal_dose
        """
        cfg = self.dr_config
        n = len(T)

        # 극단값 트림
        q_low = np.quantile(T, cfg.trim_quantile)
        q_high = np.quantile(T, 1 - cfg.trim_quantile)
        t_grid = np.linspace(q_low, q_high, cfg.n_grid_points)

        self.logger.info(
            "용량-반응 추정 시작: n=%d, T ∈ [%.2f, %.2f], grid=%d점",
            n, q_low, q_high, cfg.n_grid_points,
        )

        # 메인 곡선 추정
        dr_curve = self._estimate_curve(X, T, Y, t_grid)

        # 한계 효과 (수치 미분)
        marginal_effect = np.gradient(dr_curve, t_grid)

        # Bootstrap 신뢰구간
        ci_lower, ci_upper = None, None
        if cfg.bootstrap_ci:
            ci_lower, ci_upper = self._bootstrap_ci(X, T, Y, t_grid)

        # 최적 용량 (곡선의 최대/최소)
        optimal_idx = np.nanargmax(dr_curve) if not np.all(np.isnan(dr_curve)) else 0
        optimal_dose = float(t_grid[optimal_idx])
        optimal_effect = float(dr_curve[optimal_idx]) if not np.isnan(dr_curve[optimal_idx]) else 0.0
        
        # ──────────────────────────────────────────
        # 시뮬레이션을 위한 반응 모델 학습 (LightGBM)
        # E[Y | X, T] 근사
        # ──────────────────────────────────────────
        try:
            self.response_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            # 입력: [X, T]
            XT = np.column_stack([X, T])
            self.response_model.fit(XT, Y)
            self.logger.info("시뮬레이션용 반응 모델 학습 완료 (LGBM)")
        except Exception as e:
            self.logger.warning("반응 모델 학습 실패: %s", e)

        # 효과 유무 판정 (곡선의 기울기 변동)
        effect_magnitude = np.nanstd(dr_curve)
        has_effect = bool(effect_magnitude > np.nanstd(Y) * 0.01)

        self.logger.info(
            "추정 완료: optimal_dose=%.2f, effect_magnitude=%.4f, has_effect=%s",
            optimal_dose, effect_magnitude, has_effect,
        )

        result = {
            "t_grid": t_grid.tolist(),
            "dr_curve": dr_curve.tolist(),
            "marginal_effect": marginal_effect.tolist(),
            "optimal_dose": optimal_dose,
            "optimal_effect": optimal_effect,
            "effect_magnitude": float(effect_magnitude),
            "has_effect": has_effect,
        }

        if ci_lower is not None:
            result["ci_lower"] = ci_lower.tolist()
            result["ci_upper"] = ci_upper.tolist()

        return result

    def predict(self, X: np.ndarray, T_new: np.ndarray) -> np.ndarray:
        """새로운 처치 T_new에 대한 결과 Y를 예측합니다.
        
        Args:
            X: (n, p) 공변량
            T_new: (n,) 새로운 처치 값
            
        Returns:
            Y_pred: (n,) 예측된 결과
        """
        if self.response_model is None:
            raise ValueError("반응 모델이 학습되지 않았습니다. 먼저 estimate()를 실행하세요.")
            
        XT_new = np.column_stack([X, T_new])
        return self.response_model.predict(XT_new)

    def _estimate_curve(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        """단일 곡선 추정 (설정에 따라 방법 선택)."""
        cfg = self.dr_config

        if cfg.outcome_method == "polynomial":
            # GPS + 다항 결과 표면
            gps, sigma, _ = estimate_gps_gaussian(T, X)
            poly, scaler, model = estimate_outcome_surface_polynomial(
                T, gps, Y, degree=cfg.polynomial_degree,
            )
            # 격자 점에서 GPS를 평균으로 계산
            lr_gps = LinearRegression().fit(X, T)
            dr_curve = np.zeros(len(t_grid))
            for j, t in enumerate(t_grid):
                gps_j = sp_stats.norm.pdf(t, loc=lr_gps.predict(X), scale=sigma)
                y_pred = predict_outcome_surface(
                    poly, scaler, model,
                    np.full(len(X), t),
                    gps_j,
                )
                dr_curve[j] = np.mean(y_pred)
            return dr_curve
        else:
            # 커널 회귀 (기본)
            return kernel_dose_response(
                T, Y, X, t_grid,
                bandwidth=cfg.kernel_bandwidth,
            )

    def _bootstrap_ci(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        t_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap 신뢰구간."""
        cfg = self.dr_config
        rng = np.random.RandomState(42)
        n = len(T)
        boot_curves = []

        for b in range(cfg.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            curve = self._estimate_curve(X[idx], T[idx], Y[idx], t_grid)
            boot_curves.append(curve)

        boot_arr = np.array(boot_curves)
        alpha = cfg.ci_alpha
        ci_lower = np.nanpercentile(boot_arr, 100 * alpha / 2, axis=0)
        ci_upper = np.nanpercentile(boot_arr, 100 * (1 - alpha / 2), axis=0)

        return ci_lower, ci_upper

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """파이프라인 셀 인터페이스."""
        df = inputs.get("dataframe")
        feature_names = inputs.get("feature_names", [])
        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")

        if df is None:
            self.logger.warning("데이터프레임이 없습니다. 건너뜁니다.")
            return {**inputs, "dose_response": None}

        # 연속 처치 확인
        t_values = df[treatment_col].values
        n_unique = len(np.unique(t_values))
        if n_unique <= 2:
            self.logger.info(
                "이진 처치 (%d개 고유값). 용량-반응 곡선 불필요.", n_unique,
            )
            return {**inputs, "dose_response": {"skipped": True, "reason": "binary_treatment"}}

        features = feature_names or [
            c for c in df.columns if c not in [treatment_col, outcome_col]
        ]
        X = df[features].values.astype(np.float64)
        T = t_values.astype(np.float64)
        Y = df[outcome_col].values.astype(np.float64)

        result = self.estimate(X, T, Y)

        self.logger.info(
            "DoseResponse 완료: optimal_dose=%.2f, effect=%.4f",
            result["optimal_dose"], result["effect_magnitude"],
        )

        return {**inputs, "dose_response": result}
