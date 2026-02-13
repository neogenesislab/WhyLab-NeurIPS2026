# -*- coding: utf-8 -*-
"""ConformalCell — 분포무가정 CATE 예측구간.

정규분포 가정 없이, Split Conformal Prediction으로
개별 단위 수준의 유효한 CATE 신뢰구간을 생성합니다.

참고문헌:
  - Lei & Candès (2021) "Conformal Inference of Counterfactuals"
  - Vovk et al. (2005) "Algorithmic Learning in a Random World"
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class ConformalCell(BaseCell):
    """분포무가정(Distribution-Free) CATE 예측구간 셀.

    Split Conformal Prediction:
      1. 데이터를 Train/Calibration으로 분할
      2. Train으로 CATE 모델 학습
      3. Calibration에서 적합도 점수(conformity score) 계산
      4. Quantile로 예측구간 폭 결정
      5. 새 데이터에 대해 [τ̂(x) - q, τ̂(x) + q] 구간 생성
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="conformal_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Conformal CATE 예측구간을 생성합니다.

        Args:
            inputs: MetaLearnerCell 또는 CausalCell 출력.
                필수: dataframe, feature_names, treatment_col, outcome_col

        Returns:
            conformal_ci: (n, 2) 배열 — 각 개체의 CATE 예측구간
            conformal_width: float — 평균 구간 폭
            coverage: float — 캘리브레이션 적중률
        """
        self.validate_inputs(
            inputs,
            ["dataframe", "feature_names", "treatment_col", "outcome_col"],
        )

        df = inputs["dataframe"]
        X_cols = inputs["feature_names"]
        T_col = inputs["treatment_col"]
        Y_col = inputs["outcome_col"]
        alpha = self.config.dml.alpha  # 유의수준 (기본 0.05)

        X = df[X_cols].values.astype(np.float64)
        T = df[T_col].values.astype(np.float64)
        Y = df[Y_col].values.astype(np.float64)
        n = len(X)

        self.logger.info("📐 Conformal CATE 예측구간 생성 (α=%.2f)", alpha)

        # ── 1. Train/Calibration 분할 (6:4) ──
        np.random.seed(self.config.data.random_seed)
        perm = np.random.permutation(n)
        split = int(n * 0.6)
        train_idx, cal_idx = perm[:split], perm[split:]

        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_cal, T_cal, Y_cal = X[cal_idx], T[cal_idx], Y[cal_idx]

        self.logger.info("   분할: Train=%d, Calibration=%d", len(train_idx), len(cal_idx))

        # ── 2. CATE 학습: mode에 따라 분기 ──
        from engine.cells.meta_learner_cell import DRLearner

        conformal_mode = inputs.get("conformal_mode", "split")  # "split" | "cqr"

        if conformal_mode == "cqr":
            self.logger.info("   ⚡ CQR (Conformalized Quantile Regression)")
            return self._cqr_conformal(
                X, T, Y, X_tr, T_tr, Y_tr, X_cal, T_cal, Y_cal,
                train_idx, cal_idx, alpha, inputs,
            )

        # Split Conformal (기본)
        model = DRLearner(config=self.config)
        model.fit(X_tr, T_tr, Y_tr)
        cate_cal = model.predict_cate(X_cal)

        # ── 3. Conformity Score: DR Score 기반 ──
        # 반사실을 직접 관측할 수 없으므로,
        # DR (Doubly Robust) score를 사용하여 ITE 근사
        scores = self._compute_dr_scores(X_cal, T_cal, Y_cal, X_tr, T_tr, Y_tr)
        residuals = np.abs(scores - cate_cal)

        # ── 4. Quantile 계산 ──
        # q = ⌈(1-α)(n_cal+1)⌉ / n_cal 번째 잔차
        level = np.ceil((1 - alpha) * (len(cal_idx) + 1)) / len(cal_idx)
        level = min(level, 1.0)
        q = float(np.quantile(residuals, level))

        self.logger.info("   Quantile q=%.5f (level=%.3f)", q, level)

        # ── 5. 전체 데이터에 대한 예측구간 ──
        # 전체 데이터로 최종 모델 재학습
        final_model = DRLearner(config=self.config)
        final_model.fit(X, T, Y)
        cate_all = final_model.predict_cate(X)

        ci_lower = cate_all - q
        ci_upper = cate_all + q

        # ── 6. 캘리브레이션 적중률 (Empirical Coverage) ──
        cate_cal_final = final_model.predict_cate(X_cal)
        cal_lower = cate_cal_final - q
        cal_upper = cate_cal_final + q
        # scores가 DR-ITE 근사이므로 이것이 CI에 포함되는 비율
        covered = np.mean((scores >= cal_lower) & (scores <= cal_upper))

        width = float(np.mean(ci_upper - ci_lower))

        self.logger.info(
            "   결과: 평균 구간 폭=%.5f, 적중률=%.1f%% (목표 %.0f%%)",
            width, covered * 100, (1 - alpha) * 100,
        )

        conformal_results = {
            "mode": "split",
            "alpha": alpha,
            "quantile_q": q,
            "mean_width": width,
            "coverage": float(covered),
            "target_coverage": 1 - alpha,
            "n_train": len(train_idx),
            "n_calibration": len(cal_idx),
            "ci_lower_mean": float(np.mean(ci_lower)),
            "ci_upper_mean": float(np.mean(ci_upper)),
            "interpretation": (
                f"Conformal {(1-alpha)*100:.0f}% 예측구간: "
                f"폭={width:.5f}, 실제 적중률={covered*100:.1f}%"
            ),
        }

        return {
            **inputs,
            "conformal_ci_lower": ci_lower,
            "conformal_ci_upper": ci_upper,
            "conformal_cate": cate_all,
            "conformal_results": conformal_results,
        }

    def _cqr_conformal(
        self,
        X: np.ndarray, T: np.ndarray, Y: np.ndarray,
        X_tr: np.ndarray, T_tr: np.ndarray, Y_tr: np.ndarray,
        X_cal: np.ndarray, T_cal: np.ndarray, Y_cal: np.ndarray,
        train_idx: np.ndarray, cal_idx: np.ndarray,
        alpha: float,
        inputs: dict,
    ) -> dict:
        """CQR (Conformalized Quantile Regression).

        Romano et al. (2019) 기반의 적응적 구간:
        X에 따라 CI 폭이 변함 — 불확실 영역에서 넓고, 확실 영역에서 좁음.

        Step 1: 분위수 회귀 q_lo(x), q_hi(x) 학습
        Step 2: Calibration에서 적합도 점수
          sᵢ = max(q_lo(Xᵢ) - Γᵢ, Γᵢ - q_hi(Xᵢ))
        Step 3: q = Quantile(1-α, {sᵢ})
        Step 4: CI(x) = [q_lo(x) - q, q_hi(x) + q]
        """
        from lightgbm import LGBMRegressor
        from engine.gpu_factory import create_lgbm_regressor

        # DR Score로 ITE 근사
        scores_cal = self._compute_dr_scores(
            X_cal, T_cal, Y_cal, X_tr, T_tr, Y_tr,
        )
        scores_tr = self._compute_dr_scores(
            X_tr, T_tr, Y_tr, X_tr, T_tr, Y_tr,
        )

        # 분위수 회귀: 하한 q_{α/2}(x), 상한 q_{1-α/2}(x)
        q_lo_model = create_lgbm_regressor(self.config, lightweight=True)
        q_hi_model = create_lgbm_regressor(self.config, lightweight=True)

        # LightGBM quantile objective
        q_lo_model.set_params(objective="quantile", alpha=alpha / 2)
        q_hi_model.set_params(objective="quantile", alpha=1 - alpha / 2)

        q_lo_model.fit(X_tr, scores_tr)
        q_hi_model.fit(X_tr, scores_tr)

        # Calibration에서 적합도 점수
        q_lo_cal = q_lo_model.predict(X_cal)
        q_hi_cal = q_hi_model.predict(X_cal)

        conformity = np.maximum(
            q_lo_cal - scores_cal,
            scores_cal - q_hi_cal,
        )

        # 분위수
        level = np.ceil((1 - alpha) * (len(cal_idx) + 1)) / len(cal_idx)
        level = min(level, 1.0)
        q = float(np.quantile(conformity, level))

        self.logger.info("   CQR Quantile q=%.5f", q)

        # 전체 데이터에 적응적 구간 생성
        q_lo_all = q_lo_model.predict(X)
        q_hi_all = q_hi_model.predict(X)

        ci_lower = q_lo_all - q
        ci_upper = q_hi_all + q

        # CATE 점추정 (중앙값)
        from engine.cells.meta_learner_cell import DRLearner
        final_model = DRLearner(config=self.config)
        final_model.fit(X, T, Y)
        cate_all = final_model.predict_cate(X)

        # 적중률
        scores_all_cal = self._compute_dr_scores(
            X_cal, T_cal, Y_cal, X_tr, T_tr, Y_tr,
        )
        cal_lo = q_lo_model.predict(X_cal) - q
        cal_hi = q_hi_model.predict(X_cal) + q
        covered = np.mean((scores_all_cal >= cal_lo) & (scores_all_cal <= cal_hi))

        widths = ci_upper - ci_lower
        mean_width = float(np.mean(widths))
        std_width = float(np.std(widths))

        self.logger.info(
            "   CQR 결과: 평균폭=%.5f±%.5f, 적중률=%.1f%%",
            mean_width, std_width, covered * 100,
        )

        conformal_results = {
            "mode": "cqr",
            "alpha": alpha,
            "quantile_q": q,
            "mean_width": mean_width,
            "width_std": std_width,
            "coverage": float(covered),
            "target_coverage": 1 - alpha,
            "n_train": len(train_idx),
            "n_calibration": len(cal_idx),
            "ci_lower_mean": float(np.mean(ci_lower)),
            "ci_upper_mean": float(np.mean(ci_upper)),
            "adaptive": True,
            "interpretation": (
                f"CQR {(1-alpha)*100:.0f}% 적응적 예측구간: "
                f"평균폭={mean_width:.5f}±{std_width:.5f}, "
                f"적중률={covered*100:.1f}%"
            ),
        }

        return {
            **inputs,
            "conformal_ci_lower": ci_lower,
            "conformal_ci_upper": ci_upper,
            "conformal_cate": cate_all,
            "conformal_results": conformal_results,
        }

    def _compute_dr_scores(
        self,
        X_cal: np.ndarray,
        T_cal: np.ndarray,
        Y_cal: np.ndarray,
        X_tr: np.ndarray,
        T_tr: np.ndarray,
        Y_tr: np.ndarray,
    ) -> np.ndarray:
        """DR Score 계산: ITE (Individual Treatment Effect) 근사.

        Γ̂ᵢ = μ̂₁(Xᵢ) - μ̂₀(Xᵢ)
            + Tᵢ·(Yᵢ - μ̂₁(Xᵢ))/ê(Xᵢ)
            - (1-Tᵢ)·(Yᵢ - μ̂₀(Xᵢ))/(1-ê(Xᵢ))
        """
        from engine.gpu_factory import create_lgbm_regressor
        from sklearn.linear_model import LogisticRegression

        # 이진화
        threshold = np.median(T_tr)
        T_tr_bin = (T_tr >= threshold).astype(int)
        T_cal_bin = (T_cal >= threshold).astype(int)

        # Outcome 모델 (처치별) — GPU 가속
        mask1 = T_tr_bin == 1
        mask0 = T_tr_bin == 0

        mu1 = create_lgbm_regressor(self.config, lightweight=True)
        mu0 = create_lgbm_regressor(self.config, lightweight=True)
        mu1.fit(X_tr[mask1], Y_tr[mask1])
        mu0.fit(X_tr[mask0], Y_tr[mask0])

        # Propensity Score
        ps = LogisticRegression(max_iter=1000, random_state=42)
        ps.fit(X_tr, T_tr_bin)
        e_hat = np.clip(ps.predict_proba(X_cal)[:, 1], 0.01, 0.99)

        # DR Score
        mu1_cal = mu1.predict(X_cal)
        mu0_cal = mu0.predict(X_cal)
        gamma = (mu1_cal - mu0_cal
                 + T_cal_bin * (Y_cal - mu1_cal) / e_hat
                 - (1 - T_cal_bin) * (Y_cal - mu0_cal) / (1 - e_hat))

        return gamma
