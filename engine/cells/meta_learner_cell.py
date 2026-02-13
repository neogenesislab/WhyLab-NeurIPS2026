# -*- coding: utf-8 -*-
"""MetaLearnerCell — 5종 메타러너 + Oracle 앙상블 선택.

단일 LinearDML 래퍼를 넘어, 5가지 메타러너(S/T/X/DR/R)를 직접 구현하고
Cross-Validated MSE 기반으로 최적 메타러너를 자동 선택합니다.

참고문헌:
  - Künzel et al. (2019) "Metalearners for estimating HTE"
  - Kennedy (2023) "Towards optimal doubly robust estimation"
  - Nie & Wager (2021) "Quasi-oracle estimation of HTE"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


# ──────────────────────────────────────────
# 개별 메타러너 구현
# ──────────────────────────────────────────

class _BaseMetaLearner:
    """메타러너 공통 인터페이스."""

    name: str = "base"

    def __init__(self, base_model_factory=None, config=None):
        """base_model_factory: sklearn 호환 모델 생성 함수."""
        self._config = config
        self._factory = base_model_factory or self._default_factory

    def _default_factory(self):
        if self._config is not None:
            from engine.gpu_factory import create_lgbm_regressor
            return create_lgbm_regressor(self._config)
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=200, max_depth=5, num_leaves=31,
            learning_rate=0.05, verbose=-1,
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "_BaseMetaLearner":
        raise NotImplementedError

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SLearner(_BaseMetaLearner):
    """S-Learner: T를 피처에 포함한 단일 모델.

    μ̂(x, t) = f(X, T)
    τ̂(x) = μ̂(x, 1) - μ̂(x, 0)
    """

    name = "S-Learner"

    def fit(self, X, T, Y):
        self.model_ = self._factory()
        XT = np.column_stack([X, T.reshape(-1, 1)])
        self.model_.fit(XT, Y)
        return self

    def predict_cate(self, X):
        n = X.shape[0]
        X1 = np.column_stack([X, np.ones((n, 1))])
        X0 = np.column_stack([X, np.zeros((n, 1))])
        return self.model_.predict(X1) - self.model_.predict(X0)


class TLearner(_BaseMetaLearner):
    """T-Learner: 처치/통제 분리 모델.

    μ̂₀(x) = E[Y | X=x, T=0]
    μ̂₁(x) = E[Y | X=x, T=1]
    τ̂(x) = μ̂₁(x) - μ̂₀(x)
    """

    name = "T-Learner"

    def fit(self, X, T, Y):
        # 이진화: 중앙값 기준 분리
        self.threshold_ = np.median(T)
        mask1 = T >= self.threshold_
        mask0 = ~mask1

        self.model_1_ = self._factory()
        self.model_0_ = self._factory()
        self.model_1_.fit(X[mask1], Y[mask1])
        self.model_0_.fit(X[mask0], Y[mask0])
        return self

    def predict_cate(self, X):
        return self.model_1_.predict(X) - self.model_0_.predict(X)


class XLearner(_BaseMetaLearner):
    """X-Learner (Künzel et al., 2019): 유사잔차 + PS 가중.

    Step 1: T-Learner 학습
    Step 2: 유사잔차 D̂₁ = Y - μ̂₀(X), D̂₀ = μ̂₁(X) - Y
    Step 3: CATE 학습 τ̂₁(x), τ̂₀(x)
    Step 4: τ̂(x) = g(x)·τ̂₀(x) + (1-g(x))·τ̂₁(x)
    """

    name = "X-Learner"

    def fit(self, X, T, Y):
        self.threshold_ = np.median(T)
        mask1 = T >= self.threshold_
        mask0 = ~mask1

        # Step 1: T-Learner
        mu1 = self._factory()
        mu0 = self._factory()
        mu1.fit(X[mask1], Y[mask1])
        mu0.fit(X[mask0], Y[mask0])

        # Step 2: 유사잔차 (Imputed Treatment Effects)
        D1 = Y[mask1] - mu0.predict(X[mask1])  # 처치군: 관측 - 반사실
        D0 = mu1.predict(X[mask0]) - Y[mask0]  # 통제군: 반사실 - 관측

        # Step 3: CATE 학습
        self.tau1_ = self._factory()
        self.tau0_ = self._factory()
        self.tau1_.fit(X[mask1], D1)
        self.tau0_.fit(X[mask0], D0)

        # Step 4: Propensity (처치 확률) — 가중치
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        T_binary = (T >= self.threshold_).astype(int)
        ps_model.fit(X, T_binary)
        self.ps_model_ = ps_model

        return self

    def predict_cate(self, X):
        g = self.ps_model_.predict_proba(X)[:, 1]  # P(T=1|X)
        tau1 = self.tau1_.predict(X)
        tau0 = self.tau0_.predict(X)
        return g * tau0 + (1 - g) * tau1


class DRLearner(_BaseMetaLearner):
    """DR-Learner (Kennedy, 2023): Doubly Robust CATE 추정.

    Γ̂ᵢ = μ̂₁(Xᵢ) - μ̂₀(Xᵢ)
        + Tᵢ·(Yᵢ - μ̂₁(Xᵢ))/ê(Xᵢ)
        - (1-Tᵢ)·(Yᵢ - μ̂₀(Xᵢ))/(1-ê(Xᵢ))

    τ̂(x) = E[Γ̂ | X=x]  (2단계 회귀)
    """

    name = "DR-Learner"

    def fit(self, X, T, Y):
        self.threshold_ = np.median(T)
        T_binary = (T >= self.threshold_).astype(int)
        mask1 = T_binary == 1
        mask0 = T_binary == 0

        # Outcome 모델 (처치별)
        mu1 = self._factory()
        mu0 = self._factory()
        mu1.fit(X[mask1], Y[mask1])
        mu0.fit(X[mask0], Y[mask0])

        # Propensity Score
        from sklearn.linear_model import LogisticRegression
        ps = LogisticRegression(max_iter=1000, random_state=42)
        ps.fit(X, T_binary)
        e_hat = np.clip(ps.predict_proba(X)[:, 1], 0.01, 0.99)

        # Doubly Robust Score 구성
        mu1_pred = mu1.predict(X)
        mu0_pred = mu0.predict(X)
        gamma = (mu1_pred - mu0_pred
                 + T_binary * (Y - mu1_pred) / e_hat
                 - (1 - T_binary) * (Y - mu0_pred) / (1 - e_hat))

        # 2단계: Γ̂를 X에 회귀
        self.final_model_ = self._factory()
        self.final_model_.fit(X, gamma)
        return self

    def predict_cate(self, X):
        return self.final_model_.predict(X)


class RLearner(_BaseMetaLearner):
    """R-Learner (Nie & Wager, 2021): Robinson Decomposition.

    Ỹ = Y - m̂(X)     (outcome 잔차)
    T̃ = T - ê(X)      (treatment 잔차)
    τ̂ = argmin_τ Σ [(Ỹ - τ(X)·T̃)² + λ·||τ||²]
    """

    name = "R-Learner"

    def fit(self, X, T, Y):
        # m̂(X) = E[Y|X] (marginal outcome model)
        m_model = self._factory()
        m_model.fit(X, Y)
        Y_tilde = Y - m_model.predict(X)

        # ê(X) = E[T|X] (marginal treatment model)
        e_model = self._factory()
        e_model.fit(X, T)
        T_tilde = T - e_model.predict(X)

        # T̃² 가중 잔차 회귀: min Σ (Ỹ/T̃ - τ(X))² · T̃²
        # 안전 가드: |T̃| 이 매우 작으면 불안정 → 클리핑
        eps = 0.01
        T_tilde_safe = np.where(np.abs(T_tilde) < eps, eps * np.sign(T_tilde + 1e-10), T_tilde)
        pseudo_outcome = Y_tilde / T_tilde_safe

        # 가중치: T̃²
        weights = T_tilde ** 2

        self.final_model_ = self._factory()
        self.final_model_.fit(X, pseudo_outcome, sample_weight=weights)
        return self

    def predict_cate(self, X):
        return self.final_model_.predict(X)


# ──────────────────────────────────────────
# 메타러너 셀 (통합)
# ──────────────────────────────────────────

class MetaLearnerCell(BaseCell):
    """5종 메타러너 + Oracle 선택 셀.

    모든 메타러너를 학습하고, Cross-Validated MSE 기반으로
    최적 메타러너를 자동 선택하거나 앙상블합니다.
    """

    LEARNER_REGISTRY = {
        "S-Learner": SLearner,
        "T-Learner": TLearner,
        "X-Learner": XLearner,
        "DR-Learner": DRLearner,
        "R-Learner": RLearner,
    }

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="meta_learner_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """5종 메타러너 학습 → Oracle 선택 → 앙상블 CATE.

        Args:
            inputs: CausalCell 출력 (dataframe, feature_names, treatment_col, outcome_col).

        Returns:
            개별 메타러너 결과 + Oracle 선택 + 앙상블 CATE.
        """
        self.validate_inputs(
            inputs, ["dataframe", "feature_names", "treatment_col", "outcome_col"],
        )

        df = inputs["dataframe"]
        X_cols = inputs["feature_names"]
        T_col = inputs["treatment_col"]
        Y_col = inputs["outcome_col"]

        X = df[X_cols].values.astype(np.float64)
        T = df[T_col].values.astype(np.float64)
        Y = df[Y_col].values.astype(np.float64)

        cfg = self.config.dml

        self.logger.info("🧬 메타러너 학습 시작 (5종)")

        # ── 1. 각 메타러너 학습 + CV-MSE 평가 ──
        learner_results: Dict[str, Dict] = {}

        for name, LearnerClass in self.LEARNER_REGISTRY.items():
            self.logger.info("   ▶ %s 학습 중...", name)
            try:
                result = self._train_and_evaluate(
                    LearnerClass, X, T, Y, cv_folds=cfg.cv_folds,
                )
                learner_results[name] = result
                self.logger.info(
                    "     %s: ATE=%.5f, CV-MSE=%.6f",
                    name, result["ate"], result["cv_mse"],
                )
            except Exception as e:
                self.logger.warning("     %s 실패: %s", name, e)
                learner_results[name] = {
                    "ate": 0.0, "cate": np.zeros(len(X)),
                    "cv_mse": float("inf"), "error": str(e),
                }

        # ── 2. Oracle 선택: CV-MSE 최소 ──
        valid = {k: v for k, v in learner_results.items() if v["cv_mse"] < float("inf")}
        if not valid:
            self.logger.error("모든 메타러너 실패!")
            return {**inputs, "meta_learner_results": {}}

        best_name = min(valid, key=lambda k: valid[k]["cv_mse"])
        best_result = valid[best_name]

        # ── 3. 앙상블 (MSE 역수 가중 평균) ──
        mse_values = np.array([v["cv_mse"] for v in valid.values()])
        # Softmax of negative MSE → 낮은 MSE에 높은 가중치
        weights = np.exp(-mse_values / (mse_values.mean() + 1e-10))
        weights = weights / weights.sum()

        cate_stack = np.column_stack([v["cate"] for v in valid.values()])
        ensemble_cate = (cate_stack * weights[np.newaxis, :]).sum(axis=1)
        ensemble_ate = float(np.mean(ensemble_cate))

        # 합의율: ATE 부호가 같은 메타러너 비율
        signs = [np.sign(v["ate"]) for v in valid.values()]
        majority_sign = np.sign(ensemble_ate) if ensemble_ate != 0 else 1
        consensus = sum(1 for s in signs if s == majority_sign) / len(signs)

        self.logger.info(
            "🏆 Oracle: %s (CV-MSE=%.6f), 앙상블 ATE=%.5f, 합의율=%.0f%%",
            best_name, best_result["cv_mse"], ensemble_ate, consensus * 100,
        )

        meta_results = {
            "learners": {
                name: {
                    "ate": float(r["ate"]),
                    "cv_mse": float(r["cv_mse"]),
                    "cate_mean": float(np.mean(r["cate"])),
                    "cate_std": float(np.std(r["cate"])),
                }
                for name, r in learner_results.items()
            },
            "oracle": {
                "best_learner": best_name,
                "cv_mse": float(best_result["cv_mse"]),
                "ate": float(best_result["ate"]),
            },
            "ensemble": {
                "ate": ensemble_ate,
                "weights": {n: float(w) for n, w in zip(valid.keys(), weights)},
                "consensus": consensus,
            },
        }

        return {
            **inputs,
            "meta_learner_results": meta_results,
            "ensemble_cate": ensemble_cate,
            "ensemble_ate": ensemble_ate,
            "best_learner": best_name,
        }

    def _train_and_evaluate(
        self,
        LearnerClass: type,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """메타러너를 학습하고 CV-MSE로 평가합니다.

        CV 방식:
          - K-Fold로 분할
          - 각 fold에서 학습 → 검증 데이터의 CATE 예측
          - T-transformation MSE: (Ỹ - τ̂(X)·T̃)²
            (Ground truth 없이도 평가 가능한 R-Risk)
        """
        kf = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
        oos_mse = []

        for train_idx, val_idx in kf.split(X):
            X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
            X_val, T_val, Y_val = X[val_idx], T[val_idx], Y[val_idx]

            learner = LearnerClass(config=self.config)
            learner.fit(X_tr, T_tr, Y_tr)
            cate_val = learner.predict_cate(X_val)

            # R-Risk: (Y - m̂(X) - τ̂(X)·(T - ê(X)))²
            # m̂(X)와 ê(X)를 간이 학습 (GPU 가속)
            from engine.gpu_factory import create_lgbm_regressor
            m = create_lgbm_regressor(self.config, lightweight=True)
            e = create_lgbm_regressor(self.config, lightweight=True)
            m.fit(X_tr, Y_tr)
            e.fit(X_tr, T_tr)

            m_val = m.predict(X_val)
            e_val = e.predict(X_val)

            Y_tilde = Y_val - m_val
            T_tilde = T_val - e_val
            residual = Y_tilde - cate_val * T_tilde
            mse = float(np.mean(residual ** 2))
            oos_mse.append(mse)

        # 전체 데이터로 최종 모델 학습
        final_learner = LearnerClass(config=self.config)
        final_learner.fit(X, T, Y)
        cate = final_learner.predict_cate(X)
        ate = float(np.mean(cate))

        return {
            "ate": ate,
            "cate": cate,
            "cv_mse": float(np.mean(oos_mse)),
        }
