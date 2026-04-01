# -*- coding: utf-8 -*-
"""RefutationCell — 진짜 인과 효과 반증 엔진.

Mock이 아닌 실제 모델 재학습 기반 반증 테스트를 수행합니다.
- Placebo Test: Treatment 무작위 셔플 → 모델 재학습 → Null ATE 분포
- Bootstrap CI: 비모수 부트스트랩 → ATE 신뢰구간
- Leave-One-Out Confounder: 교란 변수 제거 → ATE 안정성
- Subset Validation: 데이터 크기별 안정성 검증
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class RefutationCell(BaseCell):
    """실제 모델 재학습 기반 인과 효과 반증 셀.

    기존 SensitivityCell의 Mock 코드를 대체합니다.
    매 반증마다 DML 모델을 재학습하여 진짜 Null 분포를 생성합니다.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="refutation_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """전체 반증 파이프라인을 수행합니다.

        Args:
            inputs: CausalCell 출력 + 원본 데이터.
                필수: dataframe, feature_names, treatment_col, outcome_col,
                      ate, model_type, discrete_treatment

        Returns:
            반증 결과 딕셔너리 (각 테스트의 Pass/Fail, 분포, p-value).
        """
        self.validate_inputs(
            inputs,
            ["dataframe", "feature_names", "treatment_col", "outcome_col", "ate"],
        )

        df = inputs["dataframe"]
        X_cols = inputs["feature_names"]
        T_col = inputs["treatment_col"]
        Y_col = inputs["outcome_col"]
        original_ate = inputs["ate"]
        is_discrete = inputs.get("discrete_treatment", False)
        cfg = self.config.sensitivity

        results = {}

        # ── 1. Placebo Treatment Test ──
        if cfg.placebo_treatment:
            self.logger.info("🔬 [Refutation 1/4] Placebo Test 시작 (n=%d)", cfg.n_refutation_iter)
            results["placebo_test"] = self._placebo_test(
                df, T_col, Y_col, X_cols, original_ate,
                is_discrete, n_iter=cfg.n_refutation_iter,
            )

        # ── 2. Bootstrap CI ──
        self.logger.info("🔬 [Refutation 2/4] Bootstrap CI 시작 (n=%d)", cfg.n_bootstrap)
        results["bootstrap"] = self._bootstrap_ci(
            df, T_col, Y_col, X_cols, is_discrete,
            n_boot=cfg.n_bootstrap,
        )

        # ── 3. Leave-One-Out Confounder ──
        self.logger.info("🔬 [Refutation 3/4] Leave-One-Out Confounder 시작")
        results["leave_one_out"] = self._leave_one_out_confounder(
            df, T_col, Y_col, X_cols, original_ate, is_discrete,
        )

        # ── 4. Subset Validation ──
        self.logger.info("🔬 [Refutation 4/4] Subset Validation 시작")
        results["subset"] = self._subset_validation(
            df, T_col, Y_col, X_cols, original_ate, is_discrete,
        )

        # Overall 판정
        pass_count = sum(
            1 for v in results.values()
            if isinstance(v, dict) and v.get("status") == "Pass"
        )
        total = len(results)
        results["overall"] = {
            "pass_count": pass_count,
            "total": total,
            "status": "Pass" if pass_count >= total * 0.75 else "Fail",
        }

        self.logger.info(
            "🛡️ 반증 종합: %d/%d Pass → %s",
            pass_count, total, results["overall"]["status"],
        )

        return {**inputs, "refutation_results": results}

    # ──────────────────────────────────────────
    # 1. Placebo Treatment Test
    # ──────────────────────────────────────────
    def _placebo_test(
        self,
        df: pd.DataFrame,
        T_col: str,
        Y_col: str,
        X_cols: List[str],
        original_ate: float,
        is_discrete: bool,
        n_iter: int = 20,
    ) -> Dict[str, Any]:
        """Treatment를 무작위 셔플 후 모델 재학습 → Null ATE 분포 생성.

        H₀: Treatment가 Outcome에 영향 없음
        검증: 셔플된 Treatment로 추정한 ATE가 0 근처에 분포해야 함.
        """
        null_ates = []

        for i in range(n_iter):
            df_shuf = df.copy()
            df_shuf[T_col] = np.random.permutation(df[T_col].values)

            ate_null = self._fit_and_estimate_ate(
                df_shuf, T_col, Y_col, X_cols, is_discrete,
            )
            null_ates.append(ate_null)

            if (i + 1) % 5 == 0:
                self.logger.info("      Placebo iter %d/%d, null_ate=%.5f", i + 1, n_iter, ate_null)

        null_ates = np.array(null_ates)
        # p-value: |null_ate| ≥ |original_ate| 인 비율
        p_value = float(np.mean(np.abs(null_ates) >= np.abs(original_ate)))
        null_mean = float(np.mean(null_ates))
        null_std = float(np.std(null_ates))

        return {
            "null_mean": null_mean,
            "null_std": null_std,
            "p_value": p_value,
            "original_ate": original_ate,
            "n_iter": n_iter,
            "status": "Pass" if p_value < 0.05 else "Fail",
            "interpretation": (
                f"Placebo ATE 평균={null_mean:.5f} (≈0), "
                f"원래 ATE={original_ate:.5f}는 Null 분포에서 "
                f"{'이례적 (p={p_value:.3f}<0.05) → 진짜 효과' if p_value < 0.05 else '구별 불가'}"
            ),
        }

    # ──────────────────────────────────────────
    # 2. Bootstrap CI
    # ──────────────────────────────────────────
    def _bootstrap_ci(
        self,
        df: pd.DataFrame,
        T_col: str,
        Y_col: str,
        X_cols: List[str],
        is_discrete: bool,
        n_boot: int = 100,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """비모수 부트스트랩 → ATE 신뢰구간.

        정규분포 가정 없이,
        데이터를 복원추출하여 각 샘플의 ATE를 추정합니다.
        """
        boot_ates = []
        n = len(df)

        for i in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            df_boot = df.iloc[idx].reset_index(drop=True)

            ate_boot = self._fit_and_estimate_ate(
                df_boot, T_col, Y_col, X_cols, is_discrete,
            )
            boot_ates.append(ate_boot)

            if (i + 1) % 25 == 0:
                self.logger.info("      Bootstrap iter %d/%d", i + 1, n_boot)

        boot_ates = np.array(boot_ates)
        ci_lower = float(np.percentile(boot_ates, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_ates, 100 * (1 - alpha / 2)))
        mean_ate = float(np.mean(boot_ates))
        std_ate = float(np.std(boot_ates))

        # 0이 CI에 포함되면 유의하지 않음
        significant = not (ci_lower <= 0 <= ci_upper)

        return {
            "mean_ate": mean_ate,
            "std_ate": std_ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_boot": n_boot,
            "significant": significant,
            "status": "Pass" if significant else "Fail",
            "interpretation": (
                f"Bootstrap 95% CI: [{ci_lower:.5f}, {ci_upper:.5f}], "
                f"{'0을 포함하지 않음 → 통계적으로 유의' if significant else '0을 포함 → 유의하지 않음'}"
            ),
        }

    # ──────────────────────────────────────────
    # 3. Leave-One-Out Confounder
    # ──────────────────────────────────────────
    def _leave_one_out_confounder(
        self,
        df: pd.DataFrame,
        T_col: str,
        Y_col: str,
        X_cols: List[str],
        original_ate: float,
        is_discrete: bool,
    ) -> Dict[str, Any]:
        """교란 변수를 하나씩 제거 후 ATE 변화 측정.

        목적: 각 교란 변수의 기여도와 ATE의 안정성 파악.
        ATE 부호가 뒤집히면 해당 변수가 핵심 교란임.
        """
        loo_results = []

        for excluded in X_cols:
            remaining = [c for c in X_cols if c != excluded]
            if not remaining:
                continue

            ate_loo = self._fit_and_estimate_ate(
                df, T_col, Y_col, remaining, is_discrete,
            )
            delta = ate_loo - original_ate
            sign_flip = (np.sign(ate_loo) != np.sign(original_ate))

            loo_results.append({
                "excluded_variable": excluded,
                "ate_without": float(ate_loo),
                "delta": float(delta),
                "pct_change": float(abs(delta) / (abs(original_ate) + 1e-10) * 100),
                "sign_flip": bool(sign_flip),
            })
            self.logger.info(
                "      LOO [-%s]: ATE=%.5f (Δ=%.5f, %s)",
                excluded, ate_loo, delta,
                "⚠️ 부호 반전!" if sign_flip else "안정",
            )

        # 최대 변화율 기준 판정
        max_change = max(r["pct_change"] for r in loo_results) if loo_results else 0
        any_flip = any(r["sign_flip"] for r in loo_results)

        return {
            "results": loo_results,
            "max_pct_change": max_change,
            "any_sign_flip": any_flip,
            "status": "Fail" if any_flip else ("Pass" if max_change < 50 else "Warning"),
            "interpretation": (
                f"최대 ATE 변화: {max_change:.1f}%. "
                + ("부호 반전 감지 → 핵심 교란 존재 가능" if any_flip else "모든 변수 제거 후에도 방향 일관")
            ),
        }

    # ──────────────────────────────────────────
    # 4. Subset Validation
    # ──────────────────────────────────────────
    def _subset_validation(
        self,
        df: pd.DataFrame,
        T_col: str,
        Y_col: str,
        X_cols: List[str],
        original_ate: float,
        is_discrete: bool,
        fractions: list = None,
    ) -> Dict[str, Any]:
        """서브샘플 안정성: 데이터 비율별 ATE 안정성 검증.

        50%, 70%, 90% 서브샘플에서 ATE가 안정적이면 견고.
        """
        if fractions is None:
            fractions = [0.5, 0.7, 0.9]

        subset_results = []
        n = len(df)

        for frac in fractions:
            sub_n = int(n * frac)
            idx = np.random.choice(n, sub_n, replace=False)
            df_sub = df.iloc[idx].reset_index(drop=True)

            ate_sub = self._fit_and_estimate_ate(
                df_sub, T_col, Y_col, X_cols, is_discrete,
            )
            delta = ate_sub - original_ate
            stability = 1.0 - abs(delta) / (abs(original_ate) + 1e-10)

            subset_results.append({
                "fraction": frac,
                "n_samples": sub_n,
                "ate": float(ate_sub),
                "delta": float(delta),
                "stability": float(max(0, stability)),
            })
            self.logger.info(
                "      Subset %.0f%% (n=%d): ATE=%.5f, stability=%.3f",
                frac * 100, sub_n, ate_sub, stability,
            )

        avg_stability = float(np.mean([r["stability"] for r in subset_results]))

        return {
            "results": subset_results,
            "avg_stability": avg_stability,
            "status": "Pass" if avg_stability > 0.8 else "Fail",
            "interpretation": (
                f"평균 안정성: {avg_stability:.3f} "
                f"{'(>0.8 → 견고)' if avg_stability > 0.8 else '(<0.8 → 불안정)'}"
            ),
        }

    # ──────────────────────────────────────────
    # 공통 유틸: DML 재학습 → ATE
    # ──────────────────────────────────────────
    def _fit_and_estimate_ate(
        self,
        df: pd.DataFrame,
        T_col: str,
        Y_col: str,
        X_cols: List[str],
        is_discrete: bool,
    ) -> float:
        """DML 모델을 학습하고 ATE를 반환합니다.

        경량 설정(CV=2, estimators=100)으로 빠르게 재학습합니다.
        """
        from econml.dml import LinearDML
        from engine.gpu_factory import create_lgbm_regressor

        Y = df[Y_col].values.astype(np.float64)
        T = df[T_col].values.astype(np.float64)
        X = df[X_cols].values.astype(np.float64)

        # 연속형 Treatment 정규화
        if not is_discrete:
            t_mean, t_std = float(T.mean()), float(T.std())
            if t_std > 0:
                T = (T - t_mean) / t_std

        # 경량 nuisance 모델 (GPU 가속 + 반증 최적화)
        model_y = create_lgbm_regressor(self.config, lightweight=True)
        model_t = create_lgbm_regressor(self.config, lightweight=True)

        model = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=is_discrete,
            cv=2,  # 반증용 경량 CV
            random_state=self.config.data.random_seed,
        )
        model.fit(Y=Y, T=T, X=X)

        # ATE = CATE 평균
        cate = model.effect(X)
        return float(np.mean(cate))
