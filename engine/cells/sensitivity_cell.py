# -*- coding: utf-8 -*-
"""SensitivityCell — 인과 효과 견고성 검증.

추정된 인과 효과가 우연에 의한 것이 아님을 증명하기 위해
민감도 분석(Sensitivity Analysis) 및 반박(Refutation) 테스트를 수행합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class SensitivityCell(BaseCell):
    """인과 효과의 견고성(Robustness)을 검증하는 셀.

    Args:
        config: WhyLab 전역 설정 객체.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="sensitivity_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """민감도 분석을 수행합니다.

        Args:
            inputs: CausalCell의 출력.
                필수 키: "dataframe", "feature_names", "treatment_col", "outcome_col", "model"

        Returns:
            검증 결과 딕셔너리 (Pass/Fail 여부 포함).
        """
        cfg = self.config.sensitivity
        if not cfg.enabled:
            self.logger.info("민감도 분석 비활성화됨 (Skipped)")
            return {"sensitivity_results": {"status": "Skipped"}}

        self.logger.info("🛡️ 민감도 분석 시작 (Simulations=%d)", cfg.n_simulations)
        
        df: pd.DataFrame = inputs["dataframe"]
        feature_names = inputs["feature_names"]
        treatment_col = inputs["treatment_col"]
        outcome_col = inputs["outcome_col"]
        original_ate = inputs["ate"]
        model = inputs["model"] # 학습된 모델 (재사용 불가 시 새로 학습해야 함)
        
        results = {}

        # ──────────────────────────────────────────
        # 1. Placebo Treatment Test (RefutationCell로 이관)
        # ──────────────────────────────────────────
        # 진짜 반증은 RefutationCell에서 수행합니다.
        # 여기서는 결과를 전달받아 포함합니다.
        if cfg.placebo_treatment:
            refutation = inputs.get("refutation_results")
            if refutation and "placebo_test" in refutation:
                results["placebo_test"] = refutation["placebo_test"]
                self.logger.info("   Placebo (RefutationCell): %s", results["placebo_test"]["status"])
            else:
                self.logger.info("   Placebo: RefutationCell 미실행 → Skip")
                results["placebo_test"] = {"status": "Not Run", "note": "RefutationCell 선행 필요"}

        # ──────────────────────────────────────────
        # 2. Random Common Cause → Bootstrap CI (RefutationCell로 이관)
        # ──────────────────────────────────────────
        if cfg.random_common_cause:
            refutation = inputs.get("refutation_results")
            if refutation and "bootstrap" in refutation:
                results["random_common_cause"] = {
                    "mean_effect": refutation["bootstrap"]["mean_ate"],
                    "stability": refutation["subset"]["avg_stability"] if "subset" in refutation else 0.9,
                    "bootstrap_ci": [refutation["bootstrap"]["ci_lower"], refutation["bootstrap"]["ci_upper"]],
                    "status": refutation["bootstrap"]["status"],
                }
                self.logger.info("   RCC→Bootstrap (RefutationCell): %s", results["random_common_cause"]["status"])
            else:
                self.logger.info("   RCC: RefutationCell 미실행 → Skip")
                results["random_common_cause"] = {"status": "Not Run", "stability": 0.0}

        # ──────────────────────────────────────────
        # 3. E-value (미관측 교란 변수에 대한 견고성)
        # ──────────────────────────────────────────
        # E-value: 관측 불가 교란 요인이 결과를 설명하려면
        # 얼마나 강한 상관을 가져야 하는지 측정.
        # 높을수록 견고함 (≥2.0 이면 양호).
        if cfg.e_value:
            self.logger.info("▶ E-value 계산 중...")
            e_val = self._compute_e_value(original_ate)
            e_val_ci = self._compute_e_value(
                min(abs(inputs.get("ate_ci_lower", 0)), abs(inputs.get("ate_ci_upper", 0)))
            )
            results["e_value"] = {
                "point": float(e_val),
                "ci_bound": float(e_val_ci),
                "interpretation": (
                    "매우 견고 — 미관측 교란이 결과를 뒤집기 어려움"
                    if e_val >= 3.0 else
                    "견고 — 상당한 교란이 있어야 결과가 바뀜"
                    if e_val >= 2.0 else
                    "보통 — 적당한 교란에 민감할 수 있음"
                    if e_val >= 1.5 else
                    "취약 — 작은 교란에도 결과가 바뀔 수 있음"
                ),
                "status": "Pass" if e_val >= 1.5 else "Fail",
            }
            self.logger.info("   E-value: %.2f (CI bound: %.2f, %s)",
                             e_val, e_val_ci, results["e_value"]["status"])

        # ──────────────────────────────────────────
        # 4. Overlap (Positivity) 진단
        # ──────────────────────────────────────────
        # Propensity Score 분포를 추정하여
        # Treatment/Control 간 겹침(overlap)을 평가.
        if cfg.overlap:
            self.logger.info("▶ Overlap(Positivity) 진단 중...")
            overlap_result = self._diagnose_overlap(df, treatment_col, feature_names)
            results["overlap"] = overlap_result
            self.logger.info("   Overlap Score: %.3f (%s)",
                             overlap_result["overlap_score"], overlap_result["status"])

        # ──────────────────────────────────────────
        # 5. GATES/CLAN (이질성 심화 분석)
        # ──────────────────────────────────────────
        # 그룹 평균 처치 효과(GATES) + 최지영-리-넬로-인퍼런스(CLAN)
        cate_preds = inputs.get("cate_predictions", np.array([]))
        if cfg.gates and len(cate_preds) > 0:
            self.logger.info("▶ GATES/CLAN 분석 중...")
            gates_result = self._compute_gates_clan(
                df, cate_preds, feature_names, cfg.n_gates_groups
            )
            results["gates"] = gates_result
            self.logger.info("   GATES 그룹: %d, 이질성 F-stat: %.2f",
                             len(gates_result["groups"]), gates_result["f_statistic"])

        return {"sensitivity_results": results}

    # ──────────────────────────────────────────
    # E-value 계산
    # ──────────────────────────────────────────
    @staticmethod
    def _compute_e_value(estimate: float) -> float:
        """E-value를 계산합니다.

        E-value = RR + sqrt(RR * (RR - 1))
        여기서 RR ≈ exp(|ATE|) (로그 스케일 근사).
        """
        rr = np.exp(abs(estimate))
        return float(rr + np.sqrt(rr * (rr - 1)))

    # ──────────────────────────────────────────
    # Overlap 진단
    # ──────────────────────────────────────────
    @staticmethod
    def _diagnose_overlap(
        df: pd.DataFrame,
        treatment_col: str,
        feature_names: list,
    ) -> dict:
        """Propensity Score 기반 Overlap 진단."""
        from sklearn.linear_model import LogisticRegression

        X = df[feature_names].values
        T = df[treatment_col].values

        # 이진 처치가 아니면 중앙값으로 이진화
        if len(np.unique(T)) > 2:
            T = (T > np.median(T)).astype(int)

        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X, T)
        ps = lr.predict_proba(X)[:, 1]

        # 통계
        ps_treated = ps[T == 1]
        ps_control = ps[T == 0]

        # Overlap Score: Bhattacharyya coefficient
        bins = np.linspace(0, 1, 21)
        hist_t, _ = np.histogram(ps_treated, bins=bins, density=True)
        hist_c, _ = np.histogram(ps_control, bins=bins, density=True)
        # 정규화 후 Bhattacharyya
        hist_t = hist_t / (hist_t.sum() + 1e-9)
        hist_c = hist_c / (hist_c.sum() + 1e-9)
        overlap_score = float(np.sum(np.sqrt(hist_t * hist_c)))

        # Extreme weights (위치 확인)
        iptw_weights = 1 / np.clip(ps * T + (1 - ps) * (1 - T), 0.01, 1.0)
        max_weight = float(np.max(iptw_weights))
        pct_extreme = float(np.mean(iptw_weights > 10) * 100)

        return {
            "overlap_score": round(overlap_score, 3),
            "ps_stats": {
                "treated_mean": round(float(ps_treated.mean()), 3),
                "treated_std": round(float(ps_treated.std()), 3),
                "control_mean": round(float(ps_control.mean()), 3),
                "control_std": round(float(ps_control.std()), 3),
            },
            "iptw_max_weight": round(max_weight, 2),
            "pct_extreme_weights": round(pct_extreme, 1),
            "ps_histogram": {
                "bin_edges": [round(b, 2) for b in bins.tolist()],
                "treated_counts": hist_t.tolist(),
                "control_counts": hist_c.tolist(),
            },
            "interpretation": (
                "충분한 Overlap — 처치/통제 그룹 간 균형 양호"
                if overlap_score > 0.7 else
                "보통 — 일부 구간에서 Overlap 부족"
                if overlap_score > 0.4 else
                "Overlap 부족 — Positivity 위반 위험"
            ),
            "status": "Pass" if overlap_score > 0.4 else "Fail",
        }

    # ──────────────────────────────────────────
    # GATES / CLAN
    # ──────────────────────────────────────────
    @staticmethod
    def _compute_gates_clan(
        df: pd.DataFrame,
        cate_preds: np.ndarray,
        feature_names: list,
        n_groups: int = 4,
    ) -> dict:
        """GATES(Group Average Treatment Effects) + CLAN(Classification Analysis).

        CATE 예측값 기준으로 n_groups 분위 그룹을 만들고,
        각 그룹의 평균 CATE와 주요 피처 특성을 분석합니다.
        """
        # CATE 분위 그룹 할당
        quantiles = np.quantile(cate_preds, np.linspace(0, 1, n_groups + 1))
        group_labels = np.digitize(cate_preds, quantiles[1:-1])  # 0 ~ n_groups-1

        groups = []
        group_means = []

        for g in range(n_groups):
            mask = group_labels == g
            g_cate = cate_preds[mask]
            n = int(mask.sum())

            if n == 0:
                continue

            mean_cate = float(g_cate.mean())
            std_cate = float(g_cate.std())
            group_means.append(mean_cate)

            # CLAN: 각 그룹의 피처 평균
            g_df = df.iloc[mask]
            clan_features = {}
            for feat in feature_names[:5]:  # 상위 5개 피처
                if feat in g_df.columns:
                    clan_features[feat] = round(float(g_df[feat].mean()), 3)

            groups.append({
                "group_id": g + 1,
                "label": f"Q{g + 1}",
                "n": n,
                "mean_cate": round(mean_cate, 4),
                "std_cate": round(std_cate, 4),
                "ci_lower": round(mean_cate - 1.96 * std_cate / np.sqrt(n), 4),
                "ci_upper": round(mean_cate + 1.96 * std_cate / np.sqrt(n), 4),
                "clan_features": clan_features,
            })

        # F-statistic (이질성 테스트)
        grand_mean = float(np.mean(cate_preds))
        k = len(groups)
        if k > 1:
            ss_between = sum(
                g["n"] * (g["mean_cate"] - grand_mean) ** 2 for g in groups
            ) / (k - 1)
            ss_within = sum(
                g["n"] * g["std_cate"] ** 2 for g in groups
            ) / (len(cate_preds) - k)
            f_stat = ss_between / (ss_within + 1e-9)
        else:
            f_stat = 0.0

        return {
            "n_groups": k,
            "groups": groups,
            "f_statistic": round(float(f_stat), 2),
            "heterogeneity": (
                "강한 이질성 — 세그먼트별 차등 전략 필수"
                if f_stat > 10 else
                "이질성 존재 — 타겟팅 정책 고려 필요"
                if f_stat > 3 else
                "약한 이질성 — 일괄 정책 적용 가능"
            ),
            "status": "Info",  # GATES는 Pass/Fail이 아닌 정보 제공
        }
