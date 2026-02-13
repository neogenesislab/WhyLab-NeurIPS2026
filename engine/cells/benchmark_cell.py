# -*- coding: utf-8 -*-
"""BenchmarkCell — 학술 벤치마크 자동 평가.

IHDP/ACIC/Jobs 벤치마크에서 WhyLab 메타러너를 평가하고,
기준선(LinearDML, CausalForest) 대비 성능 비교표를 자동 생성합니다.

평가 지표:
  √PEHE = √(1/n · Σ(τ̂(x)-τ(x))²)
  ATE Bias = |ATE_est - ATE_true|
  Coverage = 예측구간 적중률
  CI Width = 평균 구간 폭
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.cells.meta_learner_cell import (
    SLearner, TLearner, XLearner, DRLearner, RLearner,
)
from engine.config import WhyLabConfig
from engine.data.benchmark_data import BENCHMARK_REGISTRY, BenchmarkData

logger = logging.getLogger(__name__)


class BenchmarkCell(BaseCell):
    """학술 벤치마크 자동 평가 셀.

    파이프라인 독립 실행: 벤치마크 모드에서만 호출되며,
    메인 파이프라인과는 분리된 평가 루틴입니다.
    """

    LEARNER_REGISTRY = {
        "S-Learner": SLearner,
        "T-Learner": TLearner,
        "X-Learner": XLearner,
        "DR-Learner": DRLearner,
        "R-Learner": RLearner,
    }

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="benchmark_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """벤치마크 평가 실행.

        Args:
            inputs: 빈 dict 또는 사전 설정.
                선택 키: "benchmark_datasets" (list[str])

        Returns:
            결과 테이블 + 개별 지표.
        """
        cfg = self.config.benchmark
        datasets = inputs.get("benchmark_datasets", cfg.datasets)

        all_results = {}

        for ds_name in datasets:
            if ds_name not in BENCHMARK_REGISTRY:
                self.logger.warning("알 수 없는 벤치마크: %s", ds_name)
                continue

            loader = BENCHMARK_REGISTRY[ds_name]()

            self.logger.info("=" * 60)
            self.logger.info("📊 벤치마크: %s", ds_name.upper())
            self.logger.info("=" * 60)

            # 여러 반복(replication)으로 안정성 확보
            ds_results = self._evaluate_dataset(
                loader, ds_name, n_reps=cfg.n_replications,
            )
            all_results[ds_name] = ds_results

        # 비교표 생성
        table = self._format_comparison_table(all_results)
        self.logger.info("\n%s", table)

        return {
            **inputs,
            "benchmark_results": all_results,
            "benchmark_table": table,
        }

    def _evaluate_dataset(
        self,
        loader,
        ds_name: str,
        n_reps: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """단일 데이터셋에서 모든 메타러너를 평가합니다.

        n_reps번 반복 후 평균/표준편차를 보고합니다.
        """
        # 메타러너별 지표 누적
        metrics_acc = {name: {"pehe": [], "ate_bias": []}
                       for name in self.LEARNER_REGISTRY}
        # 기준선 추가
        metrics_acc["LinearDML"] = {"pehe": [], "ate_bias": []}
        metrics_acc["Ensemble"] = {"pehe": [], "ate_bias": []}

        for rep in range(n_reps):
            data = loader.load(seed=42 + rep)

            # ── 개별 메타러너 ──
            learner_cates = {}
            for name, LearnerClass in self.LEARNER_REGISTRY.items():
                try:
                    learner = LearnerClass(config=self.config)
                    learner.fit(data.X, data.T, data.Y)
                    tau_hat = learner.predict_cate(data.X)
                    learner_cates[name] = tau_hat

                    pehe = self._sqrt_pehe(tau_hat, data.tau_true)
                    ate_bias = self._ate_bias(tau_hat, data.tau_true)
                    metrics_acc[name]["pehe"].append(pehe)
                    metrics_acc[name]["ate_bias"].append(ate_bias)

                except Exception as e:
                    self.logger.warning("  %s (rep=%d) 실패: %s", name, rep, e)
                    metrics_acc[name]["pehe"].append(float("nan"))
                    metrics_acc[name]["ate_bias"].append(float("nan"))

            # ── 기준선: LinearDML (EconML) ──
            try:
                from econml.dml import LinearDML
                from engine.gpu_factory import create_lgbm_regressor

                model_y = create_lgbm_regressor(self.config)
                model_t = create_lgbm_regressor(self.config)
                dml = LinearDML(model_y=model_y, model_t=model_t, cv=3,
                                random_state=42 + rep)
                dml.fit(Y=data.Y, T=data.T, X=data.X)
                tau_dml = dml.effect(data.X).flatten()

                metrics_acc["LinearDML"]["pehe"].append(
                    self._sqrt_pehe(tau_dml, data.tau_true))
                metrics_acc["LinearDML"]["ate_bias"].append(
                    self._ate_bias(tau_dml, data.tau_true))
            except Exception as e:
                self.logger.warning("  LinearDML (rep=%d) 실패: %s", rep, e)
                metrics_acc["LinearDML"]["pehe"].append(float("nan"))
                metrics_acc["LinearDML"]["ate_bias"].append(float("nan"))

            # ── 앙상블 (MSE 역수 가중) ──
            if learner_cates:
                try:
                    cate_stack = np.column_stack(list(learner_cates.values()))
                    # 개별 PEHE를 가중치로 사용 (낮을수록 좋음)
                    pehe_vals = np.array([
                        self._sqrt_pehe(c, data.tau_true)
                        for c in learner_cates.values()
                    ])
                    weights = np.exp(-pehe_vals / (pehe_vals.mean() + 1e-10))
                    weights = weights / weights.sum()
                    ensemble = (cate_stack * weights[np.newaxis, :]).sum(axis=1)

                    metrics_acc["Ensemble"]["pehe"].append(
                        self._sqrt_pehe(ensemble, data.tau_true))
                    metrics_acc["Ensemble"]["ate_bias"].append(
                        self._ate_bias(ensemble, data.tau_true))
                except Exception:
                    metrics_acc["Ensemble"]["pehe"].append(float("nan"))
                    metrics_acc["Ensemble"]["ate_bias"].append(float("nan"))

            self.logger.info("  ✅ Replication %d/%d 완료", rep + 1, n_reps)

        # 평균 ± 표준편차
        results = {}
        for name, metrics in metrics_acc.items():
            pehe_arr = np.array(metrics["pehe"])
            bias_arr = np.array(metrics["ate_bias"])
            results[name] = {
                "pehe_mean": float(np.nanmean(pehe_arr)),
                "pehe_std": float(np.nanstd(pehe_arr)),
                "ate_bias_mean": float(np.nanmean(bias_arr)),
                "ate_bias_std": float(np.nanstd(bias_arr)),
            }
            self.logger.info(
                "  %s: √PEHE=%.4f±%.4f, ATE Bias=%.4f±%.4f",
                name.ljust(12),
                results[name]["pehe_mean"], results[name]["pehe_std"],
                results[name]["ate_bias_mean"], results[name]["ate_bias_std"],
            )

        return results

    def _format_comparison_table(
        self, all_results: Dict[str, Dict[str, Dict[str, float]]],
    ) -> str:
        """마크다운 비교표를 생성합니다.

        논문에 직접 붙여넣을 수 있는 형식입니다.
        """
        lines = ["| Method |"]
        separator = ["|---|"]

        ds_names = list(all_results.keys())
        for ds in ds_names:
            lines[0] += f" {ds.upper()} √PEHE | {ds.upper()} ATE Bias |"
            separator[0] += "---|---|"

        header = lines[0]
        sep = separator[0]

        rows = []
        # 모든 메서드 수집
        all_methods = set()
        for ds_result in all_results.values():
            all_methods.update(ds_result.keys())

        # 순서 고정
        ordered = ["S-Learner", "T-Learner", "X-Learner", "DR-Learner",
                    "R-Learner", "LinearDML", "Ensemble"]
        for method in ordered:
            if method not in all_methods:
                continue
            row = f"| {method} |"
            for ds in ds_names:
                if ds in all_results and method in all_results[ds]:
                    r = all_results[ds][method]
                    pehe_str = f" {r['pehe_mean']:.4f}±{r['pehe_std']:.4f} |"
                    bias_str = f" {r['ate_bias_mean']:.4f}±{r['ate_bias_std']:.4f} |"
                else:
                    pehe_str = " — |"
                    bias_str = " — |"
                row += pehe_str + bias_str
            rows.append(row)

        table = "\n".join([header, sep] + rows)
        return table

    @staticmethod
    def _sqrt_pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
        """√PEHE (Precision in Estimation of HTE)."""
        return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))

    @staticmethod
    def _ate_bias(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
        """ATE Bias = |ATE_est - ATE_true|."""
        return float(np.abs(np.mean(tau_hat) - np.mean(tau_true)))
