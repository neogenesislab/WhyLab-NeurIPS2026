# -*- coding: utf-8 -*-
"""ExportCell — JSON 직렬화 (대시보드 연동).

파이프라인의 최종 셀. 분석 결과를 JSON으로 직렬화하여
Next.js 대시보드가 정적으로 소비할 수 있는 형태로 내보냅니다.

출력 파일: dashboard/public/data/causal_results.json

입력 키 (CausalCell 출력):
    - "ate", "ate_ci_lower", "ate_ci_upper"
    - "cate_predictions", "cate_ci_lower", "cate_ci_upper"
    - "dataframe", "feature_names"
    - "treatment_col", "outcome_col"
    - "dag_edges" (DataCell에서 통과)

출력 키:
    - "json_path": str (저장된 JSON 파일 경로)
    - "json_data": dict (직렬화된 딕셔너리)
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class ExportCell(BaseCell):
    """분석 결과를 대시보드용 JSON으로 직렬화하는 셀.

    Args:
        config: WhyLab 전역 설정 객체.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="export_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과를 JSON 파일로 내보냅니다.

        Args:
            inputs: CausalCell의 출력 + DAG 정보.
                필수 키: "ate", "cate_predictions", "dataframe", "feature_names"

        Returns:
            저장된 JSON 파일 경로와 데이터를 담은 딕셔너리.
        """
        self.validate_inputs(
            inputs,
            ["ate", "cate_predictions", "dataframe", "feature_names"],
        )

        self.logger.info("JSON 직렬화 시작")

        # ──────────────────────────────────────────
        # 1. 핵심 결과 조립
        # ──────────────────────────────────────────
        # CATE 데이터 안전하게 가져오기
        cate_preds = inputs.get("cate_predictions", np.array([]))
        if len(cate_preds) > 0:
            cate_mean = float(cate_preds.mean())
            cate_std = float(cate_preds.std())
            # 히스토그램용 샘플링 (최대 1000개)
            sample_indices = np.random.choice(len(cate_preds), size=min(1000, len(cate_preds)), replace=False)
            cate_sample = cate_preds[sample_indices].tolist()
        else:
            cate_mean = 0.0
            cate_std = 0.0
            cate_sample = []

        # ──────────────────────────────────────────
        # 공통 변수 준비
        # ──────────────────────────────────────────
        df: pd.DataFrame = inputs["dataframe"]
        feature_names: List[str] = inputs["feature_names"]
        scenario = inputs.get("scenario", "Unknown")
        model_type = inputs.get("model_type", self.config.dml.model_type)
        treatment_col = inputs.get("treatment_col", "credit_limit")
        outcome_col = inputs.get("outcome_col", "is_default")
        ate_value = inputs.get("ate", 0.0)

        json_data = {
            # ── ATE (TypeScript: ate.value/ci_lower/ci_upper/alpha/description)
            "ate": {
                "value": ate_value,
                "ci_lower": inputs.get("ate_ci_lower", 0.0),
                "ci_upper": inputs.get("ate_ci_upper", 0.0),
                "alpha": getattr(self.config.dml, 'alpha', 0.05),
                "description": self._generate_ate_description(
                    ate_value, treatment_col, outcome_col, scenario
                ),
            },
            # ── CATE Distribution (TypeScript: cate_distribution)
            "cate_distribution": {
                "mean": cate_mean,
                "std": cate_std,
                "min": float(np.min(cate_preds)) if len(cate_preds) > 0 else 0.0,
                "max": float(np.max(cate_preds)) if len(cate_preds) > 0 else 0.0,
                "histogram": self._compute_histogram(cate_preds)
                if len(cate_preds) > 0
                else {"bin_edges": [], "counts": []},
            },
            # ── Segments (아래에서 추가)
            "segments": self._build_segments(df, cate_preds),
            # ── DAG
            "dag": {
                "nodes": self._extract_dag_nodes(inputs.get("dag_edges", [])),
                "edges": [
                    {"source": s, "target": t}
                    for s, t in inputs.get("dag_edges", [])
                ],
            },
            # ── Metadata (TypeScript: metadata)
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "scenario": scenario,
                "model_type": model_type,
                "n_samples": len(df),
                "feature_names": feature_names,
                "treatment_col": treatment_col,
                "outcome_col": outcome_col,
            },
            # ── Sensitivity (TypeScript: sensitivity.status + sub-tests)
            "sensitivity": self._build_sensitivity(inputs),
        }

        # ──────────────────────────────────────────
        # 4-1. Explainability (SHAP + Counterfactual)
        # ──────────────────────────────────────────
        feature_importance = inputs.get("feature_importance", {})
        counterfactuals = inputs.get("counterfactuals", [])

        if feature_importance:
            sorted_fi = sorted(
                feature_importance.items(), key=lambda x: -x[1]
            )
            json_data["explainability"] = {
                "feature_importance": [
                    {"feature": k, "importance": float(v)}
                    for k, v in sorted_fi
                ],
                "counterfactuals": [
                    {
                        "user_id": cf.get("user_id", 0),
                        "original_cate": float(cf.get("original_cate", 0)),
                        "counterfactual_cate": float(cf.get("counterfactual_cate", 0)),
                        "diff": float(cf.get("diff", 0)),
                        "description": cf.get("description", ""),
                    }
                    for cf in counterfactuals
                ],
            }

        # ──────────────────────────────────────────
        # 4-2. Estimation Accuracy (Ground Truth 검증)
        # ──────────────────────────────────────────
        estimation_accuracy = inputs.get("estimation_accuracy", {})
        if estimation_accuracy:
            json_data["estimation_accuracy"] = {
                "rmse": float(estimation_accuracy.get("rmse", 0)),
                "mae": float(estimation_accuracy.get("mae", 0)),
                "bias": float(estimation_accuracy.get("bias", 0)),
                "coverage_rate": float(estimation_accuracy.get("coverage_rate", 0)),
                "correlation": float(estimation_accuracy.get("correlation", 0)),
                "n_samples": int(estimation_accuracy.get("n_samples", 0)),
            }

        # ──────────────────────────────────────────
        # 4-3. AI Insights (규칙 기반, 대시보드용)
        # ──────────────────────────────────────────
        corr = estimation_accuracy.get("correlation", 0) if estimation_accuracy else 0
        ate_ci_lo = inputs.get("ate_ci_lower", 0.0)
        ate_ci_hi = inputs.get("ate_ci_upper", 0.0)
        is_significant = not (ate_ci_lo <= 0 <= ate_ci_hi)
        abs_ate = abs(ate_value)
        direction = "감소" if ate_value < 0 else "증가"

        if abs_ate > 0.1:
            effect_size, effect_label = "large", "큰"
        elif abs_ate > 0.01:
            effect_size, effect_label = "medium", "중간 수준의"
        else:
            effect_size, effect_label = "small", "작은"

        top_features_list = sorted(
            feature_importance.items(), key=lambda x: -x[1]
        )[:3] if feature_importance else []

        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")
        scenario_name = inputs.get("scenario_name", "Unknown")

        sig_text = "통계적으로 유의합니다" if is_significant else "통계적으로 유의하지 않습니다"
        summary_text = (
            f"{scenario_name} 분석 결과, {treatment_col}의 변화는 {outcome_col}을(를) "
            f"평균 {abs_ate*100:.2f}%p {direction}시키는 {effect_label} 효과를 보였습니다. "
            f"95% 신뢰구간 [{ate_ci_lo:.4f}, {ate_ci_hi:.4f}]을 고려하면 이 결과는 {sig_text}."
        )

        top_feature_name = top_features_list[0][0] if top_features_list else ""
        if not is_significant:
            rec_text = f"{treatment_col}의 {outcome_col}에 대한 효과가 통계적으로 유의하지 않습니다."
        elif top_feature_name:
            rec_text = f"특히 {top_feature_name}에 따라 효과 이질성이 크므로, 세그먼트별 차등 전략이 유효합니다."
        else:
            rec_text = "정책 변경 시 효과가 기대됩니다."

        json_data["ai_insights"] = {
            "summary": summary_text,
            "headline": f"{'✅' if is_significant else '⚠️'} {treatment_col} → {outcome_col}: ATE = {ate_value:.4f} ({direction} {abs_ate*100:.1f}%p)",
            "significance": "유의함" if is_significant else "유의하지 않음",
            "effect_size": effect_size,
            "effect_direction": direction,
            "top_drivers": [
                {"feature": f, "importance": round(v, 4)}
                for f, v in top_features_list
            ],
            "model_quality": (
                "excellent" if corr > 0.95 else
                "good" if corr > 0.8 else
                "moderate" if corr > 0.5 else "poor"
            ),
            "model_quality_label": (
                "우수" if corr > 0.95 else
                "양호" if corr > 0.8 else
                "보통" if corr > 0.5 else "미흡"
            ),
            "correlation": round(corr, 3),
            "rmse": round(estimation_accuracy.get("rmse", 0), 4) if estimation_accuracy else 0,
            "recommendation": rec_text,
            "generated_by": "rule_based",
        }

        # ──────────────────────────────────────────
        # 4-4. Debate 판결 (Phase 2)
        # ──────────────────────────────────────────
        debate_summary = inputs.get("debate_summary")
        if debate_summary:
            debate_export = {
                "verdict": debate_summary.get("verdict", "UNKNOWN"),
                "confidence": debate_summary.get("confidence", 0),
                "pro_score": debate_summary.get("pro_score", 0),
                "con_score": debate_summary.get("con_score", 0),
                "rounds": debate_summary.get("rounds", 0),
                "recommendation": debate_summary.get("recommendation", ""),
                "pro_evidence": debate_summary.get("pro_evidence", []),
                "con_evidence": debate_summary.get("con_evidence", []),
            }
            # LLM 토론 결과 포함 (Phase 9)
            llm_debate = debate_summary.get("llm_debate", {})
            if llm_debate:
                debate_export["llm_debate"] = {
                    "llm_active": llm_debate.get("llm_active", False),
                    "model": llm_debate.get("model", "rule_based"),
                    "advocate_argument": llm_debate.get("advocate_argument", ""),
                    "critic_argument": llm_debate.get("critic_argument", ""),
                    "judge_verdict": llm_debate.get("judge_verdict", ""),
                }
                # ai_insights의 generated_by도 LLM 상태 반영
                if "ai_insights" in json_data:
                    json_data["ai_insights"]["generated_by"] = (
                        f"llm:{llm_debate.get('model', 'unknown')}"
                        if llm_debate.get("llm_active")
                        else "rule_based"
                    )
            json_data["debate"] = debate_export

        # ──────────────────────────────────────────
        # 4-5. Conformal CI (Phase 2)
        # ──────────────────────────────────────────
        conformal = inputs.get("conformal_results")
        if conformal:
            json_data["conformal"] = {
                "mode": conformal.get("mode", "split"),
                "coverage": conformal.get("coverage", 0),
                "target_coverage": conformal.get("target_coverage", 0.95),
                "mean_width": conformal.get("mean_width", 0),
                "width_std": conformal.get("width_std", 0),
                "alpha": conformal.get("alpha", 0.05),
                "ci_lower_mean": conformal.get("ci_lower_mean", 0),
                "ci_upper_mean": conformal.get("ci_upper_mean", 0),
                "adaptive": conformal.get("adaptive", False),
                "interpretation": conformal.get("interpretation", ""),
            }

        # ──────────────────────────────────────────
        # 4-6. MetaLearner 개별 결과 (Phase 2)
        # ──────────────────────────────────────────
        meta_results = inputs.get("meta_learner_results")
        if meta_results:
            individual = {}
            for name in ["S-Learner", "T-Learner", "X-Learner",
                         "DR-Learner", "R-Learner"]:
                lr = meta_results.get(name, {})
                if lr:
                    individual[name] = {
                        "ate": lr.get("ate", 0),
                        "direction": lr.get("direction", "unknown"),
                    }
            ensemble = meta_results.get("ensemble", {})
            json_data["meta_learner"] = {
                "individual": individual,
                "ensemble": {
                    "consensus": ensemble.get("consensus", 0),
                    "oracle_ate": ensemble.get("oracle_ate", 0),
                    "method": ensemble.get("method", "mse_weighted"),
                },
            }

        # ──────────────────────────────────────────
        # 4-7. Refutation 4반증 요약 (Phase 2)
        # ──────────────────────────────────────────
        refutation = inputs.get("refutation_results")
        if refutation:
            json_data["refutation"] = {
                "placebo_test": {
                    "p_value": refutation.get("placebo_test", {}).get("p_value", 0),
                    "passed": refutation.get("placebo_test", {}).get("p_value", 0) > 0.05,
                },
                "bootstrap_ci": refutation.get("bootstrap_ci", {}),
                "leave_one_out": {
                    "any_sign_flip": refutation.get("leave_one_out", {}).get("any_sign_flip", False),
                },
                "subset_validation": {
                    "avg_stability": refutation.get("subset_validation", {}).get("avg_stability", 0),
                },
            }

        # ──────────────────────────────────────────
        # 5. 산점도용 샘플 데이터 (대시보드 성능 최적화)
        # ──────────────────────────────────────────
        max_pts = self.config.viz.max_scatter_points
        sample_df = df.sample(
            n=min(max_pts, len(df)),
            random_state=self.config.data.random_seed,
        )
        json_data["scatter_data"] = {
            col: sample_df[col].tolist()
            for col in feature_names + ["credit_limit", "is_default", "estimated_cate"]
            if col in sample_df.columns
        }

        # ──────────────────────────────────────────
        # 6. JSON 파일 저장
        # ──────────────────────────────────────────
        output_dir = self.config.paths.dashboard_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        # 시나리오별 파일 분리 (대시보드 dataLoader.ts와 동기화)
        scenario = inputs.get("scenario", "A")
        json_filename = "scenario_b.json" if scenario == "B" else "latest.json"
        json_path = output_dir / json_filename

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.logger.info("JSON 저장 완료: %s (%.1f KB)", json_path, json_path.stat().st_size / 1024)

        # CSV 백업 (paper/data/에 분석용)
        csv_path = self.config.paths.data_dir
        csv_path.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path / "synthetic_credit_data.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        self.logger.info("CSV 백업 저장: %s", csv_file)

        return {
            "json_path": str(json_path),
            "csv_path": str(csv_file),
            "json_data": json_data,
        }

    def _compute_histogram(
        self, values: np.ndarray, n_bins: int = 30
    ) -> Dict[str, list]:
        """히스토그램 데이터를 계산합니다.

        Args:
            values: CATE 예측값 배열.
            n_bins: 빈(bin) 개수.

        Returns:
            {"bin_edges": [...], "counts": [...]} 딕셔너리.
        """
        counts, bin_edges = np.histogram(values, bins=n_bins)
        return {
            "bin_edges": [float(x) for x in bin_edges],
            "counts": [int(x) for x in counts],
        }

    def _build_segments(
        self, df: pd.DataFrame, cate: np.ndarray
    ) -> List[Dict[str, Any]]:
        """유저 세그먼트별 CATE 통계를 집계합니다.

        보고서 시나리오 B의 세그먼트 분석 핵심:
        - 소득 분위별
        - 연령대별
        - 신용점수 구간별

        Args:
            df: 전체 데이터프레임.
            cate: CATE 예측값 배열.

        Returns:
            세그먼트별 통계 리스트.
        """
        segments: List[Dict[str, Any]] = []

        # 소득 분위(Quintile)별 분석
        if "income" in df.columns:
            df_temp = df.copy()
            df_temp["_cate"] = cate
            df_temp["income_quintile"] = pd.qcut(
                df_temp["income"], q=5, labels=["Q1(하)", "Q2", "Q3", "Q4", "Q5(상)"]
            )
            for label, group in df_temp.groupby("income_quintile", observed=True):
                segments.append({
                    "name": f"소득 {label}",
                    "dimension": "income",
                    "n": len(group),
                    "cate_mean": float(group["_cate"].mean()),
                    "cate_ci_lower": float(group["_cate"].quantile(0.025)),
                    "cate_ci_upper": float(group["_cate"].quantile(0.975)),
                })

        # 연령대별 분석
        if "age" in df.columns:
            df_temp = df.copy()
            df_temp["_cate"] = cate
            bins = [20, 30, 40, 50, 60, 70]
            labels = ["20대", "30대", "40대", "50대", "60대"]
            df_temp["age_group"] = pd.cut(
                df_temp["age"], bins=bins, labels=labels, right=False
            )
            for label, group in df_temp.groupby("age_group", observed=True):
                segments.append({
                    "name": f"연령 {label}",
                    "dimension": "age",
                    "n": len(group),
                    "cate_mean": float(group["_cate"].mean()),
                    "cate_ci_lower": float(group["_cate"].quantile(0.025)),
                    "cate_ci_upper": float(group["_cate"].quantile(0.975)),
                })

        # 신용점수 구간별 분석
        if "credit_score" in df.columns:
            df_temp = df.copy()
            df_temp["_cate"] = cate
            bins = [300, 500, 600, 700, 800, 900]
            labels = ["300-499", "500-599", "600-699", "700-799", "800-900"]
            df_temp["credit_group"] = pd.cut(
                df_temp["credit_score"], bins=bins, labels=labels, right=False
            )
            for label, group in df_temp.groupby("credit_group", observed=True):
                segments.append({
                    "name": f"신용 {label}",
                    "dimension": "credit_score",
                    "n": len(group),
                    "cate_mean": float(group["_cate"].mean()),
                    "cate_ci_lower": float(group["_cate"].quantile(0.025)),
                    "cate_ci_upper": float(group["_cate"].quantile(0.975)),
                })

        return segments

    def _extract_dag_nodes(
        self, edges: List[tuple], treatment: str = "credit_limit",
        outcome: str = "is_default",
    ) -> List[Dict[str, str]]:
        """DAG 엣지에서 고유 노드 목록을 추출합니다.

        Args:
            edges: (source, target) 튜플 리스트.
            treatment: 처치 변수 이름.
            outcome: 결과 변수 이름.

        Returns:
            노드 딕셔너리 리스트.
        """
        node_set: set = set()
        for src, tgt in edges:
            node_set.add(src)
            node_set.add(tgt)

        role_map = {treatment: "treatment", outcome: "outcome"}
        return [
            {
                "id": node,
                "label": node,
                "role": role_map.get(node, "confounder"),
            }
            for node in sorted(node_set)
        ]

    @staticmethod
    def _generate_ate_description(
        ate: float, treatment: str, outcome: str, scenario: str,
    ) -> str:
        """ATE를 사람이 읽을 수 있는 한 줄 설명으로 변환합니다."""
        direction = "증가" if ate > 0 else "감소"
        abs_pct = abs(ate * 100)
        if scenario == "A":
            return (
                f"신용 한도(credit_limit) 1단위 증가 시 "
                f"연체율(is_default)이 {abs_pct:.2f}%p {direction}"
            )
        elif scenario == "B":
            return (
                f"쿠폰 지급(coupon_sent) 시 "
                f"가입 확률(is_joined)이 {abs_pct:.2f}%p {direction}"
            )
        return f"{treatment} → {outcome}: ATE = {ate:.6f} ({direction})"

    @staticmethod
    def _build_sensitivity(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """sensitivity_results를 TypeScript 타입에 맞춰 정규화합니다.

        최상위 status = 모든 하위 테스트가 Pass면 Pass, 아니면 Fail.
        """
        raw = inputs.get("sensitivity_results", {})
        placebo = raw.get("placebo_test", {
            "status": "Not Run", "p_value": 0.0, "mean_effect": 0.0
        })
        rcc = raw.get("random_common_cause", {
            "status": "Not Run", "stability": 0.0, "mean_effect": 0.0
        })
        e_value = raw.get("e_value", {
            "status": "Not Run", "point": 0.0, "ci_bound": 0.0, "interpretation": ""
        })
        overlap = raw.get("overlap", {
            "status": "Not Run", "overlap_score": 0.0, "interpretation": ""
        })
        gates = raw.get("gates", {
            "status": "Not Run", "n_groups": 0, "groups": [],
            "f_statistic": 0.0, "heterogeneity": ""
        })

        # Pass 여부 계산 (Info 상태인 GATES는 Pass/Fail 판정에서 제외)
        test_statuses = [
            placebo.get("status"), rcc.get("status"),
            e_value.get("status"), overlap.get("status"),
        ]
        graded = [s for s in test_statuses if s not in ("Not Run", "Info")]
        all_pass = all(s == "Pass" for s in graded) if graded else False

        return {
            "status": "Pass" if all_pass else "Fail",
            "placebo_test": placebo,
            "random_common_cause": rcc,
            "e_value": e_value,
            "overlap": overlap,
            "gates": gates,
        }
