# -*- coding: utf-8 -*-
"""FairnessAuditCell — 인과 공정성 감사 모듈.

CATE 이질성을 기반으로 민감 속성(성별, 인종, 연령대 등)별
처치효과 공정성을 평가합니다.

평가 지표:
  1. Causal Parity: 서브그룹 간 ATE 격차
  2. Equalized CATE: CATE 분포의 서브그룹별 차이
  3. Counterfactual Fairness Index: 반사실적 공정성 정량화
  4. Disparate Impact Ratio: CATE 양수 비율 격차

학술 참조:
  - Kusner et al. (2017). "Counterfactual Fairness."
  - Makhlouf et al. (2021). "Machine Learning Fairness Notions:
    Bridging the Gap with Causal Reasoning."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

@dataclass
class FairnessConfig:
    """공정성 감사 설정."""

    sensitive_attrs: List[str] = field(default_factory=list)
    parity_threshold: float = 0.1       # ATE 격차 허용치
    dir_threshold: float = 0.8          # Disparate Impact Ratio 하한
    significance_level: float = 0.05    # 유의수준
    generate_report: bool = True        # 텍스트 보고서 생성


# ──────────────────────────────────────────────
# 공정성 지표 계산
# ──────────────────────────────────────────────

@dataclass
class SubgroupMetrics:
    """서브그룹별 공정성 지표."""
    group_name: str
    group_value: Any
    n_samples: int
    mean_cate: float
    std_cate: float
    median_cate: float
    positive_ratio: float  # CATE > 0 인 비율


@dataclass
class FairnessResult:
    """공정성 감사 결과."""
    attribute: str
    subgroups: List[SubgroupMetrics]
    causal_parity_gap: float        # max(ATE) - min(ATE) across groups
    disparate_impact_ratio: float   # min(positive_ratio) / max(positive_ratio)
    equalized_cate_score: float     # 1 - normalized std of group means
    counterfactual_fairness_idx: float  # 기대 반사실 공정성 지표
    is_fair: bool                   # 종합 판정
    violations: List[str]           # 위반 항목 목록


def compute_subgroup_metrics(
    cate: np.ndarray,
    groups: np.ndarray,
) -> List[SubgroupMetrics]:
    """서브그룹별 CATE 분포 통계."""
    unique_groups = np.unique(groups)
    metrics = []

    for g in unique_groups:
        mask = groups == g
        cate_g = cate[mask]
        metrics.append(SubgroupMetrics(
            group_name=str(g),
            group_value=g,
            n_samples=int(np.sum(mask)),
            mean_cate=float(np.mean(cate_g)),
            std_cate=float(np.std(cate_g)),
            median_cate=float(np.median(cate_g)),
            positive_ratio=float(np.mean(cate_g > 0)),
        ))

    return metrics


def compute_causal_parity_gap(subgroups: List[SubgroupMetrics]) -> float:
    """Causal Parity: 서브그룹 간 최대 ATE 격차."""
    means = [s.mean_cate for s in subgroups]
    return float(max(means) - min(means)) if means else 0.0


def compute_disparate_impact_ratio(subgroups: List[SubgroupMetrics]) -> float:
    """Disparate Impact Ratio: min(positive_ratio) / max(positive_ratio)."""
    ratios = [s.positive_ratio for s in subgroups]
    max_r = max(ratios) if ratios else 1.0
    min_r = min(ratios) if ratios else 0.0
    return float(min_r / max(max_r, 1e-10))


def compute_equalized_cate_score(subgroups: List[SubgroupMetrics]) -> float:
    """Equalized CATE: 1 - 정규화된 그룹 평균 표준편차.

    1.0 = 완벽히 평등, 0.0 = 극단적 불평등.
    """
    means = np.array([s.mean_cate for s in subgroups])
    if len(means) < 2:
        return 1.0
    overall_std = np.std(means)
    overall_mean = np.mean(np.abs(means)) + 1e-10
    # 정규화: 격차를 전체 효과 대비 비율로 표현
    normalized = overall_std / overall_mean
    return float(max(0.0, 1.0 - normalized))


def compute_counterfactual_fairness_index(
    subgroups: List[SubgroupMetrics],
) -> float:
    """반사실 공정성 지표 (0~1).

    개념: "민감 속성이 바뀌었을 때 CATE가 동일한가?"
    실무적으로는 서브그룹 간 CATE 분포 겹침(overlap) 정도로 근사.

    Cohen's d 기반: 분포 겹침이 클수록 공정.
    """
    if len(subgroups) < 2:
        return 1.0

    # 모든 쌍에 대해 Cohen's d 계산
    cohens_ds = []
    for i in range(len(subgroups)):
        for j in range(i + 1, len(subgroups)):
            pooled_std = np.sqrt(
                (subgroups[i].std_cate ** 2 + subgroups[j].std_cate ** 2) / 2
            )
            if pooled_std > 1e-10:
                d = abs(subgroups[i].mean_cate - subgroups[j].mean_cate) / pooled_std
            else:
                d = 0.0
            cohens_ds.append(d)

    avg_d = np.mean(cohens_ds) if cohens_ds else 0.0
    # d가 작을수록 공정: 지표 = 1 / (1 + avg_d)
    return float(1.0 / (1.0 + avg_d))


# ──────────────────────────────────────────────
# 보고서 생성
# ──────────────────────────────────────────────

def generate_fairness_report(results: List[FairnessResult]) -> str:
    """Markdown 공정성 감사 보고서."""
    lines = [
        "# 인과 공정성 감사 보고서",
        "",
        f"감사 대상 민감 속성: {len(results)}개",
        "",
    ]

    for r in results:
        status = "✅ 공정" if r.is_fair else "⚠️ 위반 감지"
        lines.append(f"## 민감 속성: `{r.attribute}` — {status}")
        lines.append("")

        # 서브그룹 표
        lines.append("| 그룹 | N | 평균 CATE | 중위 CATE | CATE>0 비율 |")
        lines.append("|------|--:|--------:|--------:|---------:|")
        for sg in r.subgroups:
            lines.append(
                f"| {sg.group_name} | {sg.n_samples} | "
                f"{sg.mean_cate:.4f} | {sg.median_cate:.4f} | "
                f"{sg.positive_ratio:.1%} |"
            )
        lines.append("")

        lines.append("| 지표 | 값 | 기준 | 판정 |")
        lines.append("|------|--:|:---:|:---:|")
        lines.append(
            f"| Causal Parity 격차 | {r.causal_parity_gap:.4f} | "
            f"< 허용치 | {'✅' if r.causal_parity_gap < 0.1 else '❌'} |"
        )
        lines.append(
            f"| Disparate Impact 비율 | {r.disparate_impact_ratio:.3f} | "
            f"≥ 0.8 | {'✅' if r.disparate_impact_ratio >= 0.8 else '❌'} |"
        )
        lines.append(
            f"| Equalized CATE 점수 | {r.equalized_cate_score:.3f} | "
            f"≥ 0.8 | {'✅' if r.equalized_cate_score >= 0.8 else '❌'} |"
        )
        lines.append(
            f"| 반사실 공정성 지표 | {r.counterfactual_fairness_idx:.3f} | "
            f"≥ 0.7 | {'✅' if r.counterfactual_fairness_idx >= 0.7 else '❌'} |"
        )

        if r.violations:
            lines.append("")
            lines.append("**위반 사항:**")
            for v in r.violations:
                lines.append(f"- ⚠️ {v}")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# FairnessAuditCell 메인
# ──────────────────────────────────────────────

class FairnessAuditCell:
    """인과 공정성 감사 셀.

    CATE 이질성을 기반으로 민감 속성별 공정성을 감사합니다.

    파이프라인 인터페이스:
        cell = FairnessAuditCell(config)
        result = cell.execute(inputs)

    직접 호출:
        result = cell.audit(cate, df, sensitive_attrs)
    """

    name = "FairnessAudit"

    def __init__(
        self,
        config=None,
        fairness_config: Optional[FairnessConfig] = None,
    ):
        self.config = config
        self.fairness_config = fairness_config or FairnessConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def audit(
        self,
        cate: np.ndarray,
        df: pd.DataFrame,
        sensitive_attrs: List[str],
    ) -> List[FairnessResult]:
        """공정성 감사를 수행합니다.

        Args:
            cate: (n,) 개인별 CATE 추정값.
            df: 원본 데이터프레임 (민감 속성 포함).
            sensitive_attrs: 감사 대상 민감 속성 목록.

        Returns:
            List[FairnessResult]: 속성별 감사 결과.
        """
        cfg = self.fairness_config
        results = []

        for attr in sensitive_attrs:
            if attr not in df.columns:
                self.logger.warning("민감 속성 '%s'이 데이터에 없습니다.", attr)
                continue

            groups = df[attr].values

            # 서브그룹 지표
            subgroups = compute_subgroup_metrics(cate, groups)

            # 전체 지표
            parity_gap = compute_causal_parity_gap(subgroups)
            dir_ratio = compute_disparate_impact_ratio(subgroups)
            eq_score = compute_equalized_cate_score(subgroups)
            cf_idx = compute_counterfactual_fairness_index(subgroups)

            # 위반 판정
            violations = []
            if parity_gap > cfg.parity_threshold:
                violations.append(
                    f"Causal Parity 격차({parity_gap:.4f})가 "
                    f"허용치({cfg.parity_threshold})를 초과"
                )
            if dir_ratio < cfg.dir_threshold:
                violations.append(
                    f"Disparate Impact 비율({dir_ratio:.3f})이 "
                    f"하한({cfg.dir_threshold})을 미달"
                )

            is_fair = len(violations) == 0

            self.logger.info(
                "공정성 감사 [%s]: parity_gap=%.4f, DIR=%.3f, eq=%.3f, cf=%.3f → %s",
                attr, parity_gap, dir_ratio, eq_score, cf_idx,
                "공정" if is_fair else f"위반 {len(violations)}건",
            )

            results.append(FairnessResult(
                attribute=attr,
                subgroups=subgroups,
                causal_parity_gap=parity_gap,
                disparate_impact_ratio=dir_ratio,
                equalized_cate_score=eq_score,
                counterfactual_fairness_idx=cf_idx,
                is_fair=is_fair,
                violations=violations,
            ))

        return results

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """파이프라인 셀 인터페이스."""
        df = inputs.get("dataframe")
        cate = inputs.get("cate")

        # CATE가 없으면 가설 CATE 시도
        if cate is None:
            meta = inputs.get("meta_learner_results", {})
            if isinstance(meta, dict):
                cate = meta.get("cate")
            if cate is None:
                cate_array = inputs.get("cate_values")
                if cate_array is not None:
                    cate = np.array(cate_array)

        if df is None or cate is None:
            self.logger.warning("데이터프레임 또는 CATE가 없습니다.")
            return {**inputs, "fairness_audit": None}

        cate = np.asarray(cate).ravel()

        # 민감 속성 결정
        sensitive_attrs = self.fairness_config.sensitive_attrs
        if not sensitive_attrs:
            # 자동 탐지: 카디널리티가 낮은(2~10) 컬럼을 후보로
            treatment_col = inputs.get("treatment_col", "")
            outcome_col = inputs.get("outcome_col", "")
            feature_names = inputs.get("feature_names", [])
            exclude = {treatment_col, outcome_col}

            candidates = []
            for col in df.columns:
                if col in exclude:
                    continue
                nunique = df[col].nunique()
                if 2 <= nunique <= 10:
                    candidates.append(col)

            sensitive_attrs = candidates[:3]  # 최대 3개

            if not sensitive_attrs:
                self.logger.info("감사 대상 민감 속성을 찾지 못했습니다.")
                return {**inputs, "fairness_audit": {"skipped": True, "reason": "no_sensitive_attrs"}}

        results = self.audit(cate, df, sensitive_attrs)

        # 결과 직렬화
        audit_output = {
            "audited_attributes": [r.attribute for r in results],
            "results": [
                {
                    "attribute": r.attribute,
                    "causal_parity_gap": r.causal_parity_gap,
                    "disparate_impact_ratio": r.disparate_impact_ratio,
                    "equalized_cate_score": r.equalized_cate_score,
                    "counterfactual_fairness_idx": r.counterfactual_fairness_idx,
                    "is_fair": r.is_fair,
                    "violations": r.violations,
                    "subgroups": [
                        {
                            "group": sg.group_name,
                            "n": sg.n_samples,
                            "mean_cate": sg.mean_cate,
                            "positive_ratio": sg.positive_ratio,
                        }
                        for sg in r.subgroups
                    ],
                }
                for r in results
            ],
            "overall_fair": all(r.is_fair for r in results),
        }

        # 보고서 생성
        if self.fairness_config.generate_report and results:
            audit_output["report_md"] = generate_fairness_report(results)

        return {**inputs, "fairness_audit": audit_output}
