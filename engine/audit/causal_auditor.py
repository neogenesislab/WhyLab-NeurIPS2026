# -*- coding: utf-8 -*-
"""Causal Auditor — 자동 인과 감사 오케스트레이터.

DecisionOutcomePair를 받아 WhyLab 인과추론 파이프라인을 실행하고
AuditResult를 생성합니다.

사용법:
    from engine.audit.causal_auditor import CausalAuditor

    auditor = CausalAuditor()
    result = auditor.audit(decision_outcome_pair)
    print(result.verdict)        # CAUSAL / NOT_CAUSAL / UNCERTAIN
    print(result.recommendation) # 마크다운 감사 보고서
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

from engine.audit.schemas import (
    AuditResult,
    AuditVerdict,
    DecisionOutcomePair,
)

logger = logging.getLogger("whylab.audit.auditor")

# 최소 요건 상수
MIN_PRE_OBSERVATIONS = 7
MIN_POST_OBSERVATIONS = 3
SIGNIFICANCE_THRESHOLD = 0.05


class CausalAuditor:
    """에이전트 결정에 대한 인과 감사를 수행합니다.

    하이브리드 추론 아키텍처 (리서치 기반):
        - 데이터 풍부 → CausalImpact (BSTS)
        - 데이터 희소 → GSC (Generalized Synthetic Control)
        - Phase 1 기본 → lightweight_t_test (scipy 미의존)

    Phase 2에서 DML/GSC 통합 시 method_router가 자동 스위칭합니다.
    """

    # 지원 메서드 (Phase별 확장)
    SUPPORTED_METHODS = [
        "lightweight_t_test",    # Phase 1: 기본 (현재)
        "causal_impact",         # Phase 2: 데이터 풍부 시
        "gsc",                   # Phase 2: 데이터 희소 시
        "dml",                   # Phase 2: Multi-treatment
    ]

    def __init__(
        self,
        significance_level: float = SIGNIFICANCE_THRESHOLD,
        min_pre: int = MIN_PRE_OBSERVATIONS,
        min_post: int = MIN_POST_OBSERVATIONS,
        preferred_method: str = "auto",
    ) -> None:
        self.significance_level = significance_level
        self.min_pre = min_pre
        self.min_post = min_post
        self.preferred_method = preferred_method

    def audit(self, pair: DecisionOutcomePair) -> AuditResult:
        """DecisionOutcomePair에 대한 인과 감사를 실행합니다.

        Args:
            pair: 매칭된 결정-결과 쌍

        Returns:
            AuditResult (판결, 확신도, ATE 등)
        """
        decision = pair.decision

        # 데이터 충분성 검사
        if not self._check_data_sufficiency(pair):
            return AuditResult(
                decision_id=decision.decision_id,
                verdict=AuditVerdict.INSUFFICIENT_DATA,
                confidence=0.0,
                method="data_check",
                recommendation=self._render_insufficient_report(pair),
            )

        pre_values = pair.pre_values
        post_values = pair.post_values

        # ── 경량 인과 분석 (Phase 1) ──
        # Phase 2에서 WhyLab CausalImpact/DiD로 교체 예정
        analysis = self._lightweight_causal_analysis(pre_values, post_values)

        # 판결 결정
        verdict = self._determine_verdict(analysis)
        confidence = analysis["confidence"]
        ate = analysis["ate"]

        # 감사 보고서 렌더링
        recommendation = self._render_audit_report(pair, analysis, verdict)

        result = AuditResult(
            decision_id=decision.decision_id,
            verdict=verdict,
            confidence=confidence,
            ate=ate,
            ate_ci=analysis["ate_ci"],
            p_value=analysis.get("p_value"),
            method=analysis["method"],
            refutation_passed=analysis.get("refutation_passed", False),
            recommendation=recommendation,
            pipeline_results=analysis,
        )

        logger.info(
            "📋 감사 완료: [%s] %s → %s (ATE=%.4f, conf=%.1f%%)",
            decision.decision_id[:8],
            decision.agent_name,
            verdict.value,
            ate,
            confidence * 100,
        )

        return result

    def audit_batch(self, pairs: List[DecisionOutcomePair]) -> List[AuditResult]:
        """여러 쌍을 일괄 감사합니다."""
        return [self.audit(pair) for pair in pairs]

    # ── 경량 인과 분석 (Phase 1) ──

    def _lightweight_causal_analysis(
        self,
        pre: List[float],
        post: List[float],
    ) -> Dict[str, Any]:
        """경량 인과 분석 – CausalImpact 구현 전 대체.

        방법:
        1. Pre/Post 평균 차이 (ATE 추정)
        2. Welch's t-test (유의성)
        3. Effect size (Cohen's d)
        4. 단순 Placebo test (pre 기간 분할)
        """
        pre_mean = statistics.mean(pre)
        post_mean = statistics.mean(post)
        ate = post_mean - pre_mean

        pre_std = statistics.stdev(pre) if len(pre) > 1 else 1e-10
        post_std = statistics.stdev(post) if len(post) > 1 else 1e-10

        # Welch's t-test (scipy 없이 직접 계산)
        n_pre, n_post = len(pre), len(post)
        se = (pre_std**2 / n_pre + post_std**2 / n_post) ** 0.5
        t_stat = ate / se if se > 1e-10 else 0.0

        # 자유도 (Welch-Satterthwaite)
        num = (pre_std**2 / n_pre + post_std**2 / n_post) ** 2
        denom = (
            (pre_std**2 / n_pre) ** 2 / max(n_pre - 1, 1)
            + (post_std**2 / n_post) ** 2 / max(n_post - 1, 1)
        )
        df = num / denom if denom > 0 else 1.0

        # p-value 근사 (정규 분포 근사, scipy 미의존)
        import math
        z = abs(t_stat)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

        # Cohen's d (효과 크기)
        pooled_std = ((pre_std**2 + post_std**2) / 2) ** 0.5
        cohens_d = ate / pooled_std if pooled_std > 1e-10 else 0.0

        # ATE 신뢰구간 (95%)
        margin = 1.96 * se
        ate_ci = [ate - margin, ate + margin]

        # 단순 Placebo test (pre 기간을 반으로 나눠 효과 확인)
        placebo_passed = True
        if len(pre) >= 6:
            mid = len(pre) // 2
            placebo_ate = statistics.mean(pre[mid:]) - statistics.mean(pre[:mid])
            placebo_passed = abs(placebo_ate) < abs(ate) * 0.5

        # Confidence score (0~1)
        confidence = 0.0
        if p_value < self.significance_level:
            confidence += 0.4
        if abs(cohens_d) > 0.3:
            confidence += 0.2
        if placebo_passed:
            confidence += 0.2
        if ate_ci[0] > 0 or ate_ci[1] < 0:  # CI가 0을 포함하지 않음
            confidence += 0.2

        return {
            "method": "lightweight_t_test",
            "ate": round(ate, 4),
            "ate_ci": [round(x, 4) for x in ate_ci],
            "p_value": round(p_value, 6),
            "t_statistic": round(t_stat, 4),
            "df": round(df, 1),
            "cohens_d": round(cohens_d, 4),
            "pre_mean": round(pre_mean, 4),
            "post_mean": round(post_mean, 4),
            "pre_std": round(pre_std, 4),
            "post_std": round(post_std, 4),
            "n_pre": n_pre,
            "n_post": n_post,
            "placebo_passed": placebo_passed,
            "confidence": round(min(confidence, 1.0), 2),
        }

    def _determine_verdict(self, analysis: Dict[str, Any]) -> AuditVerdict:
        """분석 결과로부터 판결을 결정합니다."""
        p = analysis["p_value"]
        d = abs(analysis["cohens_d"])
        conf = analysis["confidence"]
        placebo = analysis["placebo_passed"]

        if p < self.significance_level and d > 0.3 and placebo and conf >= 0.6:
            return AuditVerdict.CAUSAL
        elif p > 0.2 or (not placebo) or (d < 0.1):
            return AuditVerdict.NOT_CAUSAL
        else:
            return AuditVerdict.UNCERTAIN

    # ── 데이터 검사 ──

    def _check_data_sufficiency(self, pair: DecisionOutcomePair) -> bool:
        """감사에 필요한 최소 데이터가 있는지 확인."""
        if len(pair.pre_outcomes) < self.min_pre:
            logger.warning(
                "⚠️ 사전 관측 부족: %d < %d (decision: %s)",
                len(pair.pre_outcomes), self.min_pre,
                pair.decision.decision_id[:8],
            )
            return False
        if len(pair.post_outcomes) < self.min_post:
            logger.warning(
                "⚠️ 사후 관측 부족: %d < %d (decision: %s)",
                len(pair.post_outcomes), self.min_post,
                pair.decision.decision_id[:8],
            )
            return False
        return True

    # ── 보고서 렌더링 ──

    def _render_audit_report(
        self,
        pair: DecisionOutcomePair,
        analysis: Dict[str, Any],
        verdict: AuditVerdict,
    ) -> str:
        """인과 감사 보고서를 마크다운으로 렌더링합니다."""
        d = pair.decision
        icon = {"CAUSAL": "🚀", "NOT_CAUSAL": "🛑", "UNCERTAIN": "⚖️"}.get(
            verdict.value, "📋"
        )

        lines = [
            f"## {icon} Causal Audit Report",
            "",
            f"**Agent:** `{d.agent_name}` ({d.agent_type.value})",
            f"**Decision:** {d.treatment}",
            f"**Target:** {d.target_sbu} / {d.target_metric.value}",
            f"**Verdict:** `{verdict.value}` | **Confidence:** {analysis['confidence']:.0%}",
            "",
            "### Statistical Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| ATE | {analysis['ate']:+.4f} |",
            f"| 95% CI | [{analysis['ate_ci'][0]:.4f}, {analysis['ate_ci'][1]:.4f}] |",
            f"| p-value | {analysis['p_value']:.6f} |",
            f"| Cohen's d | {analysis['cohens_d']:.4f} |",
            f"| Pre Mean | {analysis['pre_mean']:.4f} (n={analysis['n_pre']}) |",
            f"| Post Mean | {analysis['post_mean']:.4f} (n={analysis['n_post']}) |",
            f"| Placebo Test | {'✅ Passed' if analysis['placebo_passed'] else '❌ Failed'} |",
            "",
        ]

        # 판결별 권고
        if verdict == AuditVerdict.CAUSAL:
            lines += [
                "### 📈 Recommendation",
                "",
                f"- 에이전트 결정 **효과 확인**. 전략 유지 권장.",
                f"- ATE {analysis['ate']:+.4f}: {d.target_metric.value} 지표에 유의미한 변화.",
                f"- Phase 2에서 CausalImpact로 정밀 재검증 예정.",
            ]
        elif verdict == AuditVerdict.NOT_CAUSAL:
            lines += [
                "### ⚠️ Recommendation",
                "",
                f"- 에이전트 결정의 **효과 미확인**. 전략 재검토 필요.",
                f"- 에이전트 전략 메모리에 '비효과적' 태그 추가 권장.",
            ]
        else:
            lines += [
                "### 🔍 Recommendation",
                "",
                f"- 추가 데이터 수집 후 재감사 필요.",
                f"- 관측 기간을 {d.observation_window_days * 2}일로 연장 권장.",
            ]

        lines.append("")
        return "\n".join(lines)

    def _render_insufficient_report(self, pair: DecisionOutcomePair) -> str:
        """데이터 부족 시 보고서."""
        d = pair.decision
        return (
            f"## ⚠️ Insufficient Data\n\n"
            f"**Agent:** `{d.agent_name}`\n"
            f"**Decision:** {d.treatment}\n\n"
            f"감사에 필요한 최소 데이터가 부족합니다.\n"
            f"- 사전 관측: {len(pair.pre_outcomes)}건 (최소 {self.min_pre}건)\n"
            f"- 사후 관측: {len(pair.post_outcomes)}건 (최소 {self.min_post}건)\n"
        )
