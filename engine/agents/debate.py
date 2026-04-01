# -*- coding: utf-8 -*-
"""Multi-Agent Debate — 인과 판결 자동화.

3개 에이전트(Advocate, Critic, Judge)가 인과추론 결과를 양측에서
공격/방어하고, 가중 스코어링으로 자동 판결합니다.

이것이 WhyLab의 핵심 차별점입니다:
  기존: "인과추론 → 숫자 → 인간 해석"
  WhyLab: "인과추론 → AI 양측 공방 → 자동 판결 + 근거 보고서"

참고: Liang et al. (2023) "Encouraging Divergent Thinking in LLMs
through Multi-Agent Debate"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class Evidence:
    """구조화된 증거 단위.

    Attributes:
        claim: 주장 내용 (예: "5개 메타러너 중 4개 동일 방향")
        evidence_type: 증거 유형 ("statistical" | "robustness" | "domain")
        strength: 증거 강도 (0.0 ~ 1.0)
        source: 증거 출처 (예: "meta_learner_consensus")
        data: 수치 근거 데이터
        business_impact: 비즈니스 영향 분석 (예: "Revenue +5%", "Risk High")
    """
    claim: str
    evidence_type: str
    strength: float
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    business_impact: Optional[str] = None


@dataclass
class Verdict:
    """최종 판결 결과.

    Attributes:
        verdict: "CAUSAL" | "NOT_CAUSAL" | "UNCERTAIN"
        confidence: 확신도 (0.0 ~ 1.0)
        pro_score: 옹호측 총점
        con_score: 비판측 총점
        pro_evidence: 옹호 증거 리스트
        con_evidence: 비판 증거 리스트
        recommendation: 추가 분석 제안
        rounds: 실행된 Debate 라운드 수
    """
    verdict: str
    confidence: float
    pro_score: float
    con_score: float
    pro_evidence: List[Evidence]
    con_evidence: List[Evidence]
    recommendation: str
    rounds: int = 1


# ──────────────────────────────────────────────
# Advocate (옹호 에이전트)
# ──────────────────────────────────────────────

class AdvocateAgent:
    """인과 효과의 존재를 옹호하는 에이전트.

    10가지 증거를 수집하여 인과 관계의 존재를 지지합니다.
    각 증거는 파이프라인 결과에서 추출되며, 누락 시 안전하게 건너뜁니다.
    """

    def gather_evidence(self, results: Dict[str, Any]) -> List[Evidence]:
        """파이프라인 결과에서 긍정 증거를 수집합니다."""
        evidence = []

        # 1. 메타러너 합의율
        evidence.append(self._meta_learner_consensus(results))

        # 2. Bootstrap p-value
        evidence.append(self._bootstrap_significance(results))

        # 3. ATE 유의성 (0 미포함)
        evidence.append(self._ate_significance(results))

        # 4. E-value 크기
        evidence.append(self._e_value_strength(results))

        # 5. Conformal CI zero-exclusion
        evidence.append(self._conformal_zero_exclusion(results))

        # 6. LOO 안정성
        evidence.append(self._loo_stability(results))

        # 7. Subset 안정성
        evidence.append(self._subset_stability(results))

        # 8. Overlap 양호
        evidence.append(self._overlap_quality(results))

        # 9. GATES 이질성
        evidence.append(self._gates_heterogeneity(results))

        # 10. SHAP-CATE 정합성
        evidence.append(self._shap_cate_coherence(results))

        # None 필터링 및 비즈니스 임팩트 주입
        valid = []
        for e in evidence:
            if e is not None:
                e.business_impact = self._generate_impact(e)
                valid.append(e)

        logger.info("📗 Advocate (Growth Hacker): %d개 증거 수집 (강도 합계: %.2f)",
                     len(valid), sum(e.strength for e in valid))
        return valid

    def _generate_impact(self, e: Evidence) -> str:
        """증거를 비즈니스 기회로 번역 (Growth Hacker Persona)."""
        if e.source == "meta_learner_consensus":
            return "모델 간 합의로 예측 신뢰도 확보 → 과감한 마케팅 집행 가능"
        if e.source == "bootstrap_ci" or e.source == "ate_significance":
            return "통계적 유의성 확보 → KPI 개선 가능성 높음"
        if e.source == "gates_heterogeneity":
            return "타겟팅 효율화 기회 발견 (상위 20% 유저 집중 공략)"
        return "안정적인 성과 기대"

    def _meta_learner_consensus(self, r: Dict) -> Optional[Evidence]:
        """메타러너 합의율."""
        meta = r.get("meta_learner_results", {})
        ensemble = meta.get("ensemble", {})
        consensus = ensemble.get("consensus", 0)
        if consensus == 0:
            return None
        return Evidence(
            claim=f"메타러너 {consensus*100:.0f}% 동일 방향",
            evidence_type="statistical",
            strength=consensus,
            source="meta_learner_consensus",
            data={"consensus": consensus},
        )

    def _bootstrap_significance(self, r: Dict) -> Optional[Evidence]:
        """Bootstrap 반증 통과."""
        refutation = r.get("refutation_results", {})
        placebo = refutation.get("placebo_test", {})
        p_value = placebo.get("p_value")
        if p_value is None:
            return None
        strength = max(0, 1 - p_value)
        return Evidence(
            claim=f"Placebo p={p_value:.4f} (원래 효과 유의)",
            evidence_type="robustness",
            strength=strength,
            source="placebo_p_value",
            data={"p_value": p_value},
        )

    def _ate_significance(self, r: Dict) -> Optional[Evidence]:
        """ATE Bootstrap CI가 0을 포함하지 않는지."""
        refutation = r.get("refutation_results", {})
        bootstrap = refutation.get("bootstrap_ci", {})
        ci_lower = bootstrap.get("ci_lower")
        ci_upper = bootstrap.get("ci_upper")
        if ci_lower is None or ci_upper is None:
            return None
        excludes_zero = (ci_lower > 0) or (ci_upper < 0)
        return Evidence(
            claim=f"Bootstrap CI [{ci_lower:.4f}, {ci_upper:.4f}] {'0 미포함 ✓' if excludes_zero else '0 포함'}",
            evidence_type="statistical",
            strength=1.0 if excludes_zero else 0.2,
            source="bootstrap_ci",
            data={"ci_lower": ci_lower, "ci_upper": ci_upper},
        )

    def _e_value_strength(self, r: Dict) -> Optional[Evidence]:
        """E-value 크기."""
        sensitivity = r.get("sensitivity_results", {})
        e_val_raw = sensitivity.get("e_value")
        if e_val_raw is None:
            return None
        # e_value는 dict (point, ci_bound) 또는 float
        e_value = e_val_raw.get("point", 0) if isinstance(e_val_raw, dict) else float(e_val_raw)
        if e_value == 0:
            return None
        strength = min(e_value / 3.0, 1.0)
        return Evidence(
            claim=f"E-value={e_value:.2f} ({'강건' if e_value >= 2 else '취약'})",
            evidence_type="robustness",
            strength=strength,
            source="e_value",
            data={"e_value": e_value},
        )

    def _conformal_zero_exclusion(self, r: Dict) -> Optional[Evidence]:
        """Conformal CI가 0을 포함하지 않는지."""
        conformal = r.get("conformal_results", {})
        ci_lower = conformal.get("ci_lower_mean")
        ci_upper = conformal.get("ci_upper_mean")
        if ci_lower is None or ci_upper is None:
            return None
        excludes = (ci_lower > 0) or (ci_upper < 0)
        return Evidence(
            claim=f"Conformal CI {'0 미포함 → 유의' if excludes else '0 포함 → 불확실'}",
            evidence_type="statistical",
            strength=1.0 if excludes else 0.1,
            source="conformal_ci",
            data={"ci_lower_mean": ci_lower, "ci_upper_mean": ci_upper},
        )

    def _loo_stability(self, r: Dict) -> Optional[Evidence]:
        """Leave-One-Out 안정성."""
        refutation = r.get("refutation_results", {})
        loo = refutation.get("leave_one_out", {})
        sign_flip = loo.get("any_sign_flip")
        if sign_flip is None:
            return None
        return Evidence(
            claim=f"LOO 교란변수 제거 후 {'부호 유지 ✓' if not sign_flip else '부호 반전 ✗'}",
            evidence_type="robustness",
            strength=1.0 if not sign_flip else 0.0,
            source="loo_confounder",
            data={"any_sign_flip": sign_flip},
        )

    def _subset_stability(self, r: Dict) -> Optional[Evidence]:
        """Subset 안정성."""
        refutation = r.get("refutation_results", {})
        subset = refutation.get("subset_validation", {})
        stability = subset.get("avg_stability")
        if stability is None:
            return None
        return Evidence(
            claim=f"서브셋 안정성: {stability:.2f}",
            evidence_type="robustness",
            strength=stability,
            source="subset_stability",
            data={"avg_stability": stability},
        )

    def _overlap_quality(self, r: Dict) -> Optional[Evidence]:
        """Overlap 양호."""
        sensitivity = r.get("sensitivity_results", {})
        overlap_raw = sensitivity.get("overlap")
        if overlap_raw is None:
            return None
        # overlap은 dict (overlap_score, status) 또는 float
        overlap = overlap_raw.get("overlap_score", 0) if isinstance(overlap_raw, dict) else float(overlap_raw)
        strength = min(overlap / 0.9, 1.0) if overlap > 0.5 else 0.0
        return Evidence(
            claim=f"Overlap={overlap:.2f} ({'양호' if overlap > 0.7 else '주의'})",
            evidence_type="statistical",
            strength=strength,
            source="overlap",
            data={"overlap": overlap},
        )

    def _gates_heterogeneity(self, r: Dict) -> Optional[Evidence]:
        """GATES F-statistic 유의."""
        sensitivity = r.get("sensitivity_results", {})
        # gates는 키가 'gates' 또는 'gates_results'일 수 있음
        gates = sensitivity.get("gates_results") or sensitivity.get("gates", {})
        if isinstance(gates, dict):
            f_sig = gates.get("f_stat_significant")
            # f_statistic이 있으면 유의성 판단
            if f_sig is None and "f_statistic" in gates:
                f_sig = gates["f_statistic"] > 3.84  # F(1,n-2) 5% 임계값
        else:
            f_sig = None
        if f_sig is None:
            return None
        return Evidence(
            claim=f"GATES 이질성 {'유의 ✓' if f_sig else '비유의'}",
            evidence_type="statistical",
            strength=1.0 if f_sig else 0.3,
            source="gates_heterogeneity",
            data={"f_stat_significant": f_sig},
        )

    def _shap_cate_coherence(self, r: Dict) -> Optional[Evidence]:
        """SHAP 중요도와 CATE 변수 정합성."""
        importance = r.get("feature_importance", {})
        feature_names = r.get("feature_names", [])
        if not importance or not feature_names:
            return None
        # 상위 3개 SHAP 변수가 실제 교란변수에 포함되는지
        top_shap = sorted(importance, key=importance.get, reverse=True)[:3]
        overlap = len(set(top_shap) & set(feature_names)) / max(len(top_shap), 1)
        return Evidence(
            claim=f"SHAP-교란변수 정합성: {overlap*100:.0f}%",
            evidence_type="domain",
            strength=overlap,
            source="shap_coherence",
            data={"top_shap": top_shap, "overlap_ratio": overlap},
        )


# ──────────────────────────────────────────────
# Critic (비판 에이전트)
# ──────────────────────────────────────────────

class CriticAgent:
    """인과 효과를 공격하는 에이전트.

    8가지 공격 벡터를 활용하여 인과 판단의 약점을 지적합니다.
    """

    def challenge(self, results: Dict[str, Any]) -> List[Evidence]:
        """파이프라인 결과에서 반론 증거를 수집합니다."""
        attacks = []

        attacks.append(self._e_value_weak(results))
        attacks.append(self._overlap_violation(results))
        attacks.append(self._ci_too_wide(results))
        attacks.append(self._placebo_failure(results))
        attacks.append(self._loo_sign_flip(results))
        attacks.append(self._meta_disagreement(results))
        attacks.append(self._subset_instability(results))
        attacks.append(self._small_sample(results))

        # None 필터링 및 비즈니스 리스크 주입
        valid = []
        for a in attacks:
            if a is not None:
                a.business_impact = self._generate_risk(a)
                valid.append(a)

        logger.info("📕 Critic (Risk Manager): %d개 공격 수집 (강도 합계: %.2f)",
                     len(valid), sum(e.strength for e in valid))
        return valid

    def _generate_risk(self, e: Evidence) -> str:
        """증거를 비즈니스 리스크로 번역 (Risk Manager Persona)."""
        if e.source == "e_value_weak":
            return "미관측 외부 변수(경기 침체 등)에 취약 → 예상치 못한 손실 위험"
        if e.source == "overlap_violation":
            return "특정 유저군에 편향된 결과 → 일반화 시 성과 하락 우려"
        if e.source == "placebo_failure":
            return "가짜 효과일 가능성 높음 → 마케팅 예산 낭비 경고"
        if e.source == "ci_too_wide":
            return "성과 변동폭이 너무 큼 → KPI 달성 불확실성 증대"
        return "운영 리스크 존재"

    def _e_value_weak(self, r: Dict) -> Optional[Evidence]:
        """E-value 취약."""
        e_val_raw = r.get("sensitivity_results", {}).get("e_value")
        if e_val_raw is None:
            return None
        e_value = e_val_raw.get("point", 0) if isinstance(e_val_raw, dict) else float(e_val_raw)
        if e_value >= 2.0:
            return None
        strength = (2.0 - e_value) / 2.0
        return Evidence(
            claim=f"E-value={e_value:.2f} < 2.0 → 미관측 교란에 취약",
            evidence_type="robustness",
            strength=strength,
            source="e_value_weak",
            data={"e_value": e_value},
        )

    def _overlap_violation(self, r: Dict) -> Optional[Evidence]:
        """Overlap 위반."""
        overlap_raw = r.get("sensitivity_results", {}).get("overlap")
        if overlap_raw is None:
            return None
        overlap = overlap_raw.get("overlap_score", 0) if isinstance(overlap_raw, dict) else float(overlap_raw)
        if overlap >= 0.7:
            return None
        strength = (0.7 - overlap) / 0.7
        return Evidence(
            claim=f"Overlap={overlap:.2f} < 0.7 → positivity 위반 위험",
            evidence_type="statistical",
            strength=strength,
            source="overlap_violation",
            data={"overlap": overlap},
        )

    def _ci_too_wide(self, r: Dict) -> Optional[Evidence]:
        """CI 과대."""
        refutation = r.get("refutation_results", {})
        bootstrap = refutation.get("bootstrap_ci", {})
        ci_lower = bootstrap.get("ci_lower")
        ci_upper = bootstrap.get("ci_upper")
        # ate는 float 또는 dict일 수 있음 (ExportCell 전/후)
        ate_raw = r.get("ate")
        if isinstance(ate_raw, dict):
            ate = ate_raw.get("value", 0)
        elif isinstance(ate_raw, (int, float)):
            ate = float(ate_raw)
        else:
            ate = None
        if ci_lower is None or ci_upper is None or ate is None or ate == 0:
            return None
        width = abs(ci_upper - ci_lower)
        ratio = width / abs(ate)
        if ratio <= 2.0:
            return None
        strength = min(ratio / 5.0, 1.0)  # 5배 이상이면 최대
        return Evidence(
            claim=f"CI 폭={width:.4f} (ATE의 {ratio:.1f}배) → 추정 불안정",
            evidence_type="statistical",
            strength=strength,
            source="ci_too_wide",
            data={"width": width, "ratio": ratio},
        )

    def _placebo_failure(self, r: Dict) -> Optional[Evidence]:
        """Placebo 실패."""
        p_value = r.get("refutation_results", {}).get("placebo_test", {}).get("p_value")
        if p_value is None or p_value >= 0.05:
            return None
        return Evidence(
            claim=f"Placebo p={p_value:.4f} < 0.05 → 허위 양성 위험",
            evidence_type="robustness",
            strength=1.0 - p_value,
            source="placebo_failure",
            data={"p_value": p_value},
        )

    def _loo_sign_flip(self, r: Dict) -> Optional[Evidence]:
        """LOO 부호 반전."""
        loo = r.get("refutation_results", {}).get("leave_one_out", {})
        details = loo.get("details", [])
        if not details:
            return None
        flips = [d for d in details if d.get("sign_flip", False)]
        if not flips:
            return None
        flip_ratio = len(flips) / len(details)
        return Evidence(
            claim=f"교란변수 {len(flips)}개 제거 시 ATE 부호 반전 ({flip_ratio*100:.0f}%)",
            evidence_type="robustness",
            strength=flip_ratio,
            source="loo_sign_flip",
            data={"flip_count": len(flips), "flip_ratio": flip_ratio},
        )

    def _meta_disagreement(self, r: Dict) -> Optional[Evidence]:
        """메타러너 불일치."""
        consensus = r.get("meta_learner_results", {}).get("ensemble", {}).get("consensus", 1)
        if consensus >= 0.6:
            return None
        strength = 1.0 - consensus
        return Evidence(
            claim=f"메타러너 합의율 {consensus*100:.0f}% < 60% → 추정 방향 불확실",
            evidence_type="statistical",
            strength=strength,
            source="meta_disagreement",
            data={"consensus": consensus},
        )

    def _subset_instability(self, r: Dict) -> Optional[Evidence]:
        """Subset 불안정."""
        stability = r.get("refutation_results", {}).get("subset_validation", {}).get("avg_stability")
        if stability is None or stability >= 0.85:
            return None
        strength = 1.0 - stability
        return Evidence(
            claim=f"서브셋 안정성={stability:.2f} < 0.85 → 표본 의존적",
            evidence_type="robustness",
            strength=strength,
            source="subset_instability",
            data={"avg_stability": stability},
        )

    def _small_sample(self, r: Dict) -> Optional[Evidence]:
        """표본 크기 부족."""
        df = r.get("dataframe")
        if df is None:
            return None
        n = len(df)
        if n >= 500:
            return None
        strength = max(0, 1 - n / 500)
        return Evidence(
            claim=f"n={n} < 500 → 소표본 불안정",
            evidence_type="statistical",
            strength=strength,
            source="small_sample",
            data={"n": n},
        )


# ──────────────────────────────────────────────
# Judge (판결 에이전트)
# ──────────────────────────────────────────────

class JudgeAgent:
    """양측 증거를 종합하여 최종 판결.

    증거 유형별 가중치:
      - statistical: 1.0 (통계적 증거 기본)
      - robustness: 1.2 (견고성 증거 가중)
      - domain: 0.8 (도메인 지식 할인)
    """

    DEFAULT_WEIGHTS = {
        "statistical": 1.0,
        "robustness": 1.2,
        "domain": 0.8,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def deliberate(
        self,
        pro: List[Evidence],
        con: List[Evidence],
        threshold: float = 0.7,
    ) -> Verdict:
        """양측 증거를 가중 합산하여 판결합니다.

        Args:
            pro: 옹호 증거 리스트.
            con: 비판 증거 리스트.
            threshold: CAUSAL 판결 최소 확신도.

        Returns:
            Verdict 데이터클래스.
        """
        pro_score = sum(
            e.strength * self.weights.get(e.evidence_type, 1.0)
            for e in pro
        )
        con_score = sum(
            e.strength * self.weights.get(e.evidence_type, 1.0)
            for e in con
        )

        total = pro_score + con_score + 1e-10
        confidence = pro_score / total

        verdict_str = "UNCERTAIN"
        if confidence >= threshold:
            verdict_str = "CAUSAL"
        elif confidence <= (1.0 - threshold):
            verdict_str = "NOT_CAUSAL"

        recommendation = self._generate_recommendation(verdict_str, pro, con, confidence)

        logger.info(
            "⚖️ Judge (Product Owner): %s (확신도=%.2f) → %s",
            verdict_str, confidence, recommendation,
        )

        return Verdict(
            verdict=verdict_str,
            confidence=confidence,
            pro_score=pro_score,
            con_score=con_score,
            pro_evidence=pro,
            con_evidence=con,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        verdict: str,
        pro: List[Evidence],
        con: List[Evidence],
        confidence: float,
    ) -> str:
        """구조화된 1-Pager 정책 결정서 생성 (Product Owner Persona).

        ROI, 리스크 비용, 증거 요약을 포함한 마크다운 형식의
        경영진 보고서를 렌더링합니다.
        """
        # ── 증거에서 비즈니스 지표 추출 ──
        pro_impacts = [e.business_impact for e in pro if e.business_impact]
        con_impacts = [e.business_impact for e in con if e.business_impact]
        pro_sources = [f"`{e.source}` ({e.strength:.0%})" for e in pro[:5]]
        con_sources = [f"`{e.source}` ({e.strength:.0%})" for e in con[:5]]

        # ── 판결별 보고서 구성 ──
        if verdict == "CAUSAL":
            icon = "🚀"
            action = "전면 배포 승인 (Rollout 100%)" if confidence > 0.9 \
                else "단계적 배포 (Rollout 20% → 50%)"
            risk_level = "LOW" if confidence > 0.9 else "MEDIUM"
        elif verdict == "NOT_CAUSAL":
            icon = "🛑"
            action = "배포 중단 — 리소스 회수 권장"
            risk_level = "HIGH"
        else:
            icon = "⚖️"
            action = "5% 트래픽 A/B 테스트 실시"
            risk_level = "MEDIUM-HIGH"

        # ── 1-Pager 마크다운 렌더링 ──
        report_lines = [
            f"## {icon} Policy Decision Report",
            "",
            f"**Verdict:** `{verdict}` | **Confidence:** {confidence:.1%} | **Risk Level:** {risk_level}",
            "",
            f"### Decision",
            f"**{action}**",
            "",
            "### Key Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Pro Evidence Count | {len(pro)} |",
            f"| Con Evidence Count | {len(con)} |",
            f"| Pro Score (weighted) | {sum(e.strength * self.weights.get(e.evidence_type, 1.0) for e in pro):.2f} |",
            f"| Con Score (weighted) | {sum(e.strength * self.weights.get(e.evidence_type, 1.0) for e in con):.2f} |",
            f"| Confidence | {confidence:.1%} |",
            "",
        ]

        # ── Growth Opportunity (Advocate 증거) ──
        if pro_impacts:
            report_lines += [
                "### 📈 Growth Opportunity",
                "",
            ]
            for impact in pro_impacts[:3]:
                report_lines.append(f"- {impact}")
            report_lines += [
                "",
                "**Supporting Evidence:** " + ", ".join(pro_sources),
                "",
            ]

        # ── Risk Factors (Critic 증거) ──
        if con_impacts:
            report_lines += [
                "### ⚠️ Risk Factors",
                "",
            ]
            for impact in con_impacts[:3]:
                report_lines.append(f"- {impact}")
            report_lines += [
                "",
                "**Risk Evidence:** " + ", ".join(con_sources),
                "",
            ]

        # ── 통제 조건 ──
        report_lines += [
            "### 🔍 Control Conditions",
            "",
        ]

        if verdict == "CAUSAL":
            report_lines.append(
                "- 배포 후 2주 시점에서 `TemporalCausalCell`(CausalImpact)을 통한 2차 검증 실시"
            )
            if confidence < 0.9:
                report_lines.append(
                    "- 예산 20% 소진 시점에서 중간 리뷰 필수"
                )
        elif verdict == "NOT_CAUSAL":
            report_lines.append("- 교란 변수(Confounder) 추가 탐색 후 재분석")
            report_lines.append("- 현 시점에서 정책 변경 원상복구")
        else:
            report_lines.append("- 5% 테스트 2주 운영 → 결과 기반 Go/No-Go 판단")
            report_lines.append("- 테스트 기간 중 주 1회 CausalDrift 모니터링")

        report_lines.append("")

        return "\n".join(report_lines)
