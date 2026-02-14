# -*- coding: utf-8 -*-
"""Debate-as-Verification (DaV) — 토론 기반 인과 검증 프로토콜.

인과 주장(Claim)을 다중 에이전트 토론으로 검증합니다.
단순 토론이 아닌 **증거 기반 교차 심문** 프로토콜:

1. 주장(Claim): 인과 효과 추정 결과를 주장으로 구성
2. 증거 수집(Evidence): ATE, CI, MetaLearner, Sensitivity 등 자동 수집
3. 교차 심문(Cross-Exam): Advocate가 주장, Critic이 반증, 교차 심문
4. 판결(Verdict): 증거 체인 기반 최종 판결
5. 감사(Audit): 판결 근거를 역추적 가능한 로그로 기록

R&D 스프린트 1: DaV 프로토콜 (축 2-2).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class Evidence:
    """개별 증거 항목."""
    source: str              # "ate", "meta_learner", "sensitivity", "refutation", ...
    claim: str               # 증거가 뒷받침하는 주장
    direction: str           # "supports" | "contradicts" | "neutral"
    strength: float          # 0.0 ~ 1.0
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DaVClaim:
    """검증 대상 인과 주장."""
    statement: str           # "T→Y 인과 효과가 존재한다"
    treatment: str
    outcome: str
    ate: float = 0.0
    confidence_interval: List[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass
class CrossExamRecord:
    """교차 심문 기록."""
    agent: str               # "advocate" | "critic"
    argument: str
    evidence_refs: List[str] = field(default_factory=list)
    strength: float = 0.0


@dataclass
class DaVVerdict:
    """DaV 판결 결과."""
    claim: DaVClaim
    verdict: str             # "VERIFIED" | "REFUTED" | "INSUFFICIENT"
    confidence: float = 0.0
    evidence_chain: List[Evidence] = field(default_factory=list)
    cross_examination: List[CrossExamRecord] = field(default_factory=list)
    reasoning: str = ""
    timestamp: str = ""


# ──────────────────────────────────────────────
# DaV 프로토콜 엔진
# ──────────────────────────────────────────────

class DaVProtocol:
    """Debate-as-Verification 프로토콜.

    파이프라인 결과(컨텍스트)를 받아 인과 주장을 자동 검증합니다.

    Args:
        verification_threshold: "VERIFIED" 판정 최소 신뢰도.
        refutation_threshold: "REFUTED" 판정 최소 반증 강도.
    """

    def __init__(
        self,
        verification_threshold: float = 0.65,
        refutation_threshold: float = 0.60,
    ) -> None:
        self.verification_threshold = verification_threshold
        self.refutation_threshold = refutation_threshold

    def verify(self, context: Dict[str, Any]) -> DaVVerdict:
        """파이프라인 컨텍스트에서 인과 주장을 검증합니다.

        Args:
            context: 파이프라인 실행 결과 딕셔너리.

        Returns:
            DaVVerdict: 검증 판결.
        """
        # ──── 1. 주장 구성 ────
        claim = self._construct_claim(context)

        # ──── 2. 증거 수집 ────
        evidence_chain = self._collect_evidence(context, claim)

        # ──── 3. 교차 심문 ────
        cross_exam = self._cross_examine(evidence_chain, claim)

        # ──── 4. 판결 ────
        verdict = self._render_verdict(claim, evidence_chain, cross_exam)

        logger.info(
            "⚖️ DaV 판결: %s (신뢰도=%.1f%%) — 증거 %d개, 심문 %d건",
            verdict.verdict,
            verdict.confidence * 100,
            len(evidence_chain),
            len(cross_exam),
        )

        return verdict

    def _construct_claim(self, context: Dict[str, Any]) -> DaVClaim:
        """파이프라인 결과에서 인과 주장을 구성합니다."""
        treatment = context.get("treatment_col", "T")
        outcome = context.get("outcome_col", "Y")

        ate_raw = context.get("ate", {})
        if isinstance(ate_raw, dict):
            ate = ate_raw.get("point_estimate", ate_raw.get("value", 0))
            ci = [ate_raw.get("ci_lower", 0), ate_raw.get("ci_upper", 0)]
        elif isinstance(ate_raw, (int, float)):
            ate = float(ate_raw)
            ci = [0.0, 0.0]
        else:
            ate, ci = 0.0, [0.0, 0.0]

        return DaVClaim(
            statement=f"{treatment} → {outcome} 인과 효과가 존재한다",
            treatment=treatment,
            outcome=outcome,
            ate=ate,
            confidence_interval=ci,
        )

    def _collect_evidence(self, context: Dict[str, Any], claim: DaVClaim) -> List[Evidence]:
        """파이프라인 결과에서 증거를 자동 수집합니다."""
        evidence = []

        # 증거 1: ATE 추정
        if claim.ate != 0:
            ci_excludes_zero = (claim.confidence_interval[0] > 0 or claim.confidence_interval[1] < 0) \
                if claim.confidence_interval != [0.0, 0.0] else False
            evidence.append(Evidence(
                source="ate",
                claim="인과 효과 크기",
                direction="supports" if ci_excludes_zero else "neutral",
                strength=0.7 if ci_excludes_zero else 0.3,
                detail={"ate": claim.ate, "ci": claim.confidence_interval},
            ))

        # 증거 2: Meta-Learner 합의
        meta = context.get("meta_learners", {})
        if isinstance(meta, dict) and meta:
            ates = []
            for name, result in meta.items():
                if isinstance(result, dict):
                    ates.append(result.get("ate", result.get("mean_cate", 0)))

            if ates:
                agreement = 1.0 - (np.std(ates) / (abs(np.mean(ates)) + 1e-10))
                agreement = max(0, min(1, agreement))
                all_same_sign = all(a > 0 for a in ates) or all(a < 0 for a in ates)

                evidence.append(Evidence(
                    source="meta_learner_consensus",
                    claim="메타러너 간 합의",
                    direction="supports" if all_same_sign and agreement > 0.5 else "contradicts",
                    strength=agreement,
                    detail={"ates": ates, "agreement": round(agreement, 3)},
                ))

        # 증거 3: Sensitivity (E-value)
        sensitivity = context.get("sensitivity", {})
        if isinstance(sensitivity, dict):
            e_value = sensitivity.get("e_value", {})
            if isinstance(e_value, dict):
                ev = e_value.get("point", e_value.get("e_value", 0))
                evidence.append(Evidence(
                    source="e_value",
                    claim="미관측 교란 견고성",
                    direction="supports" if ev > 1.5 else "contradicts",
                    strength=min(1.0, ev / 3.0) if ev else 0.2,
                    detail={"e_value": ev},
                ))

        # 증거 4: Refutation 검정
        refutation = context.get("refutation", {})
        if isinstance(refutation, dict):
            placebo = refutation.get("placebo", {})
            if isinstance(placebo, dict):
                passed = placebo.get("passed", False)
                evidence.append(Evidence(
                    source="refutation_placebo",
                    claim="위약 검정",
                    direction="supports" if passed else "contradicts",
                    strength=0.8 if passed else 0.2,
                    detail={"passed": passed},
                ))

        # 증거 5: 준실험 (IV/DiD/RDD)
        qe = context.get("quasi_experimental", {})
        if isinstance(qe, dict) and qe:
            evidence.append(Evidence(
                source="quasi_experimental",
                claim="준실험 방법론 검증",
                direction="supports",
                strength=0.6,
                detail={"methods": list(qe.keys())},
            ))

        # 증거 6: 인과 그래프 발견
        dag_edges = context.get("dag_edges", [])
        if dag_edges:
            t_to_y = any(
                (e[0] == claim.treatment and e[1] == claim.outcome)
                for e in dag_edges if isinstance(e, (list, tuple)) and len(e) >= 2
            )
            evidence.append(Evidence(
                source="causal_discovery",
                claim="인과 그래프에 T→Y 엣지 존재",
                direction="supports" if t_to_y else "contradicts",
                strength=0.7 if t_to_y else 0.3,
                detail={"t_to_y": t_to_y, "total_edges": len(dag_edges)},
            ))

        return evidence

    def _cross_examine(
        self,
        evidence: List[Evidence],
        claim: DaVClaim,
    ) -> List[CrossExamRecord]:
        """증거를 기반으로 교차 심문을 수행합니다."""
        import numpy as np

        records = []
        supporting = [e for e in evidence if e.direction == "supports"]
        contradicting = [e for e in evidence if e.direction == "contradicts"]

        # Advocate: 지지 증거 기반 주장
        if supporting:
            avg_strength = np.mean([e.strength for e in supporting])
            records.append(CrossExamRecord(
                agent="advocate",
                argument=(
                    f"{claim.treatment}→{claim.outcome} 인과 효과 ATE={claim.ate:.4f}. "
                    f"{len(supporting)}개 증거가 지지 (평균 강도={avg_strength:.2f})."
                ),
                evidence_refs=[e.source for e in supporting],
                strength=avg_strength,
            ))

        # Critic: 반증 증거 기반 반론
        if contradicting:
            avg_strength = np.mean([e.strength for e in contradicting])
            records.append(CrossExamRecord(
                agent="critic",
                argument=(
                    f"주의: {len(contradicting)}개 증거가 인과 관계에 의문 제기. "
                    f"핵심 우려: {', '.join(e.source for e in contradicting)}."
                ),
                evidence_refs=[e.source for e in contradicting],
                strength=avg_strength,
            ))

        # 교차 심문: Advocate가 Critic의 반론에 대응
        if contradicting and supporting:
            strongest_support = max(supporting, key=lambda e: e.strength)
            records.append(CrossExamRecord(
                agent="advocate_rebuttal",
                argument=(
                    f"반론 대응: {strongest_support.source} 증거 "
                    f"(강도={strongest_support.strength:.2f})가 핵심 반증을 상쇄. "
                    f"전체 증거 {len(supporting)}/{len(evidence)} 지지."
                ),
                evidence_refs=[strongest_support.source],
                strength=strongest_support.strength,
            ))

        return records

    def _render_verdict(
        self,
        claim: DaVClaim,
        evidence: List[Evidence],
        cross_exam: List[CrossExamRecord],
    ) -> DaVVerdict:
        """최종 판결을 내립니다."""
        import numpy as np

        if not evidence:
            return DaVVerdict(
                claim=claim,
                verdict="INSUFFICIENT",
                confidence=0.0,
                evidence_chain=evidence,
                cross_examination=cross_exam,
                reasoning="증거 부족으로 판단 불가.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # 가중 점수 계산
        support_score = sum(
            e.strength for e in evidence if e.direction == "supports"
        )
        contra_score = sum(
            e.strength for e in evidence if e.direction == "contradicts"
        )
        total = support_score + contra_score + 1e-10

        confidence = support_score / total

        # 판결
        if confidence >= self.verification_threshold:
            verdict = "VERIFIED"
            reasoning = (
                f"인과 효과 검증됨. "
                f"지지 증거 {support_score:.2f} vs 반증 {contra_score:.2f}. "
                f"신뢰도 {confidence:.1%}."
            )
        elif (1 - confidence) >= self.refutation_threshold:
            verdict = "REFUTED"
            reasoning = (
                f"인과 효과 기각됨. "
                f"반증 증거가 지지 증거를 압도 ({contra_score:.2f} vs {support_score:.2f})."
            )
        else:
            verdict = "INSUFFICIENT"
            reasoning = (
                f"판단 보류. 지지 {support_score:.2f} vs 반증 {contra_score:.2f}. "
                f"추가 증거 필요."
            )

        return DaVVerdict(
            claim=claim,
            verdict=verdict,
            confidence=round(confidence, 4),
            evidence_chain=evidence,
            cross_examination=cross_exam,
            reasoning=reasoning,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# 편의 함수
def verify_causal_claim(context: Dict[str, Any]) -> DaVVerdict:
    """파이프라인 결과에 대한 인과 주장을 DaV 프로토콜로 검증합니다."""
    protocol = DaVProtocol()
    return protocol.verify(context)
