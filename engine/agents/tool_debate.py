# -*- coding: utf-8 -*-
"""ToolAugmentedDebate — 도구 강화 토론 프로토콜.

DaV 프로토콜의 Advocate/Critic에게 실제 분석 도구를 부여합니다:
  - Advocate: CATE 추정, 메타러너 실행, 민감도 분석 도구
  - Critic: 반증 테스트, 위약 대조, 교란 체크 도구

일반 DaV가 "이미 있는 증거"만 평가하는 반면,
ToolAugmented DaV는 토론 중 새 증거를 동적으로 생성합니다.

학술 참조:
  - Du et al. (2023). "Improving Factuality and Reasoning
    in Language Models through Multiagent Debate."
  - Schick et al. (2023). "Toolformer: Language Models Can
    Teach Themselves to Use Tools."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from engine.agents.dav_protocol import (
    CrossExamRecord,
    DaVClaim,
    DaVProtocol,
    DaVVerdict,
    Evidence,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 도구 정의
# ──────────────────────────────────────────────

@dataclass
class Tool:
    """에이전트 사용 도구."""
    name: str
    description: str
    role: str             # "advocate" | "critic" | "both"
    execute: Callable     # (context, claim) -> Evidence

@dataclass
class ToolCallRecord:
    """도구 호출 기록."""
    agent: str
    tool_name: str
    result: Evidence
    round_num: int


# ──────────────────────────────────────────────
# 기본 제공 도구들
# ──────────────────────────────────────────────

def tool_cate_variance(context: Dict[str, Any], claim: DaVClaim) -> Evidence:
    """메타러너 CATE 분산 분석 도구."""
    meta = context.get("meta_learners", {})
    if not isinstance(meta, dict) or not meta:
        return Evidence(
            source="tool:cate_variance",
            claim="CATE 분산 분석 불가",
            direction="neutral",
            strength=0.1,
            detail={"reason": "no_meta_learners"},
        )

    ates = []
    for name, result in meta.items():
        if isinstance(result, dict):
            ates.append(result.get("ate", result.get("mean_cate", 0)))

    if not ates:
        return Evidence(
            source="tool:cate_variance",
            claim="CATE 값 없음",
            direction="neutral",
            strength=0.1,
        )

    cv = np.std(ates) / (abs(np.mean(ates)) + 1e-10)
    consistent = cv < 0.5

    return Evidence(
        source="tool:cate_variance",
        claim=f"메타러너 CATE 변동계수 CV={cv:.3f}",
        direction="supports" if consistent else "contradicts",
        strength=0.7 if consistent else 0.3,
        detail={"cv": round(cv, 4), "ates": ates, "consistent": consistent},
    )


def tool_effect_size_check(context: Dict[str, Any], claim: DaVClaim) -> Evidence:
    """효과 크기 실질적 의미 검증 도구."""
    ate = claim.ate
    # Cohen's d 기준: 0.2=작음, 0.5=중간, 0.8=큼
    y_std = context.get("outcome_std", 1.0)
    cohens_d = abs(ate) / (y_std + 1e-10)

    if cohens_d >= 0.5:
        direction = "supports"
        strength = min(1.0, cohens_d / 1.0)
        msg = f"실질적 효과 크기 (Cohen's d={cohens_d:.2f})"
    elif cohens_d >= 0.2:
        direction = "supports"
        strength = 0.4
        msg = f"소량 효과 (Cohen's d={cohens_d:.2f})"
    else:
        direction = "contradicts"
        strength = 0.5
        msg = f"무시할 효과 크기 (Cohen's d={cohens_d:.2f})"

    return Evidence(
        source="tool:effect_size",
        claim=msg,
        direction=direction,
        strength=strength,
        detail={"cohens_d": round(cohens_d, 4), "ate": ate, "y_std": y_std},
    )


def tool_placebo_refutation(context: Dict[str, Any], claim: DaVClaim) -> Evidence:
    """위약 대조 반증 도구."""
    refutation = context.get("refutation", {})
    placebo = refutation.get("placebo", {}) if isinstance(refutation, dict) else {}

    if not isinstance(placebo, dict) or not placebo:
        # 위약 검정 미수행 → 자체 간이 검정
        data = context.get("raw_data")
        if data is None:
            return Evidence(
                source="tool:placebo",
                claim="위약 검정 불가 (데이터 없음)",
                direction="neutral",
                strength=0.2,
            )

        return Evidence(
            source="tool:placebo",
            claim="위약 검정 미수행 — 인과 주장 약화",
            direction="contradicts",
            strength=0.4,
            detail={"reason": "not_conducted"},
        )

    passed = placebo.get("passed", False)
    p_value = placebo.get("p_value", None)

    return Evidence(
        source="tool:placebo",
        claim=f"위약 검정 {'통과' if passed else '실패'}",
        direction="supports" if passed else "contradicts",
        strength=0.8 if passed else 0.6,
        detail={"passed": passed, "p_value": p_value},
    )


def tool_overlap_check(context: Dict[str, Any], claim: DaVClaim) -> Evidence:
    """처치/대조 그룹 겹침 검증 도구 (Positivity assumption)."""
    propensity = context.get("propensity_scores")
    if propensity is None:
        return Evidence(
            source="tool:overlap",
            claim="성향점수 미제공 — 겹침 검증 불가",
            direction="neutral",
            strength=0.2,
        )

    ps = np.asarray(propensity)
    # 겹침 위반: 극단적 성향 점수 비율
    extreme = np.mean((ps < 0.05) | (ps > 0.95))

    if extreme < 0.05:
        return Evidence(
            source="tool:overlap",
            claim=f"양호한 겹침 (극단값 {extreme:.1%})",
            direction="supports",
            strength=0.7,
            detail={"extreme_ratio": round(extreme, 4)},
        )
    else:
        return Evidence(
            source="tool:overlap",
            claim=f"겹침 위반 의심 (극단값 {extreme:.1%})",
            direction="contradicts",
            strength=0.6,
            detail={"extreme_ratio": round(extreme, 4)},
        )


# ──────────────────────────────────────────────
# ToolAugmentedDebate
# ──────────────────────────────────────────────

class ToolAugmentedDebate(DaVProtocol):
    """도구 강화 토론 기반 인과 검증 프로토콜.

    DaV 프로토콜을 확장하여 토론 중 도구를 동적으로 호출합니다.

    사용법:
        debate = ToolAugmentedDebate()
        verdict = debate.verify(context)  # 도구 호출 포함
    """

    # 기본 도구 세트
    DEFAULT_TOOLS = [
        Tool(
            name="cate_variance",
            description="메타러너 CATE 분산 분석",
            role="advocate",
            execute=tool_cate_variance,
        ),
        Tool(
            name="effect_size",
            description="Cohen's d 효과 크기 검증",
            role="advocate",
            execute=tool_effect_size_check,
        ),
        Tool(
            name="placebo",
            description="위약 대조 반증 검정",
            role="critic",
            execute=tool_placebo_refutation,
        ),
        Tool(
            name="overlap",
            description="처치/대조 겹침(Positivity) 검증",
            role="critic",
            execute=tool_overlap_check,
        ),
    ]

    def __init__(
        self,
        verification_threshold: float = 0.65,
        refutation_threshold: float = 0.60,
        n_rounds: int = 2,
        tools: Optional[List[Tool]] = None,
    ):
        super().__init__(
            verification_threshold=verification_threshold,
            refutation_threshold=refutation_threshold,
        )
        self.n_rounds = n_rounds
        self.tools = tools if tools is not None else self.DEFAULT_TOOLS
        self.tool_call_log: List[ToolCallRecord] = []

    def verify(self, context: Dict[str, Any]) -> DaVVerdict:
        """도구 강화 검증을 수행합니다.

        1단계: 기본 증거 수집 (DaV 방식)
        2단계: 도구 호출 라운드 (Advocate → Critic 반복)
        3단계: 통합 판결
        """
        self.tool_call_log = []

        # 1. 주장 구성
        claim = self._construct_claim(context)

        # 2. 기본 증거 수집
        evidence = self._collect_evidence(context, claim)

        # 3. 도구 강화 라운드
        for round_num in range(self.n_rounds):
            # Advocate 도구 호출
            advocate_tools = [t for t in self.tools if t.role in ("advocate", "both")]
            for tool in advocate_tools:
                try:
                    new_evidence = tool.execute(context, claim)
                    evidence.append(new_evidence)
                    self.tool_call_log.append(ToolCallRecord(
                        agent="advocate",
                        tool_name=tool.name,
                        result=new_evidence,
                        round_num=round_num,
                    ))
                except Exception as e:
                    logger.warning("도구 %s 호출 실패: %s", tool.name, e)

            # Critic 도구 호출
            critic_tools = [t for t in self.tools if t.role in ("critic", "both")]
            for tool in critic_tools:
                try:
                    new_evidence = tool.execute(context, claim)
                    evidence.append(new_evidence)
                    self.tool_call_log.append(ToolCallRecord(
                        agent="critic",
                        tool_name=tool.name,
                        result=new_evidence,
                        round_num=round_num,
                    ))
                except Exception as e:
                    logger.warning("도구 %s 호출 실패: %s", tool.name, e)

        # 4. 교차 심문
        cross_exam = self._cross_examine(evidence, claim)

        # 도구 호출 기록을 교차 심문에 추가
        tool_summary = CrossExamRecord(
            agent="tool_augmented",
            argument=(
                f"도구 {len(self.tool_call_log)}회 호출 완료. "
                f"Advocate 도구: {len([t for t in self.tool_call_log if t.agent == 'advocate'])}개, "
                f"Critic 도구: {len([t for t in self.tool_call_log if t.agent == 'critic'])}개."
            ),
            evidence_refs=[t.tool_name for t in self.tool_call_log],
            strength=0.5,
        )
        cross_exam.append(tool_summary)

        # 5. 판결
        verdict = self._render_verdict(claim, evidence, cross_exam)

        logger.info(
            "🔧 ToolAugmented DaV: %s (conf=%.1f%%) — 증거 %d개, 도구 %d회",
            verdict.verdict,
            verdict.confidence * 100,
            len(evidence),
            len(self.tool_call_log),
        )

        return verdict

    def get_tool_log(self) -> List[Dict[str, Any]]:
        """도구 호출 로그를 반환합니다."""
        return [
            {
                "agent": t.agent,
                "tool": t.tool_name,
                "direction": t.result.direction,
                "strength": t.result.strength,
                "round": t.round_num,
            }
            for t in self.tool_call_log
        ]
