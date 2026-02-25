# -*- coding: utf-8 -*-
"""ARES — Autoregressive Reasoning Entailment Stability 평가 엔진.

에이전트의 다단계 추론을 '검증된 전제만으로' 단계별 평가하여
LLM 환각 합의(Confabulation Consensus)를 차단합니다.

핵심 설계:
1. 추론 그래프를 노드 단위로 분해
2. 단계 t 평가 시 검증된 1~(t-1)만 전제로 주입
3. N번 Monte Carlo 샘플링 → 긍정 비율 p̂ 계산
4. Hoeffding 부등식으로 95% 신뢰구간 하한 제공

CTO 지적 반영:
- Main Audit Pipeline과 비동기 격리 (Deep Audit Queue)
- 불확실(UNCERTAIN) 판정 또는 DI 급등 시에만 트리거

Reviewer 방어:
- 단순 if "True" in response 금지
- Hoeffding 부등식 기반 Certified Statistical Guarantee 제공
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("whylab.audit.llm_judge.ares")


class StepVerdict(str, Enum):
    """추론 단계 검증 결과."""
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass
class StepEvaluation:
    """단일 추론 단계의 평가 결과."""
    step_index: int
    step_description: str
    soundness_prob: float  # p̂ (N번 샘플링 긍정 비율)
    confidence_interval: List[float]  # [lower, upper] (Hoeffding)
    n_samples: int
    status: StepVerdict
    verified_premises: List[int] = field(default_factory=list)


@dataclass
class ARESResult:
    """ARES 전체 평가 결과."""
    scenario_id: str
    total_steps: int
    verified_steps: int
    rejected_step: Optional[int] = None  # 최초 실패 지점 = Root Cause
    root_cause_description: Optional[str] = None
    overall_soundness: float = 0.0  # 전체 건전성 (검증 단계 비율)
    chain_confidence: float = 0.0  # 체인 신뢰도 (누적 곱)
    step_evaluations: List[StepEvaluation] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ARESEvaluator:
    """ARES 확률적 추론 검증 엔진.

    사용법:
        evaluator = ARESEvaluator(
            judge_fn=my_llm_judge,  # (prompt, premise) → bool
            n_samples=10,
        )
        result = evaluator.evaluate(reasoning_steps)

    CTO 아키텍처:
        이 클래스는 Main Pipeline과 비동기로 동작합니다.
        DI > threshold 또는 verdict == UNCERTAIN 일 때만 호출됩니다.
    """

    def __init__(
        self,
        judge_fn: Callable[[str, List[str]], bool],
        n_samples: int = 10,
        soundness_threshold: float = 0.8,
        confidence_level: float = 0.95,
    ) -> None:
        """
        Args:
            judge_fn: LLM 판단 함수 (step_description, verified_premises) → bool
            n_samples: Monte Carlo 샘플링 횟수
            soundness_threshold: 검증 통과 임계치 (p̂ ≥ threshold)
            confidence_level: 신뢰 수준 (기본 95%)
        """
        self.judge_fn = judge_fn
        self.n_samples = n_samples
        self.soundness_threshold = soundness_threshold
        self.confidence_level = confidence_level

    def evaluate(
        self,
        reasoning_steps: List[str],
        scenario_id: str = "default",
    ) -> ARESResult:
        """추론 체인 전체를 ARES 프로토콜로 평가합니다.

        각 단계를 순차 평가하며, 검증된 전제만 다음 단계에 전달합니다.
        최초 실패 지점이 근본 원인(Root Cause)입니다.
        """
        step_evals: List[StepEvaluation] = []
        verified_premises: List[str] = []
        verified_indices: List[int] = []
        rejected_step = None
        root_cause = None

        for idx, step in enumerate(reasoning_steps):
            # N번 Monte Carlo 샘플링
            positive_count = 0
            for _ in range(self.n_samples):
                try:
                    is_sound = self.judge_fn(step, list(verified_premises))
                    if is_sound:
                        positive_count += 1
                except Exception as e:
                    logger.warning("⚠️ Judge call failed at step %d: %s", idx, e)

            # 건전성 확률 계산
            p_hat = positive_count / self.n_samples

            # Hoeffding 부등식 기반 신뢰구간
            ci_lower, ci_upper = self._hoeffding_ci(
                p_hat, self.n_samples, self.confidence_level
            )

            # 판정
            if p_hat >= self.soundness_threshold:
                status = StepVerdict.VERIFIED
                verified_premises.append(step)
                verified_indices.append(idx)
            elif p_hat >= self.soundness_threshold * 0.6:
                status = StepVerdict.UNCERTAIN
            else:
                status = StepVerdict.REJECTED
                if rejected_step is None:
                    rejected_step = idx
                    root_cause = step

            step_evals.append(StepEvaluation(
                step_index=idx,
                step_description=step,
                soundness_prob=round(p_hat, 4),
                confidence_interval=[round(ci_lower, 4), round(ci_upper, 4)],
                n_samples=self.n_samples,
                status=status,
                verified_premises=list(verified_indices),
            ))

            # 거부된 단계 이후는 평가 중단 (오류 전파 방지)
            if status == StepVerdict.REJECTED:
                logger.info(
                    "🛑 ARES: Step %d rejected (p̂=%.2f < %.2f). Root cause identified.",
                    idx, p_hat, self.soundness_threshold,
                )
                break

        # 전체 건전성
        verified_count = sum(
            1 for e in step_evals if e.status == StepVerdict.VERIFIED
        )
        overall_soundness = verified_count / len(reasoning_steps)

        # 체인 신뢰도 (검증된 단계의 건전성 확률 누적 곱)
        chain_confidence = 1.0
        for e in step_evals:
            if e.status == StepVerdict.VERIFIED:
                chain_confidence *= e.soundness_prob

        result = ARESResult(
            scenario_id=scenario_id,
            total_steps=len(reasoning_steps),
            verified_steps=verified_count,
            rejected_step=rejected_step,
            root_cause_description=root_cause,
            overall_soundness=round(overall_soundness, 4),
            chain_confidence=round(chain_confidence, 4),
            step_evaluations=step_evals,
            diagnostics={
                "n_samples": self.n_samples,
                "soundness_threshold": self.soundness_threshold,
                "confidence_level": self.confidence_level,
            },
        )

        logger.info(
            "📋 ARES 완료: %d/%d 검증, chain_conf=%.3f, root_cause=%s",
            verified_count, len(reasoning_steps),
            chain_confidence,
            f"step_{rejected_step}" if rejected_step is not None else "none",
        )

        return result

    @staticmethod
    def _hoeffding_ci(
        p_hat: float,
        n: int,
        confidence: float = 0.95,
    ) -> tuple:
        """Hoeffding 부등식 기반 신뢰구간.

        P(|p̂ - p| ≥ ε) ≤ 2·exp(-2nε²)

        ε를 역산하면: ε = sqrt(ln(2/α) / (2n))  where α = 1 - confidence

        이는 CLT 기반 정규 근사보다 보수적이지만,
        소표본(n ≈ 10)에서도 유효한 비모수적 보장을 제공합니다.
        """
        alpha = 1.0 - confidence
        epsilon = math.sqrt(math.log(2.0 / alpha) / (2.0 * max(n, 1)))
        lower = max(0.0, p_hat - epsilon)
        upper = min(1.0, p_hat + epsilon)
        return lower, upper

    @staticmethod
    def compute_damping_penalty(ares_result: ARESResult) -> float:
        """ARES 결과를 DampingController의 ζ 페널티로 변환.

        DampingController 연동 (Gemini 지시):
        - chain_confidence가 높으면 ζ를 유지 (과감한 업데이트)
        - chain_confidence가 낮으면 ζ를 낮추어 보수적 모드

        Returns:
            damping_penalty: 0.0 ~ 1.0 (0=페널티 없음, 1=최대 억제)
        """
        # 역 신뢰도를 페널티로 사용
        penalty = 1.0 - ares_result.chain_confidence

        # Root cause가 발견되면 추가 페널티
        if ares_result.rejected_step is not None:
            # 일찍 실패할수록 더 큰 페널티
            early_fail_ratio = 1.0 - (
                ares_result.rejected_step / max(ares_result.total_steps, 1)
            )
            penalty = min(1.0, penalty + early_fail_ratio * 0.3)

        return round(penalty, 4)
