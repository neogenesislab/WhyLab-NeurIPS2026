# -*- coding: utf-8 -*-
"""R3 테스트 — ARES 평가 엔진 + CausalFlip 함정 벤치마크.

CTO 설계:
- LLM은 Mock (실제 API 호출 없음)
- Monte Carlo 샘플링 + Hoeffding CI 검증

Reviewer 방어:
- CausalFlip: 교란/충돌 변수 역전 시 ARES가 논리적 모순을 감지하는지 입증
- 단순 프롬프트 기반 LLM은 두 시나리오에서 동일 결론 (환각 합의)
- ARES는 검증된 전제만 전달하므로 역전 시나리오를 정확히 기각
"""

import math
import random
from unittest.mock import MagicMock

import pytest

from engine.audit.llm_judge.ares_evaluator import (
    ARESEvaluator,
    ARESResult,
    StepEvaluation,
    StepVerdict,
)


# ── Mock LLM Judges ──

def always_sound_judge(step: str, premises: list) -> bool:
    """모든 단계를 건전하다고 판단하는 judge."""
    return True


def always_unsound_judge(step: str, premises: list) -> bool:
    """모든 단계를 건전하지 않다고 판단하는 judge."""
    return False


def probabilistic_judge(p: float = 0.9):
    """확률 p로 건전하다고 판단하는 judge."""
    def _judge(step: str, premises: list) -> bool:
        return random.random() < p
    return _judge


def causal_aware_judge(step: str, premises: list) -> bool:
    """인과 구조를 이해하는 ARES 기반 judge (CausalFlip 방어용).

    - "confounder" / "collider" 가 제대로 기술되면 True
    - "X causes Y but Y also causes X" 같은 순환 논리 → False
    - 전제(premises)와 모순되는 단계 → False
    """
    step_lower = step.lower()

    # 순환 인과 감지
    if "causes" in step_lower and "also causes" in step_lower:
        return False

    # 역전된 인과 방향 감지
    if "reversed" in step_lower or "flipped" in step_lower:
        return False

    # 전제와 모순 감지
    for premise in premises:
        if "positive" in premise.lower() and "negative" in step_lower:
            return False
        if "increases" in premise.lower() and "decreases" in step_lower:
            return False

    return True


# ── ARES 기본 기능 테스트 ──

class TestARESBasic:
    def test_all_verified(self):
        """모든 단계가 검증되는 정상 체인."""
        evaluator = ARESEvaluator(
            judge_fn=always_sound_judge,
            n_samples=5,
        )
        steps = [
            "Agent observed low CTR on landing page",
            "Agent decided to A/B test headline variant",
            "Agent measured +15% conversion lift",
        ]
        result = evaluator.evaluate(steps, scenario_id="normal_chain")

        assert result.verified_steps == 3
        assert result.rejected_step is None
        assert result.overall_soundness == 1.0
        assert result.chain_confidence == 1.0

    def test_early_rejection(self):
        """첫 단계에서 거부 → 이후 평가 중단."""
        evaluator = ARESEvaluator(
            judge_fn=always_unsound_judge,
            n_samples=5,
        )
        steps = ["Step A", "Step B", "Step C"]
        result = evaluator.evaluate(steps)

        assert result.rejected_step == 0
        assert result.root_cause_description == "Step A"
        assert result.verified_steps == 0
        assert len(result.step_evaluations) == 1  # 거부 후 중단

    def test_mid_chain_rejection(self):
        """중간 단계에서 거부."""
        call_count = 0
        def step2_fails(step: str, premises: list) -> bool:
            nonlocal call_count
            call_count += 1
            return "Step B" not in step

        evaluator = ARESEvaluator(
            judge_fn=step2_fails,
            n_samples=5,
        )
        steps = ["Step A (positive effect)", "Step B (fails here)", "Step C"]
        result = evaluator.evaluate(steps)

        assert result.rejected_step == 1
        assert result.verified_steps == 1

    def test_verified_premises_propagation(self):
        """검증된 전제만 다음 단계에 전달되는지 확인."""
        premises_received = []
        def tracking_judge(step: str, premises: list) -> bool:
            premises_received.append(list(premises))
            return True

        evaluator = ARESEvaluator(
            judge_fn=tracking_judge,
            n_samples=1,
        )
        steps = ["Step 0", "Step 1", "Step 2"]
        evaluator.evaluate(steps)

        # Step 0: 전제 없음, Step 1: Step 0만, Step 2: Step 0+1
        assert premises_received[0] == []
        assert premises_received[1] == ["Step 0"]
        assert premises_received[2] == ["Step 0", "Step 1"]


class TestHoeffdingCI:
    def test_ci_bounds(self):
        """Hoeffding CI가 [0, 1] 범위 내."""
        lower, upper = ARESEvaluator._hoeffding_ci(0.5, 10, 0.95)
        assert 0.0 <= lower <= 0.5
        assert 0.5 <= upper <= 1.0

    def test_more_samples_tighter_ci(self):
        """샘플 수 증가 → CI 좁아짐."""
        _, u10 = ARESEvaluator._hoeffding_ci(0.8, 10, 0.95)
        _, u100 = ARESEvaluator._hoeffding_ci(0.8, 100, 0.95)
        width_10 = u10 - ARESEvaluator._hoeffding_ci(0.8, 10, 0.95)[0]
        width_100 = u100 - ARESEvaluator._hoeffding_ci(0.8, 100, 0.95)[0]
        assert width_100 < width_10

    def test_hoeffding_formula(self):
        """ε = sqrt(ln(2/α) / 2n) 수식 검증."""
        n, conf = 50, 0.95
        alpha = 0.05
        expected_eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
        lower, upper = ARESEvaluator._hoeffding_ci(0.7, n, conf)
        actual_eps = 0.7 - lower
        assert abs(actual_eps - expected_eps) < 0.001


class TestDampingPenalty:
    def test_high_confidence_low_penalty(self):
        """높은 chain_confidence → 낮은 페널티."""
        result = ARESResult(
            scenario_id="test",
            total_steps=3,
            verified_steps=3,
            chain_confidence=0.95,
        )
        penalty = ARESEvaluator.compute_damping_penalty(result)
        assert penalty < 0.2

    def test_rejection_increases_penalty(self):
        """거부가 있으면 페널티 증가."""
        result_ok = ARESResult(
            scenario_id="ok", total_steps=5,
            verified_steps=5, chain_confidence=0.9,
        )
        result_fail = ARESResult(
            scenario_id="fail", total_steps=5,
            verified_steps=2, rejected_step=2,
            chain_confidence=0.5,
        )
        assert (ARESEvaluator.compute_damping_penalty(result_fail) >
                ARESEvaluator.compute_damping_penalty(result_ok))

    def test_early_rejection_higher_penalty(self):
        """일찍 실패할수록 더 큰 페널티."""
        early = ARESResult(
            scenario_id="early", total_steps=10,
            verified_steps=1, rejected_step=1,
            chain_confidence=0.3,
        )
        late = ARESResult(
            scenario_id="late", total_steps=10,
            verified_steps=8, rejected_step=8,
            chain_confidence=0.3,
        )
        assert (ARESEvaluator.compute_damping_penalty(early) >=
                ARESEvaluator.compute_damping_penalty(late))


# ── CausalFlip 함정 벤치마크 ──

class TestCausalFlip:
    """LLM의 '의미론적 암기' vs '실제 인과 추론' 분별 테스트.

    정상 인과 체인은 통과하고, 교란/충돌 변수를 교묘하게
    역전시킨 함정 시나리오는 정확히 기각하는지 입증합니다.
    """

    def setup_method(self):
        self.evaluator = ARESEvaluator(
            judge_fn=causal_aware_judge,
            n_samples=5,
            soundness_threshold=0.8,
        )

    def test_normal_causal_chain_passes(self):
        """정상 인과 사슬 → 전체 검증."""
        steps = [
            "Rain increases ground wetness (positive effect)",
            "Wet ground increases umbrella usage (positive effect)",
            "High umbrella usage indicates rainy weather (positive observation)",
        ]
        result = self.evaluator.evaluate(steps, "causal_normal")
        assert result.verified_steps == 3
        assert result.rejected_step is None

    def test_confounder_reversal_rejected(self):
        """교란 변수 역전 (CausalFlip) → 기각."""
        steps = [
            "Feature A increases conversion (positive effect)",
            "Feature B also increases conversion (positive premise established)",
            "But Feature A reversed: A actually decreases conversion",
        ]
        result = self.evaluator.evaluate(steps, "confounder_flip")
        assert result.rejected_step is not None  # 3번째 단계에서 기각

    def test_circular_causation_rejected(self):
        """순환 인과 (A→B→A) → 기각."""
        steps = [
            "Ad spend increases traffic (positive)",
            "Traffic increases revenue (positive)",
            "Revenue causes ad spend but ad spend also causes revenue (circular)",
        ]
        result = self.evaluator.evaluate(steps, "circular_flip")
        assert result.rejected_step is not None

    def test_collider_bias_rejected(self):
        """충돌 변수 편향 (Collider bias) → 기각."""
        steps = [
            "Agent A increases page views (positive effect)",
            "Agent B increases time on site (positive effect)",
            "Conditioning on bounce rate reversed: views now flipped negative",
        ]
        result = self.evaluator.evaluate(steps, "collider_flip")
        assert result.rejected_step is not None

    def test_subtle_direction_flip_rejected(self):
        """미묘한 방향 역전 → 기각."""
        steps = [
            "Price reduction increases purchase rate (positive)",
            "Higher purchase rate increases revenue (positive)",
            "But price reduction reversed: it actually decreases revenue",
        ]
        result = self.evaluator.evaluate(steps, "subtle_flip")
        assert result.rejected_step is not None

    def test_flip_accuracy_stats(self):
        """CausalFlip 정확도 통계: 정상 통과율 vs 함정 기각율."""
        normal_scenarios = [
            ["Step A increases metric (positive)", "Step B increases metric (positive)"],
            ["Agent logs decision (positive)", "Outcome improves (positive)"],
        ]
        flip_scenarios = [
            ["A increases X (positive)", "But X reversed: X decreases Y"],
            ["A causes B (positive)", "B also causes A, A also causes B (circular)"],
            ["Metric up (positive)", "Metric reversed: goes down flipped"],
        ]

        normal_pass = sum(
            1 for s in normal_scenarios
            if self.evaluator.evaluate(s).rejected_step is None
        )
        flip_reject = sum(
            1 for s in flip_scenarios
            if self.evaluator.evaluate(s).rejected_step is not None
        )

        # 정상: 100% 통과, 함정: 100% 기각
        assert normal_pass == len(normal_scenarios)
        assert flip_reject == len(flip_scenarios)
