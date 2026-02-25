# -*- coding: utf-8 -*-
"""R5 테스트 — Lyapunov 안정성 필터 + R3 Beta-Binomial CI 검증."""

import math
import pytest

from engine.audit.llm_judge.ares_evaluator import ARESEvaluator, ARESResult
from engine.audit.lyapunov import LyapunovFilter, LyapunovState


# ── Beta-Binomial CI 검증 (R3 패치) ──

class TestBetaBinomialCI:
    """Hoeffding 교체 후 Beta-Binomial CI 품질 검증."""

    def test_n10_k8_tighter_than_hoeffding(self):
        """n=10, k=8: Beta-Binomial이 Hoeffding보다 타이트."""
        bb_lower, bb_upper = ARESEvaluator._beta_binomial_ci(8, 10, 0.95)
        hf_lower, hf_upper = ARESEvaluator._hoeffding_ci(0.8, 10, 0.95)

        bb_width = bb_upper - bb_lower
        hf_width = hf_upper - hf_lower

        assert bb_width < hf_width  # 더 타이트
        assert bb_lower > hf_lower  # 하한이 더 높음
        assert bb_lower > 0.45  # 50% 넘는 하한 (유효!)

    def test_n10_k10_near_certainty(self):
        """n=10, k=10: 거의 확실 → CI 하한 높음."""
        lower, upper = ARESEvaluator._beta_binomial_ci(10, 10, 0.95)
        assert lower > 0.75  # 강한 확신

    def test_n10_k0_near_zero(self):
        """n=10, k=0: 전부 실패 → CI 상한 낮음."""
        lower, upper = ARESEvaluator._beta_binomial_ci(0, 10, 0.95)
        assert upper < 0.25

    def test_jeffreys_prior(self):
        """Jeffreys 사전 Beta(0.5, 0.5) 적용."""
        lower, upper = ARESEvaluator._beta_binomial_ci(
            5, 10, 0.95, prior_alpha=0.5, prior_beta=0.5
        )
        # 사후분포: Beta(5.5, 5.5) → mean=0.5
        assert 0.2 < lower < 0.5
        assert 0.5 < upper < 0.8


# ── Lyapunov Filter ──

class TestLyapunovClipping:
    """ζ 클리핑 검증."""

    def test_low_noise_no_clip(self):
        """노이즈 작을 때 → 클리핑 안 됨."""
        lyap = LyapunovFilter()
        safe = lyap.clip(
            proposed_zeta=0.3,
            ate=20.0,
            drift_index=0.1,
            ares_penalty=0.0,
            confidence=0.9,
        )
        assert safe == 0.3  # 클리핑 없음
        assert not lyap.get_state().was_clipped

    def test_high_noise_clips(self):
        """노이즈 클 때 → ζ 클리핑."""
        lyap = LyapunovFilter()
        safe = lyap.clip(
            proposed_zeta=0.5,
            ate=2.0,
            drift_index=0.9,
            ares_penalty=0.8,
            confidence=0.2,
        )
        assert safe < 0.5  # 클리핑됨
        assert lyap.get_state().was_clipped

    def test_minimum_zeta_preserved(self):
        """극단적 노이즈에서도 min_zeta 보장."""
        lyap = LyapunovFilter(min_zeta=0.01)
        safe = lyap.clip(
            proposed_zeta=0.5,
            ate=0.01,
            drift_index=0.99,
            ares_penalty=0.99,
            confidence=0.1,
        )
        assert safe >= 0.01

    def test_max_zeta_cap(self):
        """ζ가 max_zeta를 초과하지 않음."""
        lyap = LyapunovFilter(max_zeta=0.8)
        safe = lyap.clip(
            proposed_zeta=1.0,
            ate=100.0,
            drift_index=0.01,
            ares_penalty=0.0,
            confidence=0.99,
        )
        assert safe <= 0.8


class TestLyapunovConvergence:
    """수렴성 검증."""

    def test_stable_system_converges(self):
        """안정적 시스템 → 수렴."""
        lyap = LyapunovFilter()
        # 에너지가 감소하는 시뮬레이션
        for i in range(10):
            noise = 0.3 - i * 0.02  # 노이즈 감소
            lyap.clip(0.3, ate=20.0, drift_index=max(noise, 0.01),
                      confidence=0.5 + i * 0.05)
        assert lyap.is_converging()

    def test_diverging_system_detected(self):
        """발산 시스템 → 감지."""
        lyap = LyapunovFilter()
        # 에너지가 증가하는 시뮬레이션
        for i in range(10):
            noise = 0.1 + i * 0.08  # 노이즈 증가
            lyap.clip(0.5, ate=5.0, drift_index=noise,
                      ares_penalty=i * 0.05, confidence=0.5 - i * 0.04)
        # 노이즈 증가 → 에너지 증가 → 발산 감지
        assert not lyap.is_converging()


class TestProveStability:
    """논문용 안정성 증명."""

    def test_proof_structure(self):
        lyap = LyapunovFilter()
        for i in range(5):
            lyap.clip(0.3, ate=20.0, drift_index=0.1, confidence=0.8)
        proof = lyap.prove_stability()
        assert "is_stable" in proof
        assert "theorem" in proof
        assert "ζ_max" in proof["theorem"]

    def test_stable_yields_negative_slope(self):
        """안정 시스템 → 음의 기울기."""
        lyap = LyapunovFilter()
        for i in range(8):
            lyap.clip(0.2, ate=25.0, drift_index=max(0.3 - i * 0.03, 0.01),
                      confidence=0.6 + i * 0.04)
        proof = lyap.prove_stability()
        assert proof["is_stable"]
        assert proof["energy_trend_slope"] <= 0
