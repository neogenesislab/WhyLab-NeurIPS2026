# -*- coding: utf-8 -*-
"""Debate 시스템 테스트.

1. AdvocateAgent 증거 수집
2. CriticAgent 반론 수집
3. JudgeAgent 판결
4. DebateCell 통합 실행
"""

import numpy as np
import pandas as pd
import pytest

from engine.config import WhyLabConfig
from engine.agents.debate import (
    Evidence, Verdict, AdvocateAgent, CriticAgent, JudgeAgent,
)
from engine.cells.debate_cell import DebateCell


# ──────────────────────────────────────────────
# 테스트 데이터 (시뮬레이션된 파이프라인 결과)
# ──────────────────────────────────────────────

def _make_strong_causal_results():
    """강한 인과관계 결과 (CAUSAL 예상)."""
    return {
        "ate": -0.034,
        "meta_learner_results": {
            "ensemble": {"consensus": 0.8},
        },
        "refutation_results": {
            "placebo_test": {"p_value": 0.85},
            "bootstrap_ci": {"ci_lower": -0.06, "ci_upper": -0.01},
            "leave_one_out": {"any_sign_flip": False, "details": []},
            "subset_validation": {"avg_stability": 0.92},
        },
        "sensitivity_results": {
            "e_value": 2.5,
            "overlap": 0.82,
            "gates_results": {"f_stat_significant": True},
        },
        "conformal_results": {
            "ci_lower_mean": -0.07,
            "ci_upper_mean": -0.005,
        },
        "feature_importance": {"income": 0.3, "age": 0.2, "credit": 0.15},
        "feature_names": ["income", "age", "credit_score"],
    }


def _make_weak_results():
    """약한/불확실 결과 (UNCERTAIN 예상)."""
    return {
        "ate": -0.002,
        "meta_learner_results": {
            "ensemble": {"consensus": 0.4},
        },
        "refutation_results": {
            "placebo_test": {"p_value": 0.12},
            "bootstrap_ci": {"ci_lower": -0.03, "ci_upper": 0.025},
            "leave_one_out": {
                "any_sign_flip": True,
                "details": [
                    {"variable": "income", "sign_flip": True},
                    {"variable": "age", "sign_flip": False},
                ],
            },
            "subset_validation": {"avg_stability": 0.65},
        },
        "sensitivity_results": {
            "e_value": 1.1,
            "overlap": 0.55,
        },
        "conformal_results": {
            "ci_lower_mean": -0.05,
            "ci_upper_mean": 0.04,
        },
        "feature_importance": {},
        "feature_names": [],
    }


# ──────────────────────────────────────────────
# AdvocateAgent 테스트
# ──────────────────────────────────────────────

class TestAdvocateAgent:
    def test_gather_strong_evidence(self):
        """강한 결과에서 증거 수집."""
        agent = AdvocateAgent()
        evidence = agent.gather_evidence(_make_strong_causal_results())
        assert len(evidence) >= 5
        assert all(isinstance(e, Evidence) for e in evidence)

    def test_evidence_strength_range(self):
        """증거 강도는 [0, 1] 범위."""
        agent = AdvocateAgent()
        evidence = agent.gather_evidence(_make_strong_causal_results())
        for e in evidence:
            assert 0.0 <= e.strength <= 1.0, f"{e.source}: {e.strength}"

    def test_handles_missing_data(self):
        """빈 결과에서도 에러 없이 동작."""
        agent = AdvocateAgent()
        evidence = agent.gather_evidence({})
        assert isinstance(evidence, list)


# ──────────────────────────────────────────────
# CriticAgent 테스트
# ──────────────────────────────────────────────

class TestCriticAgent:
    def test_challenge_weak_results(self):
        """약한 결과에 대한 공격 수집."""
        agent = CriticAgent()
        attacks = agent.challenge(_make_weak_results())
        assert len(attacks) >= 3  # e-value, overlap, subset 등

    def test_no_attacks_on_strong(self):
        """강한 결과에는 공격이 적어야 함."""
        agent = CriticAgent()
        attacks = agent.challenge(_make_strong_causal_results())
        total_strength = sum(a.strength for a in attacks)
        assert total_strength < 2.0  # 미약한 공격만

    def test_handles_missing_data(self):
        """빈 결과에도 에러 없이 동작."""
        agent = CriticAgent()
        attacks = agent.challenge({})
        assert isinstance(attacks, list)


# ──────────────────────────────────────────────
# JudgeAgent 테스트
# ──────────────────────────────────────────────

class TestJudgeAgent:
    def test_causal_verdict(self):
        """강한 옹호 → CAUSAL 판결."""
        judge = JudgeAgent()
        pro = [Evidence("강한 증거", "statistical", 0.9, "test")]
        con = [Evidence("약한 공격", "statistical", 0.1, "test")]
        verdict = judge.deliberate(pro, con)
        assert verdict.verdict == "CAUSAL"
        assert verdict.confidence > 0.7

    def test_not_causal_verdict(self):
        """강한 비판 → NOT_CAUSAL 판결."""
        judge = JudgeAgent()
        pro = [Evidence("약한 증거", "statistical", 0.1, "test")]
        con = [Evidence("강한 공격", "robustness", 0.95, "test")]
        verdict = judge.deliberate(pro, con)
        assert verdict.verdict == "NOT_CAUSAL"

    def test_uncertain_verdict(self):
        """균형 → UNCERTAIN 판결."""
        judge = JudgeAgent()
        pro = [Evidence("보통 증거", "statistical", 0.5, "test")]
        con = [Evidence("보통 공격", "statistical", 0.5, "test")]
        verdict = judge.deliberate(pro, con)
        assert verdict.verdict == "UNCERTAIN"

    def test_robustness_weight(self):
        """견고성 증거 가중치 반영."""
        judge = JudgeAgent()
        pro = [Evidence("통계", "statistical", 0.5, "t1"),
               Evidence("견고성", "robustness", 0.5, "t2")]
        con = []
        verdict = judge.deliberate(pro, con)
        # robustness 가중치 1.2 → pro_score > 1.0
        assert verdict.pro_score > 1.0

    def test_recommendation_exists(self):
        """판결에 추가 제안이 포함."""
        judge = JudgeAgent()
        pro = [Evidence("e", "statistical", 0.5, "t")]
        con = [Evidence("e", "robustness", 0.5, "t")]
        verdict = judge.deliberate(pro, con)
        assert len(verdict.recommendation) > 0


# ──────────────────────────────────────────────
# DebateCell 통합 테스트
# ──────────────────────────────────────────────

class TestDebateCell:
    @pytest.fixture
    def config(self):
        cfg = WhyLabConfig()
        cfg.debate.max_rounds = 2
        return cfg

    def test_execute_strong(self, config):
        """강한 결과 → CAUSAL 판결."""
        cell = DebateCell(config)
        result = cell.execute(_make_strong_causal_results())
        assert "debate_verdict" in result
        assert "debate_summary" in result
        assert result["debate_verdict"].verdict == "CAUSAL"

    def test_execute_weak(self, config):
        """약한 결과 → UNCERTAIN 또는 NOT_CAUSAL."""
        cell = DebateCell(config)
        result = cell.execute(_make_weak_results())
        assert result["debate_verdict"].verdict in ("UNCERTAIN", "NOT_CAUSAL")

    def test_summary_structure(self, config):
        """Debate 요약 구조 검증."""
        cell = DebateCell(config)
        result = cell.execute(_make_strong_causal_results())
        summary = result["debate_summary"]
        assert "verdict" in summary
        assert "confidence" in summary
        assert "pro_evidence" in summary
        assert "con_evidence" in summary
        assert "recommendation" in summary

    def test_multi_round(self, config):
        """UNCERTAIN 시 다라운드 진행."""
        cell = DebateCell(config)
        # 빈 결과 → 증거 적음 → 라운드 진행
        result = cell.execute({"ate": 0.001})
        assert result["debate_verdict"].rounds >= 1
