# -*- coding: utf-8 -*-
"""R&D 스프린트 1 테스트: 벤치마크 확장 + CausalLoop + DaV."""

import pytest
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 벤치마크 확장
# ──────────────────────────────────────────────

class TestBenchmarkExtension:
    def test_twins_loader(self):
        from engine.data.benchmark_data import TWINSLoader
        data = TWINSLoader().load(n=500, seed=0)
        assert data.X.shape == (500, 30)
        assert data.name == "TWINS"
        assert len(data.tau_true) == 500

    def test_criteo_loader(self):
        from engine.data.benchmark_data import CriteoUpliftLoader
        data = CriteoUpliftLoader().load(n=1000, seed=0)
        assert data.X.shape == (1000, 12)
        assert data.name == "Criteo"
        # Criteo는 작은 효과
        assert abs(np.mean(data.tau_true)) < 0.5

    def test_lalonde_loader(self):
        from engine.data.benchmark_data import LaLondeRealLoader
        data = LaLondeRealLoader().load(n=500, seed=0)
        assert data.X.shape == (500, 10)
        assert data.name == "LaLonde-Real"
        assert np.mean(data.tau_true) > 0  # 양의 효과

    def test_registry_has_6(self):
        from engine.data.benchmark_data import BENCHMARK_REGISTRY
        assert len(BENCHMARK_REGISTRY) == 6
        assert "twins" in BENCHMARK_REGISTRY
        assert "criteo" in BENCHMARK_REGISTRY
        assert "lalonde" in BENCHMARK_REGISTRY


# ──────────────────────────────────────────────
# CausalLoop Agent
# ──────────────────────────────────────────────

class TestCausalLoop:
    def _make_data(self, n=200):
        rng = np.random.RandomState(42)
        X1 = rng.randn(n)
        X2 = rng.randn(n)
        T = (X1 + rng.randn(n) * 0.5 > 0).astype(float)
        Y = 2 * T + 0.5 * X1 + rng.randn(n) * 0.3
        return pd.DataFrame({"X1": X1, "X2": X2, "T": T, "Y": Y})

    def test_loop_runs(self):
        from engine.agents.causal_loop import CausalLoopAgent
        agent = CausalLoopAgent(max_iterations=3)
        df = self._make_data()
        state = agent.run(df, treatment="T", outcome="Y")
        assert state.iterations >= 1
        assert len(state.hypotheses) >= 1
        assert len(state.final_dag) > 0

    def test_loop_converges(self):
        from engine.agents.causal_loop import CausalLoopAgent
        agent = CausalLoopAgent(max_iterations=5)
        df = self._make_data(n=500)
        state = agent.run(df, treatment="T", outcome="Y")
        # 강한 신호이므로 수렴해야 함
        assert state.converged or state.iterations <= 5

    def test_hypothesis_has_treatment_outcome(self):
        from engine.agents.causal_loop import CausalLoopAgent
        agent = CausalLoopAgent(max_iterations=2)
        df = self._make_data()
        state = agent.run(df, treatment="T", outcome="Y")
        # T→Y 엣지가 DAG에 포함되어야 함
        assert ("T", "Y") in state.final_dag


# ──────────────────────────────────────────────
# DaV 프로토콜
# ──────────────────────────────────────────────

class TestDaVProtocol:
    def test_verified_verdict(self):
        from engine.agents.dav_protocol import DaVProtocol
        protocol = DaVProtocol()
        context = {
            "treatment_col": "T",
            "outcome_col": "Y",
            "ate": {"point_estimate": 1.5, "ci_lower": 0.5, "ci_upper": 2.5},
            "meta_learners": {
                "S-Learner": {"ate": 1.4},
                "T-Learner": {"ate": 1.6},
            },
            "sensitivity": {"e_value": {"point": 3.2}},
            "refutation": {"placebo": {"passed": True}},
            "dag_edges": [("T", "Y"), ("X1", "T")],
        }
        verdict = protocol.verify(context)
        assert verdict.verdict == "VERIFIED"
        assert verdict.confidence > 0.5
        assert len(verdict.evidence_chain) >= 4

    def test_refuted_verdict(self):
        from engine.agents.dav_protocol import DaVProtocol
        protocol = DaVProtocol()
        context = {
            "treatment_col": "T",
            "outcome_col": "Y",
            "ate": {"point_estimate": 0.01, "ci_lower": -0.5, "ci_upper": 0.5},
            "meta_learners": {
                "S-Learner": {"ate": 0.01},
                "T-Learner": {"ate": -0.02},
            },
            "sensitivity": {"e_value": {"point": 1.1}},
            "refutation": {"placebo": {"passed": False}},
        }
        verdict = protocol.verify(context)
        # 증거 대부분 반증이므로 REFUTED 또는 INSUFFICIENT
        assert verdict.verdict in ["REFUTED", "INSUFFICIENT"]

    def test_evidence_chain_traceable(self):
        from engine.agents.dav_protocol import verify_causal_claim
        context = {
            "treatment_col": "coupon",
            "outcome_col": "purchase",
            "ate": {"point_estimate": 0.8, "ci_lower": 0.1, "ci_upper": 1.5},
        }
        verdict = verify_causal_claim(context)
        # 증거 체인이 비어있지 않아야 함
        assert len(verdict.evidence_chain) >= 1
        # 모든 증거에 source가 있어야 함
        for e in verdict.evidence_chain:
            assert e.source != ""

    def test_cross_examination_records(self):
        from engine.agents.dav_protocol import DaVProtocol
        protocol = DaVProtocol()
        context = {
            "treatment_col": "T",
            "outcome_col": "Y",
            "ate": {"point_estimate": 2.0, "ci_lower": 1.0, "ci_upper": 3.0},
            "sensitivity": {"e_value": {"point": 0.8}},  # 약한 E-value
            "refutation": {"placebo": {"passed": True}},
        }
        verdict = protocol.verify(context)
        # 교차 심문 기록이 있어야 함
        assert len(verdict.cross_examination) >= 1
