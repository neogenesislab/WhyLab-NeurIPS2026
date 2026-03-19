# -*- coding: utf-8 -*-
"""Sprint 5 E2E 통합 테스트.

전체 파이프라인을 검증합니다:
1. DoseResponseCell
2. FairnessAuditCell
3. MACDiscoveryAgent
4. ToolAugmentedDebate
5. DeepCATECell GPU/CPU 분기
"""

import numpy as np
import pandas as pd
import pytest


def _has_torch():
    """PyTorch 설치 여부 확인."""
    try:
        import torch
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────

@pytest.fixture
def synth_data():
    """합성 인과 데이터 (n=500, p=5)."""
    rng = np.random.RandomState(42)
    n = 500
    X = rng.randn(n, 5)
    T = (rng.rand(n) > 0.5).astype(float)
    tau = 2.0 * X[:, 0] + 1.0  # 이질적 처치 효과
    Y = tau * T + X[:, 1] + rng.randn(n) * 0.5
    return X, T, Y, tau


@pytest.fixture
def synth_df(synth_data):
    """합성 데이터 DataFrame."""
    X, T, Y, _ = synth_data
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["treatment"] = T
    df["outcome"] = Y
    return df


# ──────────────────────────────────────────────
# S5-1a. DoseResponseCell E2E
# ──────────────────────────────────────────────

class TestDoseResponseE2E:
    """용량-반응 분석 E2E."""

    def test_full_pipeline(self, synth_data):
        from engine.cells.dose_response_cell import DoseResponseCell
        X, _, Y, _ = synth_data
        # 연속형 처치 생성
        rng = np.random.RandomState(99)
        T_cont = rng.uniform(0, 10, len(Y))
        Y_dose = 3.0 * np.log1p(T_cont) + X[:, 0] + rng.randn(len(Y)) * 0.3

        cell = DoseResponseCell()
        result = cell.estimate(X, T_cont, Y_dose)

        # 키 이름 수정: dose_grid -> t_grid
        assert "t_grid" in result
        assert "dr_curve" in result
        assert len(result["t_grid"]) > 0
        assert len(result["t_grid"]) == len(result["dr_curve"])

    def test_optimal_dose(self, synth_data):
        from engine.cells.dose_response_cell import DoseResponseCell
        X, _, Y, _ = synth_data
        rng = np.random.RandomState(99)
        T_cont = rng.uniform(0, 10, len(Y))
        Y_dose = -0.5 * (T_cont - 5) ** 2 + X[:, 0]  # 최적 용량 ≈ 5

        cell = DoseResponseCell()
        result = cell.estimate(X, T_cont, Y_dose)

        if "optimal_dose" in result:
            assert 2.0 <= result["optimal_dose"] <= 8.0


# ──────────────────────────────────────────────
# S5-1b. FairnessAuditCell E2E
# ──────────────────────────────────────────────

class TestFairnessE2E:
    """공정성 감사 E2E."""

    def test_full_audit(self, synth_data):
        from engine.cells.fairness_audit_cell import FairnessAuditCell
        X, T, Y, tau = synth_data
        
        # DataFrame 구성 (민감 속성 포함)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        df["sensitive"] = (X[:, 2] > 0).astype(int)
        df["treatment"] = T
        df["outcome"] = Y

        cell = FairnessAuditCell()
        # audit(cate, df, sensitive_attrs) 호출
        results = cell.audit(
            cate=tau,
            df=df,
            sensitive_attrs=["sensitive"],
        )

        assert len(results) == 1
        report = results[0]

        # FairnessResult 객체 속성 확인
        assert hasattr(report, "causal_parity_gap")
        assert hasattr(report, "equalized_cate_score")
        assert hasattr(report, "subgroups")
        assert len(report.subgroups) == 2


# ──────────────────────────────────────────────
# S5-1c. MACDiscoveryAgent E2E
# ──────────────────────────────────────────────

class TestMACDiscoveryE2E:
    """다중 에이전트 인과 발견 E2E."""

    def test_ensemble_discovery(self):
        from engine.agents.mac_discovery import MACDiscoveryAgent
        # 인과 구조: X0 → X1 → X2, X0 → X2
        rng = np.random.RandomState(42)
        n = 300
        X0 = rng.randn(n)
        X1 = 0.8 * X0 + rng.randn(n) * 0.3
        X2 = 0.5 * X0 + 0.6 * X1 + rng.randn(n) * 0.3
        data = np.column_stack([X0, X1, X2])

        agent = MACDiscoveryAgent()
        # 인자 수정: var_names -> variable_names
        result = agent.discover(data, variable_names=["X0", "X1", "X2"])

        # AggregatedDAG 객체 반환 확인
        assert hasattr(result, "consensus_level")
        assert hasattr(result, "edges")
        assert hasattr(result, "stability_scores")
        # 최소 1개 이상의 엣지 발견
        assert len(result.edges) > 0

    def test_stability_scores(self):
        from engine.agents.mac_discovery import MACDiscoveryAgent
        rng = np.random.RandomState(42)
        n = 200
        X0 = rng.randn(n)
        X1 = 0.9 * X0 + rng.randn(n) * 0.1
        data = np.column_stack([X0, X1])

        agent = MACDiscoveryAgent()
        result = agent.discover(data, variable_names=["X0", "X1"])

        scores = result.stability_scores
        # 강한 인과 → 높은 안정성 점수 존재
        assert len(scores) > 0
        assert max(scores.values()) > 0.3


# ──────────────────────────────────────────────
# S5-1d. ToolAugmentedDebate E2E
# ──────────────────────────────────────────────

class TestToolDebateE2E:
    """도구 강화 토론 E2E."""

    def test_verified_verdict(self, synth_data):
        from engine.agents.tool_debate import ToolAugmentedDebate
        X, T, Y, tau = synth_data

        debate = ToolAugmentedDebate()
        
        # context 딕셔너리로 묶어서 전달
        context = {
            "X": X, "T": T, "Y": Y,
            "cate_estimates": {"t_learner": tau},
            "ate": float(np.mean(tau)),
            "outcome_std": float(np.std(Y)),
            # 도구들이 필요로 하는 추가 정보
            "meta_learners": {"t_learner": {"ate": float(np.mean(tau))}},
        }
        
        verdict = debate.verify(context)

        # DaVVerdict 객체 확인
        assert verdict.verdict in ["VERIFIED", "REFUTED", "UNCERTAIN"]
        assert 0.0 <= verdict.confidence <= 1.0
        
        # 로그 확인
        log = debate.get_tool_log()
        assert isinstance(log, list)

    def test_tool_invocations_logged(self, synth_data):
        from engine.agents.tool_debate import ToolAugmentedDebate
        X, T, Y, tau = synth_data

        debate = ToolAugmentedDebate()
        context = {
            "X": X, "T": T, "Y": Y,
            "cate_estimates": {"t_learner": tau},
            "ate": float(np.mean(tau)),
            "outcome_std": float(np.std(Y)),
            "meta_learners": {"t_learner": {"ate": float(np.mean(tau))}},
        }
        
        _ = debate.verify(context)

        log = debate.get_tool_log()
        assert len(log) > 0
        for entry in log:
            assert "tool" in entry


# ──────────────────────────────────────────────
# S5-1e. DeepCATECell E2E (GPU/CPU 분기)
# ──────────────────────────────────────────────

@pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed (optional dependency)",
)
class TestDeepCATEE2E:
    """DeepCATECell GPU/CPU 분기 E2E."""

    @pytest.fixture
    def deep_data(self):
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n, 10)
        T = (rng.rand(n) > 0.5).astype(float)
        Y = 2.0 * T * X[:, 0] + X[:, 1] + rng.randn(n) * 0.5
        tau_true = 2.0 * X[:, 0]
        return X, T, Y, tau_true

    def test_dragonnet_fit_predict(self, deep_data):
        try:
            from engine.cells.deep_cate_cell import DeepCATECell, DeepCATEConfig
        except ImportError:
            pytest.skip("PyTorch 미설치")

        X, T, Y, _ = deep_data
        cfg = DeepCATEConfig(
            architecture="dragonnet",
            shared_dims=(32,),
            head_dims=(16,),
            epochs=5,  # 테스트 속도를 위해 축소
            batch_size=64,
        )
        cell = DeepCATECell(deep_config=cfg)
        cell.fit(X, T, Y)
        cate = cell.predict_cate(X)

        assert cate.shape == (len(X),)
        assert not np.any(np.isnan(cate))

    def test_tarnet_fit_predict(self, deep_data):
        try:
            from engine.cells.deep_cate_cell import DeepCATECell, DeepCATEConfig
        except ImportError:
            pytest.skip("PyTorch 미설치")

        X, T, Y, _ = deep_data
        cfg = DeepCATEConfig(
            architecture="tarnet",
            shared_dims=(32,),
            head_dims=(16,),
            epochs=5,
            batch_size=64,
        )
        cell = DeepCATECell(deep_config=cfg)
        cell.fit(X, T, Y)
        outcomes = cell.predict_outcomes(X)

        assert "y0" in outcomes
        assert "y1" in outcomes
        assert "cate" in outcomes
        assert outcomes["cate"].shape == (len(X),)

    def test_cpu_fallback(self, deep_data):
        """use_gpu=False → CPU에서도 정상 실행."""
        try:
            from engine.cells.deep_cate_cell import DeepCATECell, DeepCATEConfig
        except ImportError:
            pytest.skip("PyTorch 미설치")

        X, T, Y, _ = deep_data
        cfg = DeepCATEConfig(
            architecture="tarnet",
            shared_dims=(16,),
            head_dims=(8,),
            epochs=2,
            batch_size=64,
            use_gpu=False,
        )
        cell = DeepCATECell(deep_config=cfg)
        cell.fit(X, T, Y)
        cate = cell.predict_cate(X)
        assert cate.shape == (len(X),)


# ──────────────────────────────────────────────
# S5-1f. 전체 파이프라인 통합
# ──────────────────────────────────────────────

class TestFullPipelineE2E:
    """전체 WhyLab 파이프라인 통합 테스트."""

    def test_end_to_end(self, synth_data):
        """데이터 → 인과 발견 → CATE 추정 → 공정성 감사 → 토론 검증."""
        X, T, Y, tau = synth_data

        # Step 1: MAC 인과 발견
        from engine.agents.mac_discovery import MACDiscoveryAgent
        mac = MACDiscoveryAgent()
        # var_names -> variable_names 수정
        dag_result = mac.discover(X[:, :3], variable_names=["x0", "x1", "x2"])
        assert hasattr(dag_result, "consensus_level")

        # Step 2: 공정성 감사
        from engine.cells.fairness_audit_cell import FairnessAuditCell
        
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        df["sensitive"] = (X[:, 2] > 0).astype(int)
        
        fairness = FairnessAuditCell()
        # audit 인자 수정
        results = fairness.audit(
            cate=tau,
            df=df,
            sensitive_attrs=["sensitive"],
        )
        assert len(results) > 0

        # Step 3: 토론 검증
        from engine.agents.tool_debate import ToolAugmentedDebate
        debate = ToolAugmentedDebate()
        
        context = {
            "X": X, "T": T, "Y": Y,
            "cate_estimates": {"ground_truth": tau},
            "ate": float(np.mean(tau)),
            "outcome_std": float(np.std(Y)),
            "meta_learners": {"ground_truth": {"ate": float(np.mean(tau))}},
        }
        
        verdict = debate.verify(context)
        assert verdict.verdict in ["VERIFIED", "REFUTED", "UNCERTAIN"]
