# -*- coding: utf-8 -*-
"""Phase 10 통합 테스트: QE + Temporal + Counterfactual 셀."""

import pytest
import numpy as np
import pandas as pd

from engine.config import WhyLabConfig
from engine.cells.quasi_experimental_cell import QuasiExperimentalCell
from engine.cells.temporal_causal_cell import TemporalCausalCell
from engine.cells.counterfactual_cell import CounterfactualCell


@pytest.fixture
def config():
    return WhyLabConfig()


@pytest.fixture
def binary_df():
    """이진 처치 합성 데이터."""
    np.random.seed(42)
    n = 500
    x = np.random.normal(0, 1, n)
    t = (x + np.random.normal(0, 0.5, n) > 0).astype(float)
    y = 2 * t + 0.5 * x + np.random.normal(0, 1, n)
    z = 0.8 * x + np.random.normal(0, 0.3, n)  # 잠재 도구 변수
    return pd.DataFrame({"x": x, "treatment": t, "outcome": y, "instrument": z})


@pytest.fixture
def time_series_df():
    """시계열 합성 데이터."""
    np.random.seed(42)
    n = 200
    t_idx = np.arange(n)
    x = np.cumsum(np.random.normal(0, 1, n))
    y = 0.5 * np.roll(x, 3) + np.random.normal(0, 1, n)  # 3-lag 인과
    treatment = np.zeros(n)
    treatment[100:] = 1  # 중간에 개입
    return pd.DataFrame({"x": x, "outcome": y, "treatment": treatment})


# ──────────────────────────────────────────────
# QuasiExperimentalCell
# ──────────────────────────────────────────────

class TestQuasiExperimental:

    def test_iv_with_instrument(self, config, binary_df):
        """명시적 도구 변수로 IV 추정."""
        cell = QuasiExperimentalCell(config)
        inputs = {
            "dataframe": binary_df,
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "feature_names": ["x"],
            "instrument_col": "instrument",
        }
        result = cell.execute(inputs)
        qe = result["quasi_experimental"]
        assert "iv" in qe
        assert qe["iv"]["method"] == "2SLS"
        assert "f_stat" in qe["iv"]

    def test_iv_auto_search(self, config, binary_df):
        """도구 변수 자동 탐색."""
        cell = QuasiExperimentalCell(config)
        inputs = {
            "dataframe": binary_df,
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "feature_names": ["x", "instrument"],
        }
        result = cell.execute(inputs)
        assert "quasi_experimental" in result

    def test_did_simulation(self, config, binary_df):
        """간이 DiD 시뮬레이션."""
        cell = QuasiExperimentalCell(config)
        inputs = {
            "dataframe": binary_df,
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "feature_names": ["x"],
        }
        result = cell.execute(inputs)
        qe = result["quasi_experimental"]
        assert "did" in qe
        assert "ate" in qe["did"]

    def test_rdd_with_explicit_cutoff(self, config):
        """명시적 절단점 RDD."""
        np.random.seed(42)
        n = 500
        running = np.random.uniform(-5, 5, n)
        treatment = (running >= 0).astype(float)
        outcome = 3 * treatment + 0.5 * running + np.random.normal(0, 1, n)
        df = pd.DataFrame({"running": running, "treatment": treatment, "outcome": outcome})

        cell = QuasiExperimentalCell(config)
        inputs = {
            "dataframe": df,
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "feature_names": ["running"],
            "running_col": "running",
            "rdd_cutoff": 0.0,
        }
        result = cell.execute(inputs)
        qe = result["quasi_experimental"]
        assert "rdd" in qe
        assert abs(qe["rdd"]["ate"] - 3.0) < 1.5  # 대략 3에 가까워야

    def test_no_dataframe_skip(self, config):
        """데이터프레임 없으면 건너뜀."""
        cell = QuasiExperimentalCell(config)
        result = cell.execute({})
        assert "quasi_experimental" not in result


# ──────────────────────────────────────────────
# TemporalCausalCell
# ──────────────────────────────────────────────

class TestTemporalCausal:

    def test_granger_causality(self, config, time_series_df):
        """Granger 인과 검정."""
        cell = TemporalCausalCell(config)
        inputs = {
            "dataframe": time_series_df,
            "treatment_col": "x",
            "outcome_col": "outcome",
        }
        result = cell.execute(inputs)
        tc = result["temporal_causal"]
        assert "granger" in tc
        assert "p_value" in tc["granger"]

    def test_lag_correlation(self, config, time_series_df):
        """시차 상관 분석."""
        cell = TemporalCausalCell(config)
        inputs = {
            "dataframe": time_series_df,
            "treatment_col": "x",
            "outcome_col": "outcome",
        }
        result = cell.execute(inputs)
        tc = result["temporal_causal"]
        assert "lag_correlation" in tc
        assert "optimal_lag" in tc["lag_correlation"]

    def test_causal_impact_auto_detect(self, config, time_series_df):
        """개입 시점 자동 감지 + CausalImpact."""
        cell = TemporalCausalCell(config)
        inputs = {
            "dataframe": time_series_df,
            "treatment_col": "treatment",
            "outcome_col": "outcome",
        }
        result = cell.execute(inputs)
        tc = result["temporal_causal"]
        assert "causal_impact" in tc
        assert "absolute_effect" in tc["causal_impact"]

    def test_no_dataframe_skip(self, config):
        """데이터프레임 없으면 건너뜀."""
        cell = TemporalCausalCell(config)
        result = cell.execute({})
        assert "temporal_causal" not in result


# ──────────────────────────────────────────────
# CounterfactualCell
# ──────────────────────────────────────────────

class TestCounterfactual:

    def test_ate_based_counterfactual(self, config, binary_df):
        """ATE 기반 간이 반사실 (CATE 없을 때)."""
        cell = CounterfactualCell(config)
        inputs = {
            "dataframe": binary_df,
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "feature_names": ["x"],
            "ate": 2.0,
        }
        result = cell.execute(inputs)
        cf = result["counterfactual"]
        assert "summary" in cf
        assert cf["summary"]["n_individuals"] == 500
        assert cf["summary"]["mean_ite"] == pytest.approx(2.0)

    def test_cate_based_counterfactual(self, config, binary_df):
        """CATE 기반 반사실."""
        np.random.seed(42)
        cate = np.random.normal(2.0, 0.5, 500)
        cell = CounterfactualCell(config)
        inputs = {
            "dataframe": binary_df,
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "feature_names": ["x"],
            "cate_values": cate.tolist(),
        }
        result = cell.execute(inputs)
        cf = result["counterfactual"]
        assert cf["summary"]["positive_effect_ratio"] > 0.9  # 대부분 양의 효과
        assert len(cf["summary"]["top_beneficiaries"]) == 5

    def test_distribution_counterfactual(self, config, binary_df):
        """분포 반사실 (모두 처치/미처치)."""
        cell = CounterfactualCell(config)
        inputs = {
            "dataframe": binary_df,
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "feature_names": ["x"],
            "ate": 2.0,
        }
        result = cell.execute(inputs)
        dist = result["counterfactual"]["summary"]["distribution_shift"]
        assert "all_treated_mean" in dist
        assert "all_control_mean" in dist
        assert dist["all_treated_mean"] > dist["all_control_mean"]

    def test_no_dataframe_skip(self, config):
        """데이터프레임 없으면 건너뜀."""
        cell = CounterfactualCell(config)
        result = cell.execute({})
        assert "counterfactual" not in result


# ──────────────────────────────────────────────
# Orchestrator 통합
# ──────────────────────────────────────────────

class TestOrchestratorPhase10:

    def test_16_cell_pipeline(self, config):
        """16셀 파이프라인 구성 확인."""
        from engine.orchestrator import Orchestrator
        orch = Orchestrator(config)
        assert "quasi_experimental" in orch.cells
        assert "temporal_causal" in orch.cells
        assert "counterfactual" in orch.cells
        assert len(orch.cells) == 16
