# -*- coding: utf-8 -*-
"""Phase 9-2/9-3 통합 테스트: DiscoveryCell + AutoCausalCell."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from engine.config import WhyLabConfig
from engine.cells.discovery_cell import DiscoveryCell
from engine.cells.auto_causal_cell import AutoCausalCell, DataProfile


@pytest.fixture
def synthetic_df():
    """합성 테스트 데이터."""
    np.random.seed(42)
    n = 500
    age = np.random.normal(40, 10, n)
    income = np.random.normal(50000, 15000, n)
    credit_score = np.random.normal(700, 50, n)
    credit_limit = 10000 + 0.5 * income + 0.3 * credit_score + np.random.normal(0, 2000, n)
    is_default = (np.random.random(n) < 0.3).astype(float)
    return pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "credit_limit": credit_limit,
        "is_default": is_default,
    })


@pytest.fixture
def config():
    return WhyLabConfig()


# ──────────────────────────────────────────────
# DiscoveryCell
# ──────────────────────────────────────────────

class TestDiscoveryCell:

    def test_dag_only_mode(self, config, synthetic_df):
        """T/Y 지정 시 DAG만 발견."""
        cell = DiscoveryCell(config)
        inputs = {
            "dataframe": synthetic_df,
            "treatment_col": "credit_limit",
            "outcome_col": "is_default",
            "feature_names": ["age", "income", "credit_score"],
        }
        result = cell.execute(inputs)
        assert "dag_edges" in result
        assert result["discovery_mode"] == "dag_only"
        assert isinstance(result["dag_edges"], list)
        assert result["dag_edge_count"] >= 0

    def test_auto_discovery_mode(self, config, synthetic_df):
        """T/Y 미지정 시 전체 자동 발견."""
        cell = DiscoveryCell(config)
        inputs = {
            "dataframe": synthetic_df,
            "feature_names": ["age", "income", "credit_score"],
        }
        result = cell.execute(inputs)
        assert "dag_edges" in result
        assert result["discovery_mode"] == "auto"
        assert "treatment_col" in result
        assert "outcome_col" in result
        assert result["discovered_roles"]["treatment"] is not None

    def test_no_dataframe_skip(self, config):
        """데이터프레임 없으면 건너뜀."""
        cell = DiscoveryCell(config)
        result = cell.execute({})
        assert "dag_edges" not in result


# ──────────────────────────────────────────────
# AutoCausalCell
# ──────────────────────────────────────────────

class TestAutoCausalCell:

    def test_profile_binary_treatment(self, config, synthetic_df):
        """이진 처치 프로파일."""
        cell = AutoCausalCell(config)
        inputs = {
            "dataframe": synthetic_df,
            "treatment_col": "is_default",
            "outcome_col": "credit_limit",
            "feature_names": ["age", "income", "credit_score"],
        }
        result = cell.execute(inputs)
        profile = result["data_profile"]
        assert profile["treatment_type"] == "binary"
        assert profile["n_samples"] == 500

    def test_profile_continuous_treatment(self, config, synthetic_df):
        """연속 처치 프로파일."""
        cell = AutoCausalCell(config)
        inputs = {
            "dataframe": synthetic_df,
            "treatment_col": "credit_limit",
            "outcome_col": "is_default",
            "feature_names": ["age", "income", "credit_score"],
        }
        result = cell.execute(inputs)
        profile = result["data_profile"]
        assert profile["treatment_type"] == "continuous"
        assert profile["outcome_type"] == "binary"

    def test_recommendation_structure(self, config, synthetic_df):
        """추천 결과 구조 검증."""
        cell = AutoCausalCell(config)
        inputs = {
            "dataframe": synthetic_df,
            "treatment_col": "credit_limit",
            "outcome_col": "is_default",
            "feature_names": ["age", "income", "credit_score"],
        }
        result = cell.execute(inputs)
        rec = result["auto_recommendation"]
        assert "primary_method" in rec
        assert "nuisance_model" in rec
        assert "recommended_learners" in rec
        assert "reasoning" in rec
        assert isinstance(rec["recommended_learners"], list)
        assert len(rec["recommended_learners"]) > 0

    def test_small_sample_warning(self, config):
        """소표본 경고."""
        np.random.seed(0)
        small_df = pd.DataFrame({
            "t": np.random.binomial(1, 0.5, 50),
            "y": np.random.normal(0, 1, 50),
            "x": np.random.normal(0, 1, 50),
        })
        cell = AutoCausalCell(config)
        inputs = {
            "dataframe": small_df,
            "treatment_col": "t",
            "outcome_col": "y",
            "feature_names": ["x"],
        }
        result = cell.execute(inputs)
        assert any("소표본" in w for w in result["data_profile"]["warnings"])

    def test_no_dataframe_skip(self, config):
        """데이터프레임 없으면 건너뜀."""
        cell = AutoCausalCell(config)
        result = cell.execute({})
        assert "auto_recommendation" not in result


# ──────────────────────────────────────────────
# Orchestrator 통합
# ──────────────────────────────────────────────

class TestOrchestratorIntegration:

    def test_13_cell_pipeline(self, config):
        """13셀 파이프라인 구성 확인."""
        from engine.orchestrator import Orchestrator
        orch = Orchestrator(config)
        assert "discovery" in orch.cells
        assert "auto_causal" in orch.cells
        assert len(orch.cells) == 13
