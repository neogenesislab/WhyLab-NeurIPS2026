# -*- coding: utf-8 -*-
"""벤치마크 모듈 테스트.

1. 데이터 로더 검증 (IHDP/ACIC/Jobs)
2. BenchmarkCell 실행 + 비교표 생성 검증
3. √PEHE, ATE Bias 범위 검증
"""

import numpy as np
import pytest

from engine.config import WhyLabConfig
from engine.data.benchmark_data import (
    BenchmarkData, IHDPLoader, ACICLoader, JobsLoader, BENCHMARK_REGISTRY,
)
from engine.cells.benchmark_cell import BenchmarkCell


# ──────────────────────────────────────────────
# 데이터 로더 테스트
# ──────────────────────────────────────────────

class TestIHDPLoader:
    """IHDP 데이터 로더 검증."""

    def test_load_shape(self):
        """IHDP 로드 시 올바른 shape 반환."""
        data = IHDPLoader().load(n=200, seed=42)
        assert data.X.shape == (200, 25)
        assert data.T.shape == (200,)
        assert data.Y.shape == (200,)
        assert data.tau_true.shape == (200,)

    def test_treatment_binary(self):
        """Treatment가 0/1 이진."""
        data = IHDPLoader().load(n=500, seed=42)
        assert set(np.unique(data.T)).issubset({0.0, 1.0})

    def test_ground_truth_finite(self):
        """Ground Truth τ(x)가 유한값."""
        data = IHDPLoader().load(n=500, seed=42)
        assert np.all(np.isfinite(data.tau_true))

    def test_reproducibility(self):
        """같은 seed면 같은 데이터."""
        d1 = IHDPLoader().load(n=100, seed=123)
        d2 = IHDPLoader().load(n=100, seed=123)
        np.testing.assert_array_equal(d1.X, d2.X)
        np.testing.assert_array_equal(d1.tau_true, d2.tau_true)

    def test_overlap(self):
        """처치 비율 10~90% 사이 (overlap 보장)."""
        data = IHDPLoader().load(n=1000, seed=42)
        treat_ratio = data.T.mean()
        assert 0.1 < treat_ratio < 0.9


class TestACICLoader:
    """ACIC 데이터 로더 검증."""

    def test_load_shape(self):
        data = ACICLoader().load(n=500, seed=42)
        assert data.X.shape == (500, 58)
        assert len(data.feature_names) == 58

    def test_heterogeneous_effect(self):
        """CATE에 이질성이 있는지 확인."""
        data = ACICLoader().load(n=1000, seed=42)
        cate_std = np.std(data.tau_true)
        assert cate_std > 0.1, "CATE 이질성이 너무 낮음"


class TestJobsLoader:
    """Jobs 데이터 로더 검증."""

    def test_load_shape(self):
        data = JobsLoader().load(n=300, seed=42)
        assert data.X.shape == (300, 8)

    def test_positive_treatment(self):
        """Jobs CATE는 비음수 (클리핑 적용)."""
        data = JobsLoader().load(n=500, seed=42)
        assert np.all(data.tau_true >= 0)


# ──────────────────────────────────────────────
# BenchmarkCell 테스트
# ──────────────────────────────────────────────

class TestBenchmarkCell:
    """BenchmarkCell 통합 검증."""

    @pytest.fixture
    def config(self):
        cfg = WhyLabConfig()
        cfg.benchmark.datasets = ["ihdp"]
        cfg.benchmark.n_replications = 2  # 테스트 속도
        cfg.dml.lgbm_n_estimators = 50    # 경량화
        cfg.dml.use_gpu = False           # CI 환경 대비
        return cfg

    def test_execute_returns_results(self, config):
        """BenchmarkCell 실행 시 결과 반환."""
        cell = BenchmarkCell(config)
        result = cell.execute({})
        assert "benchmark_results" in result
        assert "benchmark_table" in result

    def test_ihdp_results_structure(self, config):
        """IHDP 결과에 모든 메타러너가 포함."""
        cell = BenchmarkCell(config)
        result = cell.execute({})
        ihdp = result["benchmark_results"]["ihdp"]
        assert "S-Learner" in ihdp
        assert "DR-Learner" in ihdp
        assert "Ensemble" in ihdp
        assert "LinearDML" in ihdp

    def test_pehe_finite(self, config):
        """√PEHE가 유한값."""
        cell = BenchmarkCell(config)
        result = cell.execute({})
        for name, metrics in result["benchmark_results"]["ihdp"].items():
            assert np.isfinite(metrics["pehe_mean"]), f"{name} PEHE NaN"

    def test_table_format(self, config):
        """비교표가 마크다운 형식."""
        cell = BenchmarkCell(config)
        result = cell.execute({})
        table = result["benchmark_table"]
        assert "| Method |" in table
        assert "S-Learner" in table
