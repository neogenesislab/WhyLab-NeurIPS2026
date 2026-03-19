# -*- coding: utf-8 -*-
"""DeepCATECell 테스트 — TARNet/DragonNet CATE 추정 검증.

검증 항목:
1. TARNet/DragonNet 학습 + 예측 동작
2. IHDP 벤치마크에서 √PEHE 측정
3. MetaLearnerCell 인터페이스 호환성
4. GPU/CPU 모드 자동 감지
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# PyTorch 필수 — 미설치 환경에서는 전체 모듈 skip
torch = pytest.importorskip("torch", reason="PyTorch not installed (optional dependency)")

# 프로젝트 루트
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.cells.deep_cate_cell import DeepCATECell, DeepCATEConfig
from engine.data.benchmark_data import IHDPLoader


# ──────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────

@pytest.fixture
def ihdp_data():
    """IHDP 벤치마크 데이터."""
    return IHDPLoader().load(n=747, seed=42)


@pytest.fixture
def small_data():
    """빠른 테스트용 소규모 합성 데이터."""
    rng = np.random.RandomState(42)
    n, p = 200, 5
    X = rng.randn(n, p)
    T = rng.binomial(1, 0.5, n).astype(float)
    tau_true = 2.0 + X[:, 0]  # 이질적 효과
    Y = X[:, 1] + tau_true * T + rng.randn(n) * 0.5
    return X, T, Y, tau_true


def _fast_config(arch: str = "tarnet") -> DeepCATEConfig:
    """빠른 테스트용 설정."""
    return DeepCATEConfig(
        architecture=arch,
        shared_dims=(32, 16),
        head_dims=(16,),
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        early_stopping_patience=5,
        use_gpu=False,
    )


# ──────────────────────────────────────────────
# TARNet 테스트
# ──────────────────────────────────────────────

class TestTARNet:
    """TARNet 아키텍처 테스트."""

    def test_fit_predict(self, small_data):
        """기본 학습 + 예측 동작."""
        X, T, Y, _ = small_data
        cell = DeepCATECell(deep_config=_fast_config("tarnet"))
        cell.fit(X, T, Y)
        cate = cell.predict_cate(X)

        assert cate.shape == (len(X),)
        assert not np.isnan(cate).any()

    def test_cate_direction(self, small_data):
        """CATE 부호가 올바른 방향인지 확인."""
        X, T, Y, tau_true = small_data
        cell = DeepCATECell(deep_config=_fast_config("tarnet"))
        cell.fit(X, T, Y)
        cate = cell.predict_cate(X)

        # 평균 CATE가 양수 (ground truth ATE > 0)
        assert np.mean(cate) > 0, f"ATE가 음수: {np.mean(cate):.4f}"

    def test_predict_outcomes(self, small_data):
        """Y₀, Y₁, CATE, propensity 예측."""
        X, T, Y, _ = small_data
        cell = DeepCATECell(deep_config=_fast_config("tarnet"))
        cell.fit(X, T, Y)
        result = cell.predict_outcomes(X)

        assert "y0" in result
        assert "y1" in result
        assert "cate" in result
        assert "propensity" in result
        np.testing.assert_array_almost_equal(
            result["cate"], result["y1"] - result["y0"],
        )


# ──────────────────────────────────────────────
# DragonNet 테스트
# ──────────────────────────────────────────────

class TestDragonNet:
    """DragonNet 아키텍처 테스트."""

    def test_fit_predict(self, small_data):
        """기본 학습 + 예측 동작."""
        X, T, Y, _ = small_data
        cell = DeepCATECell(deep_config=_fast_config("dragonnet"))
        cell.fit(X, T, Y)
        cate = cell.predict_cate(X)

        assert cate.shape == (len(X),)
        assert not np.isnan(cate).any()

    def test_propensity_output(self, small_data):
        """DragonNet 성향점수 출력 확인."""
        X, T, Y, _ = small_data
        cell = DeepCATECell(deep_config=_fast_config("dragonnet"))
        cell.fit(X, T, Y)
        result = cell.predict_outcomes(X)

        propensity = result["propensity"]
        assert propensity.shape == (len(X),)
        assert np.all(propensity >= 0) and np.all(propensity <= 1)

    def test_dragonnet_better_than_random(self, small_data):
        """DragonNet CATE가 무작위보다 나은지 확인."""
        X, T, Y, tau_true = small_data
        cell = DeepCATECell(deep_config=_fast_config("dragonnet"))
        cell.fit(X, T, Y)
        cate = cell.predict_cate(X)

        # √PEHE 계산
        pehe = np.sqrt(np.mean((cate - tau_true) ** 2))
        random_pehe = np.sqrt(np.mean(tau_true ** 2))  # CATE=0 가정

        assert pehe < random_pehe, (
            f"DragonNet √PEHE ({pehe:.4f}) ≥ 무작위 ({random_pehe:.4f})"
        )


# ──────────────────────────────────────────────
# 벤치마크 테스트
# ──────────────────────────────────────────────

class TestBenchmark:
    """IHDP 벤치마크 성능 테스트."""

    def test_ihdp_dragonnet(self, ihdp_data):
        """IHDP에서 DragonNet √PEHE 측정."""
        cfg = DeepCATEConfig(
            architecture="dragonnet",
            shared_dims=(100, 50),
            head_dims=(50,),
            epochs=50,
            batch_size=128,
            use_gpu=False,
        )
        cell = DeepCATECell(deep_config=cfg)
        cell.fit(ihdp_data.X, ihdp_data.T, ihdp_data.Y)
        cate = cell.predict_cate(ihdp_data.X)

        pehe = np.sqrt(np.mean((cate - ihdp_data.tau_true) ** 2))
        ate_bias = abs(np.mean(cate) - np.mean(ihdp_data.tau_true))

        print(f"\n  IHDP DragonNet: √PEHE={pehe:.4f}, ATE_bias={ate_bias:.4f}")

        # 관대한 임계값 (신경망은 시드에 민감)
        assert pehe < 5.0, f"√PEHE가 너무 높음: {pehe:.4f}"


# ──────────────────────────────────────────────
# 인터페이스 호환성 테스트
# ──────────────────────────────────────────────

class TestInterface:
    """MetaLearnerCell 인터페이스 호환성."""

    def test_execute_interface(self, small_data):
        """execute 메서드 호환성."""
        X, T, Y, _ = small_data
        import pandas as pd

        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        df["treatment"] = T
        df["outcome"] = Y

        cell = DeepCATECell(deep_config=_fast_config("dragonnet"))
        result = cell.execute({
            "dataframe": df,
            "feature_names": [f"x{i}" for i in range(X.shape[1])],
            "treatment_col": "treatment",
            "outcome_col": "outcome",
        })

        assert "deep_cate" in result
        assert result["deep_cate"]["architecture"] == "dragonnet"
        assert "cate" in result["deep_cate"]
        assert "ate" in result["deep_cate"]

    def test_not_fitted_error(self):
        """fit() 전 predict 호출 시 에러."""
        cell = DeepCATECell(deep_config=_fast_config())
        with pytest.raises(RuntimeError, match="fit"):
            cell.predict_cate(np.zeros((10, 5)))

    def test_name_attribute(self):
        """name 속성 존재 확인."""
        cell = DeepCATECell()
        assert hasattr(cell, "name")
        assert cell.name == "DeepCATE"
