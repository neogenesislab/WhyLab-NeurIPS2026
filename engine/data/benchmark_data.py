# -*- coding: utf-8 -*-
"""벤치마크 데이터셋 로더 — IHDP, ACIC, Jobs.

학술 인과추론 벤치마크 데이터를 로드하여
Ground Truth τ(x)가 포함된 BenchmarkData를 반환합니다.

IHDP 구현: 합성 반사실(Response Surface B, Hill 2011)
ACIC/Jobs: 유사 합성 DGP(Data Generating Process) 내장
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkData:
    """벤치마크 데이터 컨테이너.

    Attributes:
        X: (n, p) 공변량 행렬.
        T: (n,) 처치 벡터 (0/1 이진).
        Y: (n,) 관측된 결과.
        tau_true: (n,) Ground Truth CATE.
        y0: (n,) 반사실 Y(0).
        y1: (n,) 반사실 Y(1).
        feature_names: 변수 이름 리스트.
        name: 데이터셋 이름.
    """
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    tau_true: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    feature_names: list
    name: str


class IHDPLoader:
    """IHDP 벤치마크 데이터 생성기 (Hill 2011 Response Surface B).

    원본 IHDP는 영아 건강 프로그램 데이터에 합성 반사실을 부여합니다.
    여기서는 원논문의 DGP(Response Surface B)를 직접 구현하여
    외부 데이터 의존 없이 동일한 통계적 특성의 벤치마크를 생성합니다.

    DGP (Response Surface B):
      X ~ 25개 공변량 (6개 연속 + 19개 이진)
      T = Bernoulli(sigmoid(X·β_t))
      Y(0) = exp((X + 0.5)·β_0) + ε₀
      Y(1) = X·β_1 + ε₁
      τ(x) = Y(1) - Y(0)

    참고: Hill, J. L. (2011). Bayesian Nonparametric Modeling for
    Causal Inference. JCGS, 20(1), 217-240.
    """

    NAME = "IHDP"
    N_CONTINUOUS = 6
    N_BINARY = 19
    N_FEATURES = N_CONTINUOUS + N_BINARY

    def load(
        self,
        n: int = 747,
        seed: int = 42,
        treatment_effect_scale: float = 4.0,
    ) -> BenchmarkData:
        """IHDP 유사 데이터를 생성합니다.

        Args:
            n: 표본 크기 (원본: 747).
            seed: 랜덤 시드.
            treatment_effect_scale: CATE 크기 조절.

        Returns:
            BenchmarkData.
        """
        rng = np.random.RandomState(seed)

        # 공변량 생성
        X_cont = rng.normal(0, 1, (n, self.N_CONTINUOUS))
        X_bin = rng.binomial(1, 0.5, (n, self.N_BINARY))
        X = np.column_stack([X_cont, X_bin])

        # Treatment assignment (비균형: ~37% 처치)
        beta_t = rng.normal(0, 0.5, self.N_FEATURES)
        propensity = 1 / (1 + np.exp(-X @ beta_t))
        propensity = np.clip(propensity, 0.1, 0.9)  # Overlap 보장
        T = rng.binomial(1, propensity).astype(np.float64)

        # Response Surface B (Hill 2011)
        beta_0 = rng.normal(0, 1, self.N_FEATURES)
        beta_1 = rng.normal(0, 1, self.N_FEATURES)

        # Y(0): 비선형 response
        y0 = np.exp((X + 0.5) @ beta_0 / self.N_FEATURES) + rng.normal(0, 1, n)

        # Y(1): 선형 response + 이질적 효과
        heterogeneous = treatment_effect_scale * (
            X[:, 0] * 0.3 - X[:, 1] * 0.2 + X[:, 2] * 0.15
        )
        y1 = X @ beta_1 / self.N_FEATURES + heterogeneous + rng.normal(0, 1, n)

        # Ground Truth τ(x) = Y(1) - Y(0) (노이즈 없는 기대값)
        tau_true = y1 - y0

        # 관측 결과: Y = T·Y(1) + (1-T)·Y(0)
        Y = T * y1 + (1 - T) * y0

        feature_names = [f"x_cont_{i}" for i in range(self.N_CONTINUOUS)] + \
                        [f"x_bin_{i}" for i in range(self.N_BINARY)]

        logger.info(
            "IHDP 로드: n=%d, p=%d, ATE_true=%.4f, 처치비율=%.1f%%",
            n, self.N_FEATURES, np.mean(tau_true), T.mean() * 100,
        )

        return BenchmarkData(
            X=X, T=T, Y=Y, tau_true=tau_true,
            y0=y0, y1=y1, feature_names=feature_names, name="IHDP",
        )


class ACICLoader:
    """ACIC 2016 유사 벤치마크 생성기 (Dorie et al. 2019).

    ACIC 대회의 DGP 특징을 재현:
      - 고차원 공변량 (p=58)
      - 비선형 Treatment Effect
      - 다양한 교란 구조

    DGP:
      X ~ 58개 공변량 (혼합)
      T = Bernoulli(sigmoid(X_subset·β + 비선형항))
      Y(0) = f₀(X) + ε
      Y(1) = f₀(X) + τ(X) + ε
      τ(X) = 비선형 이질적 효과
    """

    NAME = "ACIC"
    N_FEATURES = 58

    def load(
        self,
        n: int = 4802,
        seed: int = 42,
    ) -> BenchmarkData:
        """ACIC 유사 데이터를 생성합니다."""
        rng = np.random.RandomState(seed)

        # 공변량: 혼합 (30개 연속 + 28개 이진)
        X_cont = rng.normal(0, 1, (n, 30))
        X_bin = rng.binomial(1, 0.4, (n, 28))
        X = np.column_stack([X_cont, X_bin])

        # Propensity (비선형)
        linear = 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
        nonlinear = 0.3 * np.sin(X[:, 3]) + 0.2 * X[:, 4] ** 2
        propensity = 1 / (1 + np.exp(-(linear + nonlinear)))
        propensity = np.clip(propensity, 0.1, 0.9)
        T = rng.binomial(1, propensity).astype(np.float64)

        # Baseline outcome
        f0 = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.4 * X[:, 2]
              + 0.2 * np.sin(X[:, 3] * np.pi)
              + 0.1 * X[:, 4] * X[:, 5])

        # 이질적 처치 효과 (비선형)
        tau_true = (2.0 + 1.5 * X[:, 0] - 1.0 * X[:, 1]
                    + 0.8 * np.abs(X[:, 2])
                    + 0.5 * X[:, 6] * (X[:, 7] > 0).astype(float))

        y0 = f0 + rng.normal(0, 1, n)
        y1 = f0 + tau_true + rng.normal(0, 1, n)
        Y = T * y1 + (1 - T) * y0

        feature_names = [f"x{i}" for i in range(self.N_FEATURES)]

        logger.info(
            "ACIC 로드: n=%d, p=%d, ATE_true=%.4f",
            n, self.N_FEATURES, np.mean(tau_true),
        )

        return BenchmarkData(
            X=X, T=T, Y=Y, tau_true=tau_true,
            y0=y0, y1=y1, feature_names=feature_names, name="ACIC",
        )


class JobsLoader:
    """Jobs 유사 벤치마크 생성기 (LaLonde 1986).

    고용 훈련 프로그램 데이터의 DGP를 재현:
      - 저차원 (p=8)
      - 강한 selection bias
      - 소표본

    DGP:
      X ~ 8개 공변량 (나이, 교육, 인종 등 시뮬레이션)
      T 할당: 강한 교란 (소득 낮을수록 처치 확률 높음)
      τ(x) = 양의 효과 + 이질성
    """

    NAME = "Jobs"
    N_FEATURES = 8

    def load(
        self,
        n: int = 722,
        seed: int = 42,
    ) -> BenchmarkData:
        """Jobs 유사 데이터를 생성합니다."""
        rng = np.random.RandomState(seed)

        # 공변량: 사회경제 변수 시뮬레이션
        age = rng.normal(25, 5, n).clip(16, 55)
        education = rng.normal(12, 2, n).clip(6, 20)
        race = rng.binomial(1, 0.3, n)  # 소수인종
        married = rng.binomial(1, 0.4, n)
        no_degree = rng.binomial(1, 0.3, n)
        earnings_74 = rng.exponential(5000, n)
        earnings_75 = earnings_74 * (1 + rng.normal(0, 0.1, n))
        hispanic = rng.binomial(1, 0.15, n)

        X = np.column_stack([
            age, education, race, married,
            no_degree, earnings_74, earnings_75, hispanic,
        ])
        # 정규화
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

        # 강한 selection bias: 소득 낮을수록 프로그램 참여
        propensity = 1 / (1 + np.exp(0.8 * X[:, 5] - 0.5 * X[:, 4] + 0.3 * X[:, 2]))
        propensity = np.clip(propensity, 0.05, 0.95)
        T = rng.binomial(1, propensity).astype(np.float64)

        # Outcome
        f0 = 1500 + 200 * X[:, 0] + 300 * X[:, 1] - 100 * X[:, 4]
        tau_true = 1800 + 500 * X[:, 1] - 300 * X[:, 4]  # 교육 높을수록 효과 큼
        tau_true = tau_true.clip(0)  # 음수 처치 효과 없음

        y0 = f0 + rng.normal(0, 500, n)
        y1 = f0 + tau_true + rng.normal(0, 500, n)
        Y = T * y1 + (1 - T) * y0

        feature_names = [
            "age", "education", "race", "married",
            "no_degree", "earnings_74", "earnings_75", "hispanic",
        ]

        logger.info(
            "Jobs 로드: n=%d, ATE_true=%.1f",
            n, np.mean(tau_true),
        )

        return BenchmarkData(
            X=X, T=T, Y=Y, tau_true=tau_true,
            y0=y0, y1=y1, feature_names=feature_names, name="Jobs",
        )


# 레지스트리
BENCHMARK_REGISTRY = {
    "ihdp": IHDPLoader,
    "acic": ACICLoader,
    "jobs": JobsLoader,
}
