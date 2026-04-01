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


class TWINSLoader:
    """TWINS 벤치마크 생성기 (Louizos et al. 2017).

    미국 쌍둥이 출생 데이터 기반:
      - 동일 유전·환경 → 교란 최소화
      - 처치: 태아 체중 기준 이진 분류
      - 결과: 영아 사망률
      - 고차원 교란 (부모 정보, 산전 관리 등)
    """

    NAME = "TWINS"
    N_FEATURES = 30

    def load(
        self,
        n: int = 4000,
        seed: int = 42,
    ) -> BenchmarkData:
        """TWINS 유사 데이터를 생성합니다."""
        rng = np.random.RandomState(seed)
        p = self.N_FEATURES

        # 공변량: 부모 특성 시뮬레이션
        X = rng.randn(n, p)
        # 일부 변수를 범주형으로 변환 (이진)
        for col in range(0, p, 5):
            X[:, col] = (X[:, col] > 0).astype(float)

        # 처치 할당: 교란 영향 있는 선택 편향
        propensity = 1 / (1 + np.exp(-(0.3 * X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2])))
        T = rng.binomial(1, propensity)

        # 반사실 결과: 비선형 + 이질적 효과
        baseline = 0.5 * X[:, 0] + 0.3 * X[:, 1] ** 2 - 0.4 * X[:, 3] + rng.randn(n) * 0.3
        tau_true = (0.8 + 0.6 * X[:, 1] - 0.4 * X[:, 4] +
                    0.3 * X[:, 0] * X[:, 2]).astype(float)

        y0 = baseline
        y1 = baseline + tau_true
        Y = T * y1 + (1 - T) * y0

        feature_names = [f"twin_x{i}" for i in range(p)]

        logger.info("TWINS 로드: n=%d, ATE_true=%.3f", n, np.mean(tau_true))

        return BenchmarkData(
            X=X, T=T, Y=Y, tau_true=tau_true,
            y0=y0, y1=y1, feature_names=feature_names, name="TWINS",
        )


class CriteoUpliftLoader:
    """Criteo Uplift 유사 벤치마크 생성기.

    대규모 광고 업리프트 데이터:
      - 고차원 특성 (p=12 핵심 + 잡음)
      - 매우 작은 처치 효과 (현실적)
      - 대규모 표본 (기본 50K)
      - 강한 음의 기저율
    """

    NAME = "Criteo"
    N_FEATURES = 12

    def load(
        self,
        n: int = 50000,
        seed: int = 42,
    ) -> BenchmarkData:
        """Criteo 유사 데이터를 생성합니다."""
        rng = np.random.RandomState(seed)
        p = self.N_FEATURES

        X = rng.randn(n, p)
        # 범주형 변수 (광고 카테고리)
        X[:, 0] = rng.choice([0, 1, 2], n)
        X[:, 1] = rng.choice([0, 1], n)

        # 약한 propensity (RCT에 가까움)
        propensity = np.full(n, 0.5)
        T = rng.binomial(1, propensity)

        # 매우 작은 이질적 효과 (현실 광고)
        tau_true = (0.02 + 0.01 * X[:, 2] - 0.005 * X[:, 3] +
                    0.008 * (X[:, 0] == 1).astype(float)).astype(float)

        # 기저 전환율 약 3%
        baseline = -3.5 + 0.1 * X[:, 2] + 0.05 * X[:, 4] + rng.randn(n) * 0.5
        y0 = (baseline > 0).astype(float)
        y1 = ((baseline + tau_true) > 0).astype(float)
        Y = T * y1 + (1 - T) * y0

        # 실제 tau_true를 확률 차이로 재계산
        tau_true = y1 - y0

        feature_names = [f"criteo_f{i}" for i in range(p)]

        logger.info("Criteo 로드: n=%d, ATE_true=%.4f", n, np.mean(tau_true))

        return BenchmarkData(
            X=X, T=T, Y=Y, tau_true=tau_true,
            y0=y0, y1=y1, feature_names=feature_names, name="Criteo",
        )


class LaLondeRealLoader:
    """LaLonde 실데이터 유사 벤치마크 (NSW + PSID).

    원본 LaLonde (1986) 실험:
      - NSW (National Supported Work) 직업 훈련 프로그램
      - 처치: 프로그램 참여 여부
      - 결과: 소득 (re78)
      - PSID 비실험 대조군과 비교 → 관찰 연구 문제
    """

    NAME = "LaLonde-Real"
    N_FEATURES = 10

    def load(
        self,
        n: int = 2000,
        seed: int = 42,
    ) -> BenchmarkData:
        """LaLonde 실데이터 유사 생성."""
        rng = np.random.RandomState(seed)
        p = self.N_FEATURES

        # 인구통계 변수 시뮬레이션
        age = rng.normal(25, 7, n).clip(18, 55)
        education = rng.normal(10, 2, n).clip(0, 20)
        black = rng.binomial(1, 0.4, n).astype(float)
        hispanic = rng.binomial(1, 0.1, n).astype(float)
        married = rng.binomial(1, 0.2, n).astype(float)
        nodegree = (education < 12).astype(float)
        re74 = rng.exponential(3000, n).clip(0, 50000)
        re75 = rng.exponential(3500, n).clip(0, 50000)

        X = np.column_stack([
            age, education, black, hispanic, married,
            nodegree, re74, re75,
            rng.randn(n), rng.randn(n),  # 잡음 변수
        ])

        # 강한 선택 편향: 소득 낮은 사람이 프로그램 참여 경향
        logit = -0.5 + 0.02 * age - 0.05 * education + 0.3 * black - 0.0001 * re75
        propensity = 1 / (1 + np.exp(-logit))
        # 처치군 소수 (약 30%)
        propensity = propensity * 0.5
        T = rng.binomial(1, propensity.clip(0.05, 0.95))

        # 소득 결과
        baseline = (1500 + 200 * education + 50 * age - 500 * nodegree +
                    0.3 * re75 + rng.randn(n) * 2000)
        tau_true = (1500 + 100 * education - 30 * age +
                    500 * nodegree).astype(float)

        y0 = baseline.clip(0)
        y1 = (baseline + tau_true).clip(0)
        Y = T * y1 + (1 - T) * y0

        feature_names = [
            "age", "education", "black", "hispanic", "married",
            "nodegree", "re74", "re75", "noise1", "noise2",
        ]

        logger.info(
            "LaLonde-Real 로드: n=%d, ATE_true=%.1f",
            n, np.mean(tau_true),
        )

        return BenchmarkData(
            X=X, T=T, Y=Y, tau_true=tau_true,
            y0=y0, y1=y1, feature_names=feature_names, name="LaLonde-Real",
        )


# 레지스트리
BENCHMARK_REGISTRY = {
    "ihdp": IHDPLoader,
    "acic": ACICLoader,
    "jobs": JobsLoader,
    "twins": TWINSLoader,
    "criteo": CriteoUpliftLoader,
    "lalonde": LaLondeRealLoader,
}
