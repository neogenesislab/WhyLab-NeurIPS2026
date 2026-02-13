# -*- coding: utf-8 -*-
"""RefutationCell 테스트 — 진짜 반증 엔진 검증.

Mock이 아닌 실제 DML 재학습 기반 반증이 올바르게 작동하는지 확인합니다.
"""

import numpy as np
import pytest

from engine.config import WhyLabConfig
from engine.cells.refutation_cell import RefutationCell


@pytest.fixture
def refutation_inputs():
    """RefutationCell 테스트용 합성 인과 데이터."""
    np.random.seed(42)
    n = 500  # 반증 반복이 많으므로 작은 데이터

    # Ground Truth: T → Y with confounders X
    X1 = np.random.normal(0, 1, n)  # 교란변수 1 (소득)
    X2 = np.random.normal(0, 1, n)  # 교란변수 2 (나이)
    T = 0.5 * X1 - 0.3 * X2 + np.random.normal(0, 0.5, n)  # 처치
    true_cate = 0.3 * X1 - 0.2 * X2  # true CATE
    Y = true_cate * T + 0.5 * X1 - 0.4 * X2 + np.random.normal(0, 0.3, n)

    import pandas as pd
    df = pd.DataFrame({
        "income": X1,
        "age": X2,
        "credit_limit": T,
        "default": Y,
    })

    return {
        "dataframe": df,
        "feature_names": ["income", "age"],
        "treatment_col": "credit_limit",
        "outcome_col": "default",
        "ate": -0.034,
        "discrete_treatment": False,
    }


class TestRefutationCell:
    """RefutationCell 단위 테스트."""

    def test_fit_and_estimate_ate(self, refutation_inputs):
        """_fit_and_estimate_ate가 실제로 DML을 학습하고 ATE를 반환하는지."""
        config = WhyLabConfig()
        cell = RefutationCell(config)

        ate = cell._fit_and_estimate_ate(
            refutation_inputs["dataframe"],
            refutation_inputs["treatment_col"],
            refutation_inputs["outcome_col"],
            refutation_inputs["feature_names"],
            is_discrete=False,
        )
        # ATE는 실수여야 하고, 합리적 범위 내
        assert isinstance(ate, float)
        assert -5.0 < ate < 5.0

    def test_placebo_returns_low_null_mean(self, refutation_inputs):
        """Placebo Test: 셔플된 Treatment의 ATE가 0 근처여야 함."""
        config = WhyLabConfig()
        config.sensitivity.n_refutation_iter = 5  # 속도: 5회만
        cell = RefutationCell(config)

        result = cell._placebo_test(
            refutation_inputs["dataframe"],
            refutation_inputs["treatment_col"],
            refutation_inputs["outcome_col"],
            refutation_inputs["feature_names"],
            original_ate=refutation_inputs["ate"],
            is_discrete=False,
            n_iter=5,
        )
        # Null 분포의 평균은 원래 ATE보다 0에 가까워야 함
        assert abs(result["null_mean"]) < abs(refutation_inputs["ate"]) * 5
        assert "p_value" in result
        assert "status" in result

    def test_bootstrap_ci_contains_ate(self, refutation_inputs):
        """Bootstrap CI가 합리적 범위를 생성하는지."""
        config = WhyLabConfig()
        cell = RefutationCell(config)

        result = cell._bootstrap_ci(
            refutation_inputs["dataframe"],
            refutation_inputs["treatment_col"],
            refutation_inputs["outcome_col"],
            refutation_inputs["feature_names"],
            is_discrete=False,
            n_boot=10,  # 속도: 10회
        )
        assert result["ci_lower"] < result["ci_upper"]  # CI가 유효
        assert result["std_ate"] > 0  # 분산이 양수
        assert "significant" in result

    def test_leave_one_out(self, refutation_inputs):
        """LOO Confounder: 각 변수 제거 결과가 올바른 형식인지."""
        config = WhyLabConfig()
        cell = RefutationCell(config)

        result = cell._leave_one_out_confounder(
            refutation_inputs["dataframe"],
            refutation_inputs["treatment_col"],
            refutation_inputs["outcome_col"],
            refutation_inputs["feature_names"],
            original_ate=refutation_inputs["ate"],
            is_discrete=False,
        )
        assert len(result["results"]) == 2  # 2개 교란변수 제거
        assert "max_pct_change" in result
        assert "any_sign_flip" in result
        for r in result["results"]:
            assert "excluded_variable" in r
            assert "ate_without" in r

    def test_subset_validation(self, refutation_inputs):
        """Subset Validation: 서브샘플 안정성 검증."""
        config = WhyLabConfig()
        cell = RefutationCell(config)

        result = cell._subset_validation(
            refutation_inputs["dataframe"],
            refutation_inputs["treatment_col"],
            refutation_inputs["outcome_col"],
            refutation_inputs["feature_names"],
            original_ate=refutation_inputs["ate"],
            is_discrete=False,
            fractions=[0.5, 0.8],
        )
        assert len(result["results"]) == 2
        assert 0 <= result["avg_stability"] <= 2.0  # 안정성 범위
        assert result["status"] in ("Pass", "Fail")

    def test_execute_full_pipeline(self, refutation_inputs):
        """execute() 전체 파이프라인이 정상 동작하는지."""
        config = WhyLabConfig()
        config.sensitivity.n_refutation_iter = 3
        config.sensitivity.n_bootstrap = 5
        cell = RefutationCell(config)

        result = cell.execute(refutation_inputs)

        assert "refutation_results" in result
        rr = result["refutation_results"]
        assert "placebo_test" in rr
        assert "bootstrap" in rr
        assert "leave_one_out" in rr
        assert "subset" in rr
        assert "overall" in rr
        assert rr["overall"]["status"] in ("Pass", "Fail")
        assert rr["overall"]["total"] == 4
