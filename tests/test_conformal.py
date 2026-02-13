# -*- coding: utf-8 -*-
"""ConformalCell 테스트 — 분포무가정 CATE 예측구간 검증."""

import numpy as np
import pytest

from engine.config import WhyLabConfig
from engine.cells.conformal_cell import ConformalCell


@pytest.fixture
def conformal_inputs():
    """ConformalCell 테스트용 합성 데이터."""
    np.random.seed(42)
    n = 600

    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    T = 0.5 * X1 - 0.3 * X2 + np.random.normal(0, 0.5, n)
    true_cate = 0.3 * X1 - 0.2 * X2
    Y = true_cate * T + 0.5 * X1 - 0.4 * X2 + np.random.normal(0, 0.3, n)

    import pandas as pd
    df = pd.DataFrame({
        "income": X1, "age": X2,
        "credit_limit": T, "default": Y,
    })

    return {
        "dataframe": df,
        "feature_names": ["income", "age"],
        "treatment_col": "credit_limit",
        "outcome_col": "default",
    }


class TestConformalCell:
    """ConformalCell 테스트."""

    def test_execute_returns_ci(self, conformal_inputs):
        """execute()가 예측구간을 반환하는지."""
        config = WhyLabConfig()
        cell = ConformalCell(config)
        result = cell.execute(conformal_inputs)

        assert "conformal_ci_lower" in result
        assert "conformal_ci_upper" in result
        assert "conformal_cate" in result
        assert "conformal_results" in result

        n = len(conformal_inputs["dataframe"])
        assert len(result["conformal_ci_lower"]) == n
        assert len(result["conformal_ci_upper"]) == n

    def test_ci_lower_less_than_upper(self, conformal_inputs):
        """CI 하한 < 상한인지."""
        config = WhyLabConfig()
        cell = ConformalCell(config)
        result = cell.execute(conformal_inputs)

        assert np.all(result["conformal_ci_lower"] <= result["conformal_ci_upper"])

    def test_quantile_positive(self, conformal_inputs):
        """Quantile q가 양수인지."""
        config = WhyLabConfig()
        cell = ConformalCell(config)
        result = cell.execute(conformal_inputs)

        assert result["conformal_results"]["quantile_q"] > 0

    def test_coverage_reasonable(self, conformal_inputs):
        """적중률이 합리적 범위 (50%~100%)인지."""
        config = WhyLabConfig()
        cell = ConformalCell(config)
        result = cell.execute(conformal_inputs)

        coverage = result["conformal_results"]["coverage"]
        assert 0.5 <= coverage <= 1.0, f"Coverage={coverage} out of range"
