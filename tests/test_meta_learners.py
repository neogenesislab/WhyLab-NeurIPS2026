# -*- coding: utf-8 -*-
"""MetaLearnerCell 테스트 — 5종 메타러너 + Oracle 선택 검증."""

import numpy as np
import pytest

from engine.config import WhyLabConfig
from engine.cells.meta_learner_cell import (
    MetaLearnerCell, SLearner, TLearner, XLearner, DRLearner, RLearner,
)


@pytest.fixture
def meta_inputs():
    """메타러너 테스트용 합성 인과 데이터."""
    np.random.seed(42)
    n = 800

    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    T = 0.5 * X1 - 0.3 * X2 + np.random.normal(0, 0.5, n)
    true_cate = 0.3 * X1 - 0.2 * X2
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
    }


class TestIndividualLearners:
    """개별 메타러너 단위 테스트."""

    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        n = 300
        X = np.random.normal(0, 1, (n, 2))
        T = 0.5 * X[:, 0] + np.random.normal(0, 0.5, n)
        Y = 0.3 * X[:, 0] * T + np.random.normal(0, 0.3, n)
        return X, T, Y

    def test_s_learner(self, simple_data):
        X, T, Y = simple_data
        learner = SLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert cate.shape == (len(X),)
        assert not np.all(cate == 0)  # 상수가 아님

    def test_t_learner(self, simple_data):
        X, T, Y = simple_data
        learner = TLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert cate.shape == (len(X),)

    def test_x_learner(self, simple_data):
        X, T, Y = simple_data
        learner = XLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert cate.shape == (len(X),)

    def test_dr_learner(self, simple_data):
        X, T, Y = simple_data
        learner = DRLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert cate.shape == (len(X),)

    def test_r_learner(self, simple_data):
        X, T, Y = simple_data
        learner = RLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert cate.shape == (len(X),)


class TestMetaLearnerCell:
    """MetaLearnerCell 통합 테스트."""

    def test_execute_returns_all_learners(self, meta_inputs):
        """execute()가 5종 메타러너 결과를 모두 반환하는지."""
        config = WhyLabConfig()
        config.dml.cv_folds = 2  # 속도
        cell = MetaLearnerCell(config)

        result = cell.execute(meta_inputs)

        assert "meta_learner_results" in result
        mlr = result["meta_learner_results"]
        assert "learners" in mlr
        assert len(mlr["learners"]) == 5
        for name in ["S-Learner", "T-Learner", "X-Learner", "DR-Learner", "R-Learner"]:
            assert name in mlr["learners"]

    def test_oracle_selection(self, meta_inputs):
        """Oracle이 CV-MSE 최소 모델을 선택하는지."""
        config = WhyLabConfig()
        config.dml.cv_folds = 2
        cell = MetaLearnerCell(config)

        result = cell.execute(meta_inputs)
        mlr = result["meta_learner_results"]

        assert "oracle" in mlr
        best = mlr["oracle"]["best_learner"]
        assert best in mlr["learners"]
        # 선택된 모델의 CV-MSE가 가장 낮은지 확인
        best_mse = mlr["oracle"]["cv_mse"]
        for name, lr in mlr["learners"].items():
            assert best_mse <= lr["cv_mse"] + 1e-10, f"{best} > {name}"

    def test_ensemble_output(self, meta_inputs):
        """앙상블 CATE가 올바른 형태인지."""
        config = WhyLabConfig()
        config.dml.cv_folds = 2
        cell = MetaLearnerCell(config)

        result = cell.execute(meta_inputs)

        assert "ensemble_cate" in result
        assert "ensemble_ate" in result
        assert len(result["ensemble_cate"]) == len(meta_inputs["dataframe"])
        # 가중치 합 = 1
        weights = result["meta_learner_results"]["ensemble"]["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_consensus(self, meta_inputs):
        """합의율이 0~1 사이인지."""
        config = WhyLabConfig()
        config.dml.cv_folds = 2
        cell = MetaLearnerCell(config)

        result = cell.execute(meta_inputs)
        consensus = result["meta_learner_results"]["ensemble"]["consensus"]
        assert 0 <= consensus <= 1.0
