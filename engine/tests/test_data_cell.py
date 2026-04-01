# -*- coding: utf-8 -*-
"""DataCell 단위 테스트.

시나리오 A(신용한도)와 B(마케팅) 데이터 생성 및 형식을 검증합니다.
"""

import pytest
import pandas as pd
from engine.config import WhyLabConfig
from engine.cells.data_cell import DataCell


@pytest.fixture
def config():
    cfg = WhyLabConfig()
    cfg.data.n_samples = 1000  # 테스트용 샘플 수 축소
    return cfg


def test_scenario_a_generation(config):
    """시나리오 A(신용 한도) 데이터 생성 테스트."""
    cell = DataCell(config)
    output = cell.run({"scenario": "A"})
    
    df = output["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "credit_limit" in df.columns
    assert "is_default" in df.columns
    assert output["treatment_col"] == "credit_limit"
    assert output["outcome_col"] == "is_default"
    
    # DuckDB 전처리 컬럼 확인
    assert "avg_spend_3m" in df.columns


def test_scenario_b_generation(config):
    """시나리오 B(마케팅 쿠폰) 데이터 생성 테스트."""
    cell = DataCell(config)
    output = cell.run({"scenario": "B"})
    
    df = output["dataframe"]
    assert "coupon_sent" in df.columns  # 이진 처치
    assert "is_joined" in df.columns    # 이진 결과
    assert output["treatment_col"] == "coupon_sent"
    assert output["outcome_col"] == "is_joined"
    
    # 이진 처치값 확인
    assert set(df["coupon_sent"].unique()) <= {0, 1}
