# -*- coding: utf-8 -*-
"""CausalCell 단위 테스트.

LinearDML 및 CausalForestDML 모델 학습과 ATE/CATE 추정을 검증합니다.
"""

import pytest
import numpy as np
import pandas as pd
from engine.config import WhyLabConfig
from engine.cells.data_cell import DataCell
from engine.cells.causal_cell import CausalCell


@pytest.fixture
def config():
    cfg = WhyLabConfig()
    cfg.data.n_samples = 500  # 테스트 속도를 위해 샘플 축소
    cfg.dml.cv_folds = 2      # CV 폴드 축소
    cfg.dml.lgbm_n_estimators = 10 # 학습 속도 최적화
    return cfg


def test_causal_cell_scenario_a(config):
    """시나리오 A(연속형 처치) DML 학습 테스트."""
    # 1. 데이터 준비
    data_cell = DataCell(config)
    data_out = data_cell.run({"scenario": "A"})
    
    # 2. Causal 모델 학습
    causal_cell = CausalCell(config)
    causal_out = causal_cell.run(data_out)
    
    # 3. 결과 검증
    assert "ate" in causal_out
    assert "cate_predictions" in causal_out
    assert len(causal_out["cate_predictions"]) == 500
    assert isinstance(causal_out["ate"], float)
    
    # 95% 신뢰구간 존재 확인
    assert "ate_ci_lower" in causal_out
    assert "ate_ci_upper" in causal_out
    assert causal_out["ate_ci_lower"] <= causal_out["ate"] <= causal_out["ate_ci_upper"]


def test_causal_cell_scenario_b(config):
    """시나리오 B(이진형 처치) DML 학습 테스트."""
    # 1. 데이터 준비
    data_cell = DataCell(config)
    data_out = data_cell.run({"scenario": "B"}) # scenario B 명시
    
    # 2. Causal 모델 학습
    causal_cell = CausalCell(config)
    # DataCell 출력에는 "scenario" 키가 없으므로 직접 주입 필요할 수도 있으나
    # Orchestrator는 context를 병합하므로 여기서는 수동 병합 시뮬레이션
    inputs = {**data_out, "scenario": "B"}
    causal_out = causal_cell.run(inputs)
    
    # 3. 결과 검증
    assert "ate" in causal_out
    # 이진 처치도 ATE 계산됨
    assert isinstance(causal_out["ate"], float)
