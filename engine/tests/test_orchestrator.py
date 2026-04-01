# -*- coding: utf-8 -*-
"""Orchestrator 통합 테스트.

전체 파이프라인(Data -> Causal -> Explain -> Viz -> Export)의
End-to-End 실행을 검증합니다.
"""

import pytest
from pathlib import Path
from engine.orchestrator import Orchestrator
from engine.config import WhyLabConfig


@pytest.fixture
def config_override():
    """테스트용으로 설정을 오버라이드합니다."""
    # 실제 Config 객체를 수정하는 대신, 
    # Orchestrator 내부에서 사용하는 Config를 mocking하거나
    # 테스트 환경에서는 파일 생성을 최소화해야 함.
    # 여기서는 샘플 수만 줄여서 실제 실행 테스트.
    cfg = WhyLabConfig()
    cfg.data.n_samples = 200
    cfg.dml.cv_folds = 2
    cfg.dml.lgbm_n_estimators = 5
    cfg.explain.shap_sample_size = 10
    cfg.viz.max_scatter_points = 50
    return cfg


def test_orchestrator_pipeline_a(config_override, tmp_path):
    """시나리오 A 파이프라인 통합 테스트."""
    orc = Orchestrator()
    # Config 주입 (DataCell 등 내부 셀들이 참조하는 Config를 교체)
    # 현재 설계상 Orchestrator 생성자에서 Config를 생성하므로
    # 테스트용 Config로 교체하는 로직이 필요함.
    # Python은 동적 언어이므로 속성 할당으로 교체 가능.
    orc.config = config_override
    orc.config.paths.project_root = tmp_path # 임시 경로 사용
    
    # 각 셀에도 새 설정 전파
    for cell in orc.cells.values():
        cell.config = config_override

    # 실행
    result = orc.run_pipeline(scenario="A")
    
    # 검증
    assert "ate" in result
    assert "json_path" in result
    assert result["scenario"] == "A"
    
    # 파일 생성 확인
    json_path = Path(result["json_path"])
    assert json_path.exists()


def test_orchestrator_pipeline_b(config_override, tmp_path):
    """시나리오 B 파이프라인 통합 테스트."""
    orc = Orchestrator()
    orc.config = config_override
    orc.config.paths.project_root = tmp_path
    
    for cell in orc.cells.values():
        cell.config = config_override

    result = orc.run_pipeline(scenario="B")
    
    assert "ate" in result
    assert result["scenario"] == "B"
    assert "saved_figures" in result
    assert len(result["saved_figures"]) > 0
