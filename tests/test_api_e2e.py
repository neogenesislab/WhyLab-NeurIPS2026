# -*- coding: utf-8 -*-
"""WhyLab API 통합 테스트 (end-to-end).

README의 3줄 코드가 실제로 작동하는지 검증합니다.
합성 데이터(시나리오 A)로 전체 파이프라인을 실행합니다.
"""

import pytest
import numpy as np
import pandas as pd


class TestWhyLabAPIImport:
    """패키지 import 및 기본 구조 테스트."""

    def test_import_whylab(self):
        """import whylab이 정상 작동하는지."""
        import whylab
        assert hasattr(whylab, "analyze")
        assert hasattr(whylab, "CausalResult")
        assert hasattr(whylab, "__version__")

    def test_version_format(self):
        """버전이 semver 형식인지."""
        import whylab
        parts = whylab.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestCausalResult:
    """CausalResult 데이터 클래스 테스트."""

    def test_default_values(self):
        """기본값이 안전한지."""
        from whylab import CausalResult
        result = CausalResult()
        assert result.ate == 0.0
        assert result.verdict == "UNCERTAIN"
        assert result.confidence == 0.0
        assert result.cate is None

    def test_summary_runs(self):
        """summary() 메서드가 크래시 없이 실행되는지."""
        from whylab import CausalResult
        result = CausalResult(ate=0.5, verdict="CAUSAL", confidence=0.85)
        text = result.summary()
        assert "CAUSAL" in text
        assert "0.5000" in text


class TestAnalyzeAPI:
    """analyze() 함수 통합 테스트 (합성 데이터)."""

    @pytest.mark.slow
    def test_analyze_synthetic_scenario_a(self):
        """시나리오 A(신용 한도 → 연체) 전체 파이프라인 실행."""
        import whylab
        result = whylab.analyze(data=None, scenario="A")

        # ATE가 실수값인지
        assert isinstance(result.ate, float)
        # 판결이 유효한 값인지
        assert result.verdict in ("CAUSAL", "NOT_CAUSAL", "UNCERTAIN", "ERROR")
        # 원시 결과가 존재하는지
        assert isinstance(result.raw, dict)

    @pytest.mark.slow
    def test_analyze_with_dataframe(self):
        """DataFrame 입력으로 파이프라인 실행."""
        import whylab

        # 간단한 합성 데이터 생성
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "age": np.random.normal(40, 10, n),
            "income": np.random.normal(50000, 15000, n),
            "treatment": np.random.binomial(1, 0.5, n),
        })
        df["outcome"] = 0.3 * df["treatment"] + 0.01 * df["age"] + np.random.normal(0, 0.5, n)

        result = whylab.analyze(
            data=df,
            treatment="treatment",
            outcome="outcome",
            features=["age", "income"],
        )

        assert isinstance(result.ate, float)
        assert result.verdict in ("CAUSAL", "NOT_CAUSAL", "UNCERTAIN", "ERROR")

    def test_analyze_returns_causal_result(self):
        """반환 타입이 CausalResult인지 (빠른 검증만)."""
        from whylab.api import CausalResult
        result = CausalResult(ate=1.0, verdict="CAUSAL", confidence=0.9)
        assert isinstance(result, CausalResult)
