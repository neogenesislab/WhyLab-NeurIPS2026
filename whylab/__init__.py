# -*- coding: utf-8 -*-
"""WhyLab: AI-Driven Causal Inference Engine.

3줄 코드로 인과 분석을 수행할 수 있는 간편 API를 제공합니다.

사용법:
    import whylab
    result = whylab.analyze("data.csv", treatment="T", outcome="Y")
    result.summary()
"""

from whylab.api import analyze, CausalResult

__version__ = "0.1.0"
__all__ = ["analyze", "CausalResult", "__version__"]
