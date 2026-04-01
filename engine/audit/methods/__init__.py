# -*- coding: utf-8 -*-
"""인과 추론 메서드 패키지.

CausalAuditor의 method_router가 선택하는 분석 메서드 모듈을 제공합니다.

지원 메서드:
- lightweight: Welch t-test + Cohen's d + Placebo (Phase 1, stdlib만 사용)
- causal_impact: CausalImpact BSTS (데이터 풍부 시)
- gsc: 일반화된 합성 대조군 (데이터 희소 시)
- dml: 이중 기계 학습 Multi-Treatment (다중 동시 처치)
- blame: MACIE Blame Attribution (에이전트별 책임 분배)
"""

from engine.audit.methods.base import AnalysisResult, BaseMethod

__all__ = ["AnalysisResult", "BaseMethod"]
