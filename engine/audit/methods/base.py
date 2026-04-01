# -*- coding: utf-8 -*-
"""인과 추론 메서드 베이스 클래스 및 공통 결과 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisResult:
    """메서드별 표준 분석 결과.

    모든 메서드가 이 형식으로 결과를 반환하여
    CausalAuditor가 일관되게 처리할 수 있도록 합니다.
    """

    method: str
    ate: float = 0.0
    ate_ci: List[float] = field(default_factory=lambda: [0.0, 0.0])
    p_value: Optional[float] = None
    confidence: float = 0.0
    effect_size: float = 0.0  # Cohen's d 등
    placebo_passed: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    sensitivity: Dict[str, Any] = field(default_factory=dict)  # R2: E-value 민감도

    @property
    def is_significant(self) -> bool:
        """p < 0.05 유의성 판단."""
        return self.p_value is not None and self.p_value < 0.05


class BaseMethod:
    """인과 추론 메서드의 추상 베이스 클래스."""

    METHOD_NAME: str = "base"
    REQUIRES: List[str] = []  # 필요한 외부 패키지

    def analyze(self, pre: List[float], post: List[float], **kwargs) -> AnalysisResult:
        """분석 실행. 서브클래스에서 반드시 구현해야 합니다."""
        raise NotImplementedError

    def check_requirements(self) -> bool:
        """필요한 패키지가 설치되어 있는지 확인."""
        for pkg in self.REQUIRES:
            try:
                __import__(pkg)
            except ImportError:
                return False
        return True

    @property
    def is_available(self) -> bool:
        """이 메서드를 사용할 수 있는지."""
        return self.check_requirements()
