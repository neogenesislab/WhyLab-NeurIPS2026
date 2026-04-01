# -*- coding: utf-8 -*-
"""경량 인과 분석 메서드 — stdlib 전용.

기존 CausalAuditor 내장 분석 로직을 BaseMethod 인터페이스로 분리.
외부 의존성 없이 Welch's t-test, Cohen's d, Placebo test를 수행합니다.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List

from engine.audit.methods.base import AnalysisResult, BaseMethod


class LightweightMethod(BaseMethod):
    """경량 인과 분석 (Phase 1 기본).

    방법:
    1. Pre/Post 평균 차이 (ATE 추정)
    2. Welch's t-test (유의성)
    3. Cohen's d (효과 크기)
    4. Placebo test (pre 기간 분할 검증)
    """

    METHOD_NAME = "lightweight_t_test"
    REQUIRES = []  # stdlib만 사용

    def analyze(
        self,
        pre: List[float],
        post: List[float],
        significance_level: float = 0.05,
        **kwargs,
    ) -> AnalysisResult:
        """경량 인과 분석을 실행합니다."""
        pre_mean = statistics.mean(pre)
        post_mean = statistics.mean(post)
        ate = post_mean - pre_mean

        pre_std = statistics.stdev(pre) if len(pre) > 1 else 1e-10
        post_std = statistics.stdev(post) if len(post) > 1 else 1e-10
        n_pre, n_post = len(pre), len(post)

        # Welch's t-test
        se = (pre_std**2 / n_pre + post_std**2 / n_post) ** 0.5
        t_stat = ate / se if se > 1e-10 else 0.0

        # p-value (정규 근사)
        z = abs(t_stat)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

        # Cohen's d
        pooled_std = ((pre_std**2 + post_std**2) / 2) ** 0.5
        cohens_d = ate / pooled_std if pooled_std > 1e-10 else 0.0

        # 95% CI
        margin = 1.96 * se
        ate_ci = [ate - margin, ate + margin]

        # Placebo test
        placebo_passed = True
        if len(pre) >= 6:
            mid = len(pre) // 2
            placebo_ate = statistics.mean(pre[mid:]) - statistics.mean(pre[:mid])
            placebo_passed = abs(placebo_ate) < abs(ate) * 0.5

        # Confidence score
        confidence = 0.0
        if p_value < significance_level:
            confidence += 0.4
        if abs(cohens_d) > 0.3:
            confidence += 0.2
        if placebo_passed:
            confidence += 0.2
        if ate_ci[0] > 0 or ate_ci[1] < 0:
            confidence += 0.2

        return AnalysisResult(
            method=self.METHOD_NAME,
            ate=round(ate, 4),
            ate_ci=[round(x, 4) for x in ate_ci],
            p_value=round(p_value, 6),
            confidence=round(min(confidence, 1.0), 2),
            effect_size=round(cohens_d, 4),
            placebo_passed=placebo_passed,
            diagnostics={
                "t_statistic": round(t_stat, 4),
                "pre_mean": round(pre_mean, 4),
                "post_mean": round(post_mean, 4),
                "pre_std": round(pre_std, 4),
                "post_std": round(post_std, 4),
                "n_pre": n_pre,
                "n_post": n_post,
            },
        )
