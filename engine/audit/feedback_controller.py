# -*- coding: utf-8 -*-
"""적응형 감쇠 피드백 컨트롤러 — 에이전트 전략 업데이트 안정화.

인과 감사 결과를 에이전트 전략 메모리에 반영할 때,
제어 이론적(Control-theoretic) 감쇠 인자(Damping Factor)를 적용하여
정책 진동(Policy oscillation)과 과적합을 방지합니다.

핵심 원리:
    - 높은 신뢰도 + 낮은 드리프트 → ζ 상향 → 공격적 업데이트
    - 낮은 신뢰도 + 높은 드리프트 → ζ 하향 → 보수적 유지
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engine.audit.schemas import AuditResult, AuditVerdict

logger = logging.getLogger("whylab.audit.feedback")


@dataclass
class FeedbackSignal:
    """에이전트에게 전달될 필터링된 피드백 신호.

    Attributes:
        decision_id: 감사 대상 결정 ID
        agent_name: 에이전트 이름
        verdict: 감사 판결
        confidence: 감사 확신도
        damping_factor: 적용된 감쇠 인자 (0~1)
        effective_weight: 실제 반영 가중치 (confidence × damping)
        action: 권장 액션 (reinforce / suppress / hold)
        memo: 전략 메모리에 주입할 텍스트
    """

    decision_id: str
    agent_name: str
    verdict: AuditVerdict
    confidence: float
    damping_factor: float
    effective_weight: float
    action: str  # reinforce | suppress | hold
    memo: str


class DampingController:
    """적응형 감쇠 인자(Adaptive Damping Factor) 컨트롤러.

    감사 결과의 불확실성과 환경 드리프트에 비례하여
    피드백 반영 강도를 동적으로 조율합니다.

    Parameters:
        base_damping: 기본 감쇠 인자 (default: 0.3)
        min_damping: 최소 감쇠 인자 (극보수적 업데이트)
        max_damping: 최대 감쇠 인자 (공격적 업데이트)
        drift_threshold: 드리프트 임계값
        confidence_threshold: 신뢰도 임계값
    """

    def __init__(
        self,
        base_damping: float = 0.3,
        min_damping: float = 0.05,
        max_damping: float = 0.8,
        drift_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.base_damping = base_damping
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.drift_threshold = drift_threshold
        self.confidence_threshold = confidence_threshold
        self._history: List[Dict[str, Any]] = []

    def compute_damping(
        self,
        confidence: float,
        drift_index: float = 0.0,
        data_density: float = 1.0,
    ) -> float:
        """환경 상태에 따른 감쇠 인자를 계산합니다.

        Args:
            confidence: 감사 결과 확신도 (0~1)
            drift_index: 인과 드리프트 지수 (0~1, 높을수록 불안정)
            data_density: 데이터 밀도 (0~1, 낮을수록 희소)

        Returns:
            감쇠 인자 ζ (min_damping ~ max_damping)
        """
        # 기본 감쇠에서 출발
        zeta = self.base_damping

        # 신뢰도가 높으면 상향
        if confidence >= self.confidence_threshold:
            zeta += (confidence - self.confidence_threshold) * 0.5

        # 드리프트가 높으면 하향
        if drift_index > self.drift_threshold:
            zeta -= (drift_index - self.drift_threshold) * 0.8

        # 데이터 희소 시 하향
        if data_density < 0.5:
            zeta *= data_density + 0.5

        # 범위 제한
        zeta = max(self.min_damping, min(self.max_damping, zeta))

        logger.debug(
            "⚙️ Damping: ζ=%.3f (conf=%.2f, drift=%.2f, density=%.2f)",
            zeta, confidence, drift_index, data_density,
        )

        return round(zeta, 4)

    def generate_feedback(
        self,
        audit_result: AuditResult,
        drift_index: float = 0.0,
        data_density: float = 1.0,
    ) -> FeedbackSignal:
        """감사 결과를 안정화된 피드백 신호로 변환합니다.

        Args:
            audit_result: 인과 감사 결과
            drift_index: 현재 환경 드리프트 지수
            data_density: 데이터 밀도

        Returns:
            안정화된 FeedbackSignal
        """
        damping = self.compute_damping(
            confidence=audit_result.confidence,
            drift_index=drift_index,
            data_density=data_density,
        )

        effective_weight = audit_result.confidence * damping

        # 액션 결정
        if audit_result.verdict == AuditVerdict.CAUSAL and effective_weight > 0.3:
            action = "reinforce"
            memo = (
                f"[CAUSAL] 전략 유지/강화 권장. "
                f"ATE={audit_result.ate:+.4f}, "
                f"반영 가중치={effective_weight:.2%}"
            )
        elif audit_result.verdict == AuditVerdict.NOT_CAUSAL and effective_weight > 0.2:
            action = "suppress"
            memo = (
                f"[NOT_CAUSAL] 전략 억제/철회 권장. "
                f"ATE={audit_result.ate:+.4f}, "
                f"반영 가중치={effective_weight:.2%}"
            )
        else:
            action = "hold"
            memo = (
                f"[HOLD] 현 전략 유지. 추가 데이터 수집 후 재감사. "
                f"ζ={damping:.3f}, weight={effective_weight:.2%}"
            )

        signal = FeedbackSignal(
            decision_id=audit_result.decision_id,
            agent_name="",  # 호출자가 채움
            verdict=audit_result.verdict,
            confidence=audit_result.confidence,
            damping_factor=damping,
            effective_weight=effective_weight,
            action=action,
            memo=memo,
        )

        # 이력 기록
        self._history.append({
            "decision_id": audit_result.decision_id,
            "verdict": audit_result.verdict.value,
            "confidence": audit_result.confidence,
            "damping": damping,
            "effective_weight": effective_weight,
            "action": action,
        })

        logger.info(
            "📡 Feedback: [%s] %s → ζ=%.3f, weight=%.2f%%, action=%s",
            audit_result.decision_id[:8],
            audit_result.verdict.value,
            damping,
            effective_weight * 100,
            action,
        )

        return signal

    @property
    def history(self) -> List[Dict[str, Any]]:
        """피드백 이력."""
        return list(self._history)
