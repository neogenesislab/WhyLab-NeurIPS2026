# -*- coding: utf-8 -*-
"""피드백 API — 에이전트 전략 메모리 업데이트 인터페이스.

감사 결과를 에이전트에게 안전하게 전달하는 API 계층입니다.
DriftMonitor → DampingController → FeedbackAPI 순으로 체이닝되어
안정화된 피드백만 에이전트에게 도달합니다.

엔드포인트 (MCP Tool 확장 준비):
- push_feedback: 에이전트에게 피드백 주입
- get_history: 피드백 이력 조회
- get_scoreboard: 에이전트별 감사 성적표
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from engine.audit.schemas import AuditResult, AuditVerdict
from engine.audit.feedback_controller import DampingController, FeedbackSignal
from engine.audit.drift_monitor import CausalDriftMonitor

logger = logging.getLogger("whylab.audit.feedback_api")


class AgentScore:
    """에이전트별 누적 감사 성적."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.total_audits = 0
        self.causal_count = 0
        self.not_causal_count = 0
        self.uncertain_count = 0
        self.avg_confidence = 0.0
        self.avg_ate = 0.0
        self.total_effective_weight = 0.0
        self._confidences: List[float] = []
        self._ates: List[float] = []

    def update(self, result: AuditResult, signal: FeedbackSignal) -> None:
        self.total_audits += 1
        self._confidences.append(result.confidence)
        self._ates.append(result.ate)
        self.total_effective_weight += signal.effective_weight

        if result.verdict == AuditVerdict.CAUSAL:
            self.causal_count += 1
        elif result.verdict == AuditVerdict.NOT_CAUSAL:
            self.not_causal_count += 1
        else:
            self.uncertain_count += 1

        self.avg_confidence = sum(self._confidences) / len(self._confidences)
        self.avg_ate = sum(self._ates) / len(self._ates)

    @property
    def success_rate(self) -> float:
        """인과 성공률 (CAUSAL / total)."""
        return self.causal_count / max(self.total_audits, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_audits": self.total_audits,
            "causal": self.causal_count,
            "not_causal": self.not_causal_count,
            "uncertain": self.uncertain_count,
            "success_rate": round(self.success_rate, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_ate": round(self.avg_ate, 4),
            "total_effective_weight": round(self.total_effective_weight, 4),
        }


class FeedbackAPI:
    """에이전트 전략 메모리 업데이트 API.

    전체 체인: AuditResult → DriftMonitor → DampingController → FeedbackSignal → Agent
    """

    def __init__(
        self,
        damping_controller: Optional[DampingController] = None,
        drift_monitor: Optional[CausalDriftMonitor] = None,
    ) -> None:
        self._controller = damping_controller or DampingController()
        self._monitor = drift_monitor or CausalDriftMonitor()
        self._scores: Dict[str, AgentScore] = {}
        self._feedback_history: List[Dict[str, Any]] = []

    def process_audit_result(
        self,
        agent_name: str,
        result: AuditResult,
        data_density: float = 1.0,
    ) -> FeedbackSignal:
        """감사 결과를 안전한 피드백 신호로 변환합니다.

        1. DriftMonitor에 결과 기록 → DI 계산
        2. DampingController에 (결과, DI, density) 전달 → ζ 조절
        3. FeedbackSignal 생성 → 에이전트 성적표 업데이트

        Args:
            agent_name: 대상 에이전트 이름
            result: 인과 감사 결과
            data_density: 데이터 밀도 (0~1)

        Returns:
            안정화된 FeedbackSignal
        """
        # 1. 드리프트 계산
        drift_index = self._monitor.record(result)

        # 2. 감쇠 피드백 생성
        signal = self._controller.generate_feedback(
            audit_result=result,
            drift_index=drift_index,
            data_density=data_density,
        )
        signal.agent_name = agent_name

        # 3. 성적표 업데이트
        if agent_name not in self._scores:
            self._scores[agent_name] = AgentScore(agent_name)
        self._scores[agent_name].update(result, signal)

        # 4. 이력 기록
        self._feedback_history.append({
            "agent_name": agent_name,
            "decision_id": result.decision_id,
            "verdict": result.verdict.value,
            "action": signal.action,
            "damping": signal.damping_factor,
            "effective_weight": signal.effective_weight,
            "drift_index": drift_index,
        })

        logger.info(
            "📡 [%s] %s → %s (ζ=%.3f, weight=%.2f%%, DI=%.3f)",
            agent_name,
            result.verdict.value,
            signal.action,
            signal.damping_factor,
            signal.effective_weight * 100,
            drift_index,
        )

        return signal

    def get_agent_scoreboard(self) -> Dict[str, Dict[str, Any]]:
        """에이전트별 누적 감사 성적표."""
        return {
            name: score.to_dict()
            for name, score in sorted(
                self._scores.items(),
                key=lambda x: -x[1].success_rate,
            )
        }

    def get_feedback_history(
        self,
        agent_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """피드백 이력 조회."""
        history = self._feedback_history
        if agent_name:
            history = [h for h in history if h["agent_name"] == agent_name]
        return history[-limit:]

    def get_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태."""
        drift_status = self._monitor.get_status()
        return {
            "drift": drift_status,
            "agents_tracked": len(self._scores),
            "total_feedbacks": len(self._feedback_history),
            "scoreboard": self.get_agent_scoreboard(),
        }
