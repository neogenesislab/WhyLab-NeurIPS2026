# -*- coding: utf-8 -*-
"""감사 오케스트레이터 — E2E 파이프라인 통합.

Decision → GA4/Supabase → Match → Method Route → Audit → Drift → Damping → Feedback
전체 파이프라인을 단일 인터페이스로 통합합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from engine.audit.schemas import (
    AuditResult,
    DecisionEvent,
    DecisionOutcomePair,
    OutcomeEvent,
    OutcomeMetric,
)
from engine.audit.decision_logger import DecisionLogger
from engine.audit.matcher import DecisionOutcomeMatcher
from engine.audit.causal_auditor import CausalAuditor
from engine.audit.feedback_api import FeedbackAPI
from engine.audit.feedback_controller import FeedbackSignal
from engine.connectors.ga4_connector import GA4Connector

logger = logging.getLogger("whylab.audit.orchestrator")


class AuditOrchestrator:
    """Causal Audit E2E 오케스트레이터.

    사용법:
        orchestrator = AuditOrchestrator()

        # 1. 에이전트 결정 기록
        decision = orchestrator.log_decision(
            agent_name="hive_mind_toolpick",
            treatment="키워드 전략 변경",
            ...)

        # 2. 전체 파이프라인 실행
        signal = orchestrator.run_audit(decision)
        print(signal.action)  # reinforce / suppress / hold
    """

    def __init__(
        self,
        decision_logger: Optional[DecisionLogger] = None,
        ga4_connector: Optional[GA4Connector] = None,
        matcher: Optional[DecisionOutcomeMatcher] = None,
        auditor: Optional[CausalAuditor] = None,
        feedback_api: Optional[FeedbackAPI] = None,
    ) -> None:
        self.logger = decision_logger or DecisionLogger()
        self.ga4 = ga4_connector or GA4Connector()
        self.matcher = matcher or DecisionOutcomeMatcher()
        self.auditor = auditor or CausalAuditor()
        self.feedback = feedback_api or FeedbackAPI()

    def log_decision(self, **kwargs) -> DecisionEvent:
        """에이전트 결정을 기록합니다."""
        return self.logger.log_decision(**kwargs)

    def run_audit(
        self,
        decision: DecisionEvent,
        outcomes: Optional[List[OutcomeEvent]] = None,
    ) -> Optional[FeedbackSignal]:
        """단일 결정에 대한 전체 감사 파이프라인을 실행합니다.

        Args:
            decision: 감사 대상 결정
            outcomes: 결과 데이터 (없으면 GA4에서 수집)

        Returns:
            안정화된 FeedbackSignal (데이터 부족 시 None)
        """
        # 1. Outcome 데이터 수집
        if outcomes is None:
            from datetime import datetime, timedelta
            dt = datetime.fromisoformat(decision.timestamp)
            pre_start = (dt - timedelta(days=14)).strftime("%Y-%m-%d")
            post_end = (dt + timedelta(days=decision.observation_window_days)).strftime("%Y-%m-%d")

            outcomes = self.ga4.fetch_outcomes(
                metric=decision.target_metric,
                start_date=pre_start,
                end_date=post_end,
                sbu=decision.target_sbu,
            )

        # 2. 매칭
        pair = self.matcher.match_single(decision, outcomes)
        if pair is None:
            logger.warning("⚠️ 매칭 실패: decision=%s", decision.decision_id[:8])
            return None

        # 3. 감사
        result = self.auditor.audit(pair)

        # 4. 피드백 생성 (DriftMonitor → DampingController → Signal)
        data_density = min(len(pair.pre_outcomes) / 14, 1.0)
        signal = self.feedback.process_audit_result(
            agent_name=decision.agent_name,
            result=result,
            data_density=data_density,
        )

        logger.info(
            "✅ E2E 감사 완료: [%s] %s → %s → %s (weight=%.1f%%)",
            decision.decision_id[:8],
            decision.agent_name,
            result.verdict.value,
            signal.action,
            signal.effective_weight * 100,
        )

        return signal

    def run_pending_audits(self) -> List[FeedbackSignal]:
        """관측 기간이 지난 모든 대기 결정을 일괄 감사합니다."""
        pending = self.logger.get_pending_audits()
        signals = []

        for decision in pending:
            signal = self.run_audit(decision)
            if signal:
                signals.append(signal)

        logger.info("📋 일괄 감사 완료: %d/%d건 처리", len(signals), len(pending))
        return signals

    def get_status(self) -> Dict[str, Any]:
        """전체 시스템 상태."""
        return {
            "orchestrator": "active",
            "ga4_connected": self.ga4.is_connected,
            **self.feedback.get_system_status(),
        }
