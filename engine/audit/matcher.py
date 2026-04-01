# -*- coding: utf-8 -*-
"""Decision-Outcome 매처 — 결정과 결과를 시계열로 매칭.

에이전트 결정 로그와 GA4/PostHog 결과 데이터를 시간 기반으로 매칭하여
인과 감사 파이프라인에 투입 가능한 DecisionOutcomePair를 생성합니다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from engine.audit.schemas import (
    DecisionEvent,
    DecisionOutcomePair,
    OutcomeEvent,
)

logger = logging.getLogger("whylab.audit.matcher")

# 기본 사전 관측 기간 (결정 전 N일)
DEFAULT_PRE_WINDOW_DAYS = 14


class DecisionOutcomeMatcher:
    """결정 이벤트와 관측 결과를 시계열 기반으로 매칭합니다.

    매칭 로직:
        Decision(timestamp=T, observation_window_days=W)
        → pre_outcomes: [T - PRE_WINDOW, T)
        → post_outcomes: [T, T + W]
    """

    def __init__(self, pre_window_days: int = DEFAULT_PRE_WINDOW_DAYS) -> None:
        self.pre_window_days = pre_window_days

    def match(
        self,
        decisions: List[DecisionEvent],
        outcomes: List[OutcomeEvent],
    ) -> List[DecisionOutcomePair]:
        """결정 리스트와 결과 리스트를 매칭합니다.

        Args:
            decisions: 에이전트 결정 이벤트 리스트
            outcomes: 관측된 결과 이벤트 리스트

        Returns:
            매칭된 DecisionOutcomePair 리스트
        """
        pairs = []

        for decision in decisions:
            # 동일 SBU + 동일 지표만 매칭
            relevant_outcomes = [
                o for o in outcomes
                if o.sbu == decision.target_sbu
                and o.metric == decision.target_metric
            ]

            if not relevant_outcomes:
                logger.debug(
                    "⚠️ 매칭 데이터 없음: %s → %s/%s",
                    decision.decision_id[:8],
                    decision.target_sbu,
                    decision.target_metric.value,
                )
                continue

            decision_time = datetime.fromisoformat(decision.timestamp)
            pre_start = decision_time - timedelta(days=self.pre_window_days)
            post_end = decision_time + timedelta(days=decision.observation_window_days)

            pre_outcomes = [
                o for o in relevant_outcomes
                if pre_start <= datetime.fromisoformat(o.timestamp) < decision_time
            ]
            post_outcomes = [
                o for o in relevant_outcomes
                if decision_time <= datetime.fromisoformat(o.timestamp) <= post_end
            ]

            pair = DecisionOutcomePair(
                decision=decision,
                pre_outcomes=sorted(pre_outcomes, key=lambda x: x.timestamp),
                post_outcomes=sorted(post_outcomes, key=lambda x: x.timestamp),
            )

            logger.info(
                "🔗 매칭 완료: [%s] pre=%d, post=%d, ready=%s",
                decision.decision_id[:8],
                len(pre_outcomes),
                len(post_outcomes),
                pair.is_ready_for_audit,
            )

            pairs.append(pair)

        return pairs

    def match_single(
        self,
        decision: DecisionEvent,
        outcomes: List[OutcomeEvent],
    ) -> Optional[DecisionOutcomePair]:
        """단일 결정에 대한 매칭."""
        results = self.match([decision], outcomes)
        return results[0] if results else None
