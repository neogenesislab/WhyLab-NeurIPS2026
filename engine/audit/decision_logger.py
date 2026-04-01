# -*- coding: utf-8 -*-
"""에이전트 결정 로거 — Decision Event 기록 및 저장.

에이전트가 결정을 내릴 때 호출하여 표준 형식으로 기록합니다.
추후 WhyLab 인과 감사 파이프라인에서 이 로그를 읽어 감사합니다.

사용법:
    from engine.audit.decision_logger import DecisionLogger

    logger = DecisionLogger()
    logger.log_decision(
        agent_type=AgentType.HIVE_MIND,
        agent_name="hive_mind_toolpick",
        decision_type=DecisionType.CONTENT_STRATEGY,
        treatment="키워드 전략을 AI Tools로 변경",
        target_sbu="toolpick",
        target_metric=OutcomeMetric.ORGANIC_TRAFFIC,
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.audit.schemas import (
    AgentType,
    DecisionEvent,
    DecisionType,
    OutcomeMetric,
)

logger = logging.getLogger("whylab.audit.decision_logger")

# 기본 저장 경로 (환경변수로 오버라이드 가능)
DEFAULT_LOG_DIR = os.environ.get("WHYLAB_DECISION_LOG_DIR", "logs/decisions")


class DecisionLogger:
    """에이전트 결정을 기록하고 조회하는 로거.

    결정 이벤트를 JSONL 파일에 append 방식으로 저장합니다.
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self._decisions: List[DecisionEvent] = []

    def log_decision(
        self,
        agent_type: AgentType,
        agent_name: str,
        decision_type: DecisionType,
        treatment: str,
        target_sbu: str,
        target_metric: OutcomeMetric,
        treatment_value: Any = None,
        context: Optional[Dict[str, Any]] = None,
        expected_effect: str = "positive",
        observation_window_days: int = 7,
    ) -> DecisionEvent:
        """결정 이벤트를 기록합니다.

        Args:
            agent_type: 에이전트 유형
            agent_name: 에이전트 이름
            decision_type: 결정 유형
            treatment: 처치 변수 설명
            target_sbu: 대상 SBU
            target_metric: 기대 영향 지표
            treatment_value: 처치 변수 값
            context: 추가 맥락
            expected_effect: 기대 효과 방향
            observation_window_days: 관측 기간

        Returns:
            기록된 DecisionEvent
        """
        decision = DecisionEvent(
            agent_type=agent_type,
            agent_name=agent_name,
            decision_type=decision_type,
            treatment=treatment,
            target_sbu=target_sbu,
            target_metric=target_metric,
            treatment_value=treatment_value,
            context=context or {},
            expected_effect=expected_effect,
            observation_window_days=observation_window_days,
        )

        self._decisions.append(decision)
        self._persist(decision)

        logger.info(
            "📝 Decision logged: [%s] %s → %s (SBU: %s, window: %dd)",
            decision.decision_id[:8],
            agent_name,
            treatment[:50],
            target_sbu,
            observation_window_days,
        )

        return decision

    def get_decisions(
        self,
        agent_type: Optional[AgentType] = None,
        sbu: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[DecisionEvent]:
        """저장된 결정 이벤트를 조회합니다.

        Args:
            agent_type: 필터링할 에이전트 유형
            sbu: 필터링할 SBU
            since: 이 시점 이후의 결정만 (ISO 형식)

        Returns:
            필터링된 DecisionEvent 리스트
        """
        decisions = self._load_all()

        if agent_type:
            decisions = [d for d in decisions if d.agent_type == agent_type]
        if sbu:
            decisions = [d for d in decisions if d.target_sbu == sbu]
        if since:
            decisions = [d for d in decisions if d.timestamp >= since]

        return decisions

    def get_pending_audits(self) -> List[DecisionEvent]:
        """감사 대기 중인 결정(관측 기간이 지난 것)을 반환합니다."""
        now = datetime.now(timezone.utc)
        pending = []

        for d in self._load_all():
            from datetime import timedelta
            decision_time = datetime.fromisoformat(d.timestamp)
            window_end = decision_time + timedelta(days=d.observation_window_days)
            if now >= window_end:
                pending.append(d)

        return pending

    # ── 저장/로드 ──

    def _persist(self, decision: DecisionEvent) -> None:
        """JSONL 파일에 결정 이벤트를 append합니다."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            filepath = self._log_dir / "decisions.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("⚠️ Decision 저장 실패: %s", e)

    def _load_all(self) -> List[DecisionEvent]:
        """JSONL 파일에서 모든 결정 이벤트를 로드합니다."""
        filepath = self._log_dir / "decisions.jsonl"
        if not filepath.exists():
            return list(self._decisions)

        decisions = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    decisions.append(DecisionEvent(
                        decision_id=d["decision_id"],
                        agent_type=AgentType(d["agent_type"]),
                        agent_name=d["agent_name"],
                        decision_type=DecisionType(d["decision_type"]),
                        treatment=d["treatment"],
                        target_sbu=d["target_sbu"],
                        target_metric=OutcomeMetric(d["target_metric"]),
                        treatment_value=d.get("treatment_value"),
                        timestamp=d["timestamp"],
                        context=d.get("context", {}),
                        expected_effect=d.get("expected_effect", "positive"),
                        observation_window_days=d.get("observation_window_days", 7),
                    ))
        except Exception as e:
            logger.warning("⚠️ Decision 로드 실패: %s", e)

        return decisions
