# -*- coding: utf-8 -*-
"""Supabase Audit 커넥터 — Decision/Outcome/AuditResult 영속화.

로컬 JSONL과 병행하여 Supabase에 감사 데이터를 저장합니다.
Supabase 미연결 시 로컬 전용 모드로 자동 전환됩니다.

환경변수:
    SUPABASE_URL: Supabase 프로젝트 URL
    SUPABASE_KEY: Supabase 서비스 키 (anon 또는 service_role)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.audit.schemas import (
    AuditResult,
    DecisionEvent,
    OutcomeEvent,
    OutcomeMetric,
)

logger = logging.getLogger("whylab.connectors.supabase")


class SupabaseAuditConnector:
    """Supabase 기반 감사 데이터 영속화 커넥터.

    스키마 최적화 (리서치 기반):
    - decisions 테이블: (agent_type, decision_type) 복합 인덱스
    - outcomes 테이블: timestamp DESC 인덱스, (sbu, metric) 복합 인덱스
    - audit_results 테이블: decision_id FK, verdict 인덱스
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        self._url = url or os.environ.get("SUPABASE_URL", "")
        self._key = key or os.environ.get("SUPABASE_KEY", "")
        self._client = None
        self._connected = False

    def _ensure_client(self) -> bool:
        """Supabase 클라이언트를 지연 초기화합니다."""
        if self._client is not None:
            return self._connected

        if not self._url or not self._key:
            logger.info("📋 Supabase 미설정 → 로컬 전용 모드")
            return False

        try:
            from supabase import create_client
            self._client = create_client(self._url, self._key)
            self._connected = True
            logger.info("✅ Supabase 연결 완료: %s", self._url[:30])
            return True
        except ImportError:
            logger.warning("⚠️ supabase 패키지 미설치. pip install supabase 필요")
            return False
        except Exception as e:
            logger.warning("⚠️ Supabase 연결 실패: %s", e)
            return False

    # ── Decision CRUD ──

    def save_decision(self, event: DecisionEvent) -> Optional[str]:
        """Decision 이벤트를 저장합니다."""
        if not self._ensure_client():
            return None

        try:
            data = {
                "decision_id": event.decision_id,
                "agent_type": event.agent_type.value,
                "agent_name": event.agent_name,
                "decision_type": event.decision_type.value,
                "treatment": event.treatment,
                "target_sbu": event.target_sbu,
                "target_metric": event.target_metric.value,
                "treatment_value": event.treatment_value,
                "context": event.context,
                "expected_effect": event.expected_effect,
                "observation_window_days": event.observation_window_days,
                "created_at": event.timestamp,
            }
            result = self._client.table("audit_decisions").insert(data).execute()
            logger.debug("💾 Decision 저장: %s", event.decision_id[:8])
            return event.decision_id
        except Exception as e:
            logger.warning("⚠️ Decision 저장 실패: %s", e)
            return None

    # ── Outcome CRUD ──

    def save_outcome(self, event: OutcomeEvent) -> Optional[str]:
        """Outcome 이벤트를 저장합니다."""
        if not self._ensure_client():
            return None

        try:
            data = {
                "outcome_id": event.outcome_id,
                "metric": event.metric.value,
                "value": event.value,
                "sbu": event.sbu,
                "source": event.source,
                "period": event.period,
                "metadata": event.metadata,
                "observed_at": event.timestamp,
            }
            result = self._client.table("audit_outcomes").insert(data).execute()
            return event.outcome_id
        except Exception as e:
            logger.warning("⚠️ Outcome 저장 실패: %s", e)
            return None

    def save_outcomes_batch(self, events: List[OutcomeEvent]) -> int:
        """Outcome 이벤트를 일괄 저장합니다."""
        if not self._ensure_client():
            return 0

        try:
            data = [
                {
                    "outcome_id": e.outcome_id,
                    "metric": e.metric.value,
                    "value": e.value,
                    "sbu": e.sbu,
                    "source": e.source,
                    "period": e.period,
                    "metadata": e.metadata,
                    "observed_at": e.timestamp,
                }
                for e in events
            ]
            result = self._client.table("audit_outcomes").insert(data).execute()
            logger.info("💾 Outcomes 일괄 저장: %d건", len(events))
            return len(events)
        except Exception as e:
            logger.warning("⚠️ Outcomes 일괄 저장 실패: %s", e)
            return 0

    def query_outcomes(
        self,
        sbu: str,
        metric: OutcomeMetric,
        start_date: str,
        end_date: str,
    ) -> List[OutcomeEvent]:
        """기간별 Outcome 데이터를 조회합니다."""
        if not self._ensure_client():
            return []

        try:
            result = (
                self._client.table("audit_outcomes")
                .select("*")
                .eq("sbu", sbu)
                .eq("metric", metric.value)
                .gte("observed_at", start_date)
                .lte("observed_at", end_date)
                .order("observed_at")
                .execute()
            )
            return [
                OutcomeEvent(
                    outcome_id=r["outcome_id"],
                    metric=OutcomeMetric(r["metric"]),
                    value=r["value"],
                    sbu=r["sbu"],
                    timestamp=r["observed_at"],
                    source=r.get("source", "supabase"),
                    period=r.get("period", "daily"),
                    metadata=r.get("metadata", {}),
                )
                for r in result.data
            ]
        except Exception as e:
            logger.warning("⚠️ Outcomes 조회 실패: %s", e)
            return []

    # ── AuditResult CRUD ──

    def save_audit_result(self, result: AuditResult) -> Optional[str]:
        """감사 결과를 저장합니다."""
        if not self._ensure_client():
            return None

        try:
            data = {
                "audit_id": result.audit_id,
                "decision_id": result.decision_id,
                "verdict": result.verdict.value,
                "confidence": result.confidence,
                "ate": result.ate,
                "ate_ci": result.ate_ci,
                "p_value": result.p_value,
                "method": result.method,
                "refutation_passed": result.refutation_passed,
                "recommendation": result.recommendation,
                "pipeline_results": result.pipeline_results,
            }
            self._client.table("audit_results").insert(data).execute()
            logger.debug("💾 AuditResult 저장: %s", result.audit_id[:8])
            return result.audit_id
        except Exception as e:
            logger.warning("⚠️ AuditResult 저장 실패: %s", e)
            return None

    # ── 유틸리티 ──

    @property
    def is_connected(self) -> bool:
        """Supabase 연결 상태."""
        return self._connected
