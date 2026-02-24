# -*- coding: utf-8 -*-
"""GA4 API 할당량 관리자 — 큐 기반 Lazy Fetching.

동시 10요청 / 시간당 40K토큰 / 일일 200K토큰 제약을 방어합니다.
결정 기록 시점에는 메타데이터만 저장하고, 백그라운드 워커가
할당량 여유분을 모니터링하며 Outcome 데이터를 지연 수집합니다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from engine.audit.schemas import OutcomeMetric

logger = logging.getLogger("whylab.connectors.ga4_quota")


@dataclass
class OutcomeFetchRequest:
    """GA4 Outcome 수집 요청."""

    request_id: str
    sbu: str
    metric: OutcomeMetric
    start_date: str
    end_date: str
    priority: int = 0  # 높을수록 우선
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "pending"  # pending | processing | done | failed
    estimated_tokens: int = 100  # 예상 토큰 소모량


@dataclass
class QuotaStatus:
    """현재 할당량 상태."""

    concurrent_used: int = 0
    concurrent_max: int = 10
    tokens_used_hour: int = 0
    tokens_max_hour: int = 40_000
    tokens_used_day: int = 0
    tokens_max_day: int = 200_000
    queue_size: int = 0
    last_reset_hour: str = ""

    @property
    def concurrent_available(self) -> int:
        return max(0, self.concurrent_max - self.concurrent_used)

    @property
    def tokens_available_hour(self) -> int:
        return max(0, self.tokens_max_hour - self.tokens_used_hour)

    @property
    def tokens_available_day(self) -> int:
        return max(0, self.tokens_max_day - self.tokens_used_day)

    @property
    def can_process(self) -> bool:
        return (
            self.concurrent_available > 0
            and self.tokens_available_hour > 0
            and self.tokens_available_day > 0
        )


class GA4QuotaManager:
    """큐 기반 GA4 API 할당량 관리자.

    에이전트의 결정 시점에는 수집 '요청'만 큐에 추가합니다.
    process_queue()를 주기적으로 호출하면,
    할당량 여유분을 확인한 뒤 안전하게 데이터를 수집합니다.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        tokens_per_hour: int = 40_000,
        tokens_per_day: int = 200_000,
        safety_margin: float = 0.2,  # 20% 여유분 확보
    ) -> None:
        self._max_concurrent = max_concurrent
        self._tokens_per_hour = tokens_per_hour
        self._tokens_per_day = tokens_per_day
        self._safety_margin = safety_margin
        self._queue: List[OutcomeFetchRequest] = []
        self._concurrent_count = 0
        self._tokens_used_hour = 0
        self._tokens_used_day = 0
        self._last_hour_reset = time.time()
        self._lock = Lock()
        self._processed: List[OutcomeFetchRequest] = []

    def enqueue(self, request: OutcomeFetchRequest) -> str:
        """수집 요청을 큐에 추가합니다. 즉시 실행하지 않습니다.

        Returns:
            요청 ID
        """
        with self._lock:
            self._queue.append(request)
            self._queue.sort(key=lambda r: -r.priority)

        logger.info(
            "📥 GA4 요청 큐 추가: %s (%s/%s, %s~%s) [큐 크기: %d]",
            request.request_id,
            request.sbu,
            request.metric.value,
            request.start_date,
            request.end_date,
            len(self._queue),
        )
        return request.request_id

    def process_queue(
        self,
        fetch_fn: Callable[[OutcomeFetchRequest], Any],
        max_batch: int = 5,
    ) -> int:
        """할당량 여유분 확인 후 큐에서 요청을 처리합니다.

        Args:
            fetch_fn: 실제 GA4 API 호출 함수
            max_batch: 한 번에 처리할 최대 요청 수

        Returns:
            처리된 요청 수
        """
        self._maybe_reset_hourly()

        processed = 0
        with self._lock:
            while self._queue and processed < max_batch:
                status = self.get_quota_status()
                if not status.can_process:
                    logger.warning(
                        "⚠️ GA4 할당량 소진 — 큐 대기. "
                        "concurrent=%d/%d, tokens_h=%d/%d",
                        status.concurrent_used,
                        status.concurrent_max,
                        status.tokens_used_hour,
                        status.tokens_max_hour,
                    )
                    break

                request = self._queue.pop(0)
                est_tokens = request.estimated_tokens
                safe_limit = int(self._tokens_per_hour * (1 - self._safety_margin))

                if self._tokens_used_hour + est_tokens > safe_limit:
                    self._queue.insert(0, request)
                    logger.info("⏸️ 토큰 안전 마진 도달 — 다음 주기로 연기")
                    break

                request.status = "processing"
                self._concurrent_count += 1

                try:
                    fetch_fn(request)
                    request.status = "done"
                    self._tokens_used_hour += est_tokens
                    self._tokens_used_day += est_tokens
                    processed += 1
                except Exception as e:
                    request.status = "failed"
                    logger.warning("❌ GA4 요청 실패: %s — %s", request.request_id, e)
                finally:
                    self._concurrent_count -= 1
                    self._processed.append(request)

        if processed > 0:
            logger.info("✅ GA4 큐 처리: %d건 완료, 잔여 %d건", processed, len(self._queue))

        return processed

    def get_quota_status(self) -> QuotaStatus:
        """현재 할당량 상태를 반환합니다."""
        self._maybe_reset_hourly()
        return QuotaStatus(
            concurrent_used=self._concurrent_count,
            concurrent_max=self._max_concurrent,
            tokens_used_hour=self._tokens_used_hour,
            tokens_max_hour=self._tokens_per_hour,
            tokens_used_day=self._tokens_used_day,
            tokens_max_day=self._tokens_per_day,
            queue_size=len(self._queue),
        )

    def _maybe_reset_hourly(self) -> None:
        """시간당 토큰 카운터를 리셋합니다."""
        now = time.time()
        if now - self._last_hour_reset >= 3600:
            self._tokens_used_hour = 0
            self._last_hour_reset = now
            logger.debug("🔄 GA4 시간당 토큰 카운터 리셋")

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def processed_count(self) -> int:
        return len(self._processed)
