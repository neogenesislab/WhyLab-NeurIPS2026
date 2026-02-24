# -*- coding: utf-8 -*-
"""Transactional Outbox — 결정 로그 무손실 보장.

에이전트가 결정을 내릴 때 로컬 WAL(Write-Ahead Log)에 먼저
기록하고, 비동기 워커가 이를 Supabase로 확실하게 전달합니다.

CTO 지적 (C3): "네트워크 문제로 Supabase 로깅 누락 시
인과 감사 엔진이 잘못된 피드백을 주게 된다."

보장:
- At-least-once delivery (최소 1회 전달)
- 로컬 WAL → Supabase 동기화
- 실패 시 지수 백오프 재시도
- Dead Letter Queue (DLQ) → 인간 개입
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("whylab.audit.outbox")


class OutboxStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class OutboxEntry:
    """Outbox 항목."""
    entry_id: str
    table: str
    payload: Dict[str, Any]
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    max_attempts: int = 5
    created_at: float = field(default_factory=time.time)
    last_attempt_at: Optional[float] = None
    error: Optional[str] = None


class TransactionalOutbox:
    """결정 로그 무손실 전달을 보장하는 Outbox 패턴 구현.

    사용법:
        outbox = TransactionalOutbox(wal_dir="./data/outbox")

        # 에이전트 결정 시 — 로컬 WAL에 먼저 기록
        outbox.enqueue("audit_decisions", {"decision_id": "d1", ...})

        # 비동기 워커 — Supabase로 전달
        outbox.flush(deliver_fn=supabase_client.table("x").insert)
    """

    def __init__(
        self,
        wal_dir: str = "./data/outbox",
        max_attempts: int = 5,
        base_backoff_sec: float = 1.0,
        max_backoff_sec: float = 60.0,
    ) -> None:
        self._wal_dir = Path(wal_dir)
        self._wal_dir.mkdir(parents=True, exist_ok=True)
        self._max_attempts = max_attempts
        self._base_backoff = base_backoff_sec
        self._max_backoff = max_backoff_sec
        self._lock = threading.Lock()
        self._queue: List[OutboxEntry] = []
        self._dlq: List[OutboxEntry] = []

        # 시작 시 미전달 WAL 복구
        self._recover_wal()

    def enqueue(self, table: str, payload: Dict[str, Any], entry_id: Optional[str] = None) -> str:
        """Outbox에 항목을 추가합니다 (로컬 WAL 기록).

        Returns:
            생성된 entry_id
        """
        if entry_id is None:
            import uuid
            entry_id = str(uuid.uuid4())

        entry = OutboxEntry(
            entry_id=entry_id,
            table=table,
            payload=payload,
            max_attempts=self._max_attempts,
        )

        with self._lock:
            self._queue.append(entry)
            self._write_wal(entry)

        logger.debug("📥 Outbox enqueue: %s → %s", entry_id[:8], table)
        return entry_id

    def flush(
        self,
        deliver_fn: Callable[[str, Dict[str, Any]], bool],
        max_batch: int = 50,
    ) -> Dict[str, int]:
        """대기 중인 항목을 일괄 전달합니다.

        Args:
            deliver_fn: (table, payload) → bool (성공 여부)
            max_batch: 1회 플러시 최대 건수

        Returns:
            {"delivered": N, "failed": N, "dead_letter": N}
        """
        stats = {"delivered": 0, "failed": 0, "dead_letter": 0}

        with self._lock:
            pending = [
                e for e in self._queue
                if e.status in (OutboxStatus.PENDING, OutboxStatus.FAILED)
            ][:max_batch]

        for entry in pending:
            entry.status = OutboxStatus.PROCESSING
            entry.attempts += 1
            entry.last_attempt_at = time.time()

            try:
                success = deliver_fn(entry.table, entry.payload)
                if success:
                    entry.status = OutboxStatus.DELIVERED
                    self._remove_wal(entry.entry_id)
                    stats["delivered"] += 1
                    logger.debug("✅ Delivered: %s", entry.entry_id[:8])
                else:
                    raise RuntimeError("deliver_fn returned False")
            except Exception as e:
                entry.error = str(e)
                if entry.attempts >= entry.max_attempts:
                    entry.status = OutboxStatus.DEAD_LETTER
                    self._dlq.append(entry)
                    stats["dead_letter"] += 1
                    logger.error(
                        "💀 Dead Letter: %s after %d attempts: %s",
                        entry.entry_id[:8], entry.attempts, e,
                    )
                else:
                    entry.status = OutboxStatus.FAILED
                    stats["failed"] += 1
                    logger.warning(
                        "⚠️ Retry %d/%d: %s — %s",
                        entry.attempts, entry.max_attempts,
                        entry.entry_id[:8], e,
                    )

        # 전달 완료된 항목 제거
        with self._lock:
            self._queue = [
                e for e in self._queue
                if e.status not in (OutboxStatus.DELIVERED, OutboxStatus.DEAD_LETTER)
            ]

        if stats["delivered"] > 0 or stats["dead_letter"] > 0:
            logger.info(
                "📤 Outbox flush: delivered=%d, failed=%d, dlq=%d, pending=%d",
                stats["delivered"], stats["failed"], stats["dead_letter"],
                len(self._queue),
            )

        return stats

    def get_backoff_seconds(self, attempt: int) -> float:
        """지수 백오프 계산 (jitter 포함)."""
        import random
        delay = min(
            self._base_backoff * (2 ** (attempt - 1)),
            self._max_backoff,
        )
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter

    def get_status(self) -> Dict[str, Any]:
        """Outbox 현황."""
        with self._lock:
            return {
                "pending": sum(1 for e in self._queue if e.status == OutboxStatus.PENDING),
                "failed": sum(1 for e in self._queue if e.status == OutboxStatus.FAILED),
                "dead_letter": len(self._dlq),
                "total_in_queue": len(self._queue),
                "wal_dir": str(self._wal_dir),
            }

    @property
    def dead_letters(self) -> List[OutboxEntry]:
        """Dead Letter Queue 조회."""
        return list(self._dlq)

    # ── WAL (Write-Ahead Log) ──

    def _write_wal(self, entry: OutboxEntry) -> None:
        """WAL 파일에 항목 기록."""
        wal_file = self._wal_dir / f"{entry.entry_id}.json"
        data = {
            "entry_id": entry.entry_id,
            "table": entry.table,
            "payload": entry.payload,
            "created_at": entry.created_at,
        }
        wal_file.write_text(json.dumps(data, default=str), encoding="utf-8")

    def _remove_wal(self, entry_id: str) -> None:
        """전달 완료된 WAL 파일 삭제."""
        wal_file = self._wal_dir / f"{entry_id}.json"
        if wal_file.exists():
            wal_file.unlink()

    def _recover_wal(self) -> None:
        """시작 시 미전달 WAL 파일 복구."""
        wal_files = list(self._wal_dir.glob("*.json"))
        if not wal_files:
            return

        recovered = 0
        for wal_file in wal_files:
            try:
                data = json.loads(wal_file.read_text(encoding="utf-8"))
                entry = OutboxEntry(
                    entry_id=data["entry_id"],
                    table=data["table"],
                    payload=data["payload"],
                    created_at=data.get("created_at", time.time()),
                )
                self._queue.append(entry)
                recovered += 1
            except Exception as e:
                logger.warning("⚠️ WAL 복구 실패: %s — %s", wal_file.name, e)

        if recovered > 0:
            logger.info("🔄 WAL 복구: %d건 미전달 항목 발견", recovered)
