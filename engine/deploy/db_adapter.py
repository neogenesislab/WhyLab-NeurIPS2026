# -*- coding: utf-8 -*-
"""네이티브 Postgres 어댑터 (DLQ/무결성 해시용).

프로토콜 분리:
- DATABASE_URL (포트 6543, Supavisor): DLQ/해시 쓰기 (psycopg2)
- SUPABASE_URL (HTTPS): 읽기 전용 REST API (urllib, 폴백)

고부하 방어:
- ThreadPoolExecutor 비동기 래퍼: 메인 스레드 블로킹 차단
- 배치 삽입 버퍼: 1초 간격 일괄 INSERT
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("whylab.deploy.db_adapter")

# ─── Config ───────────────────────────────────────

# 네이티브 PG (Supavisor Transaction Mode, 포트 6543)
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# HTTPS REST API (PostgREST, 폴백 전용)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


def _get_pg_connection():
    """psycopg2 커넥션 생성 (Supavisor 경유)."""
    if not DATABASE_URL:
        return None
    try:
        import psycopg2
        return psycopg2.connect(DATABASE_URL, connect_timeout=5)
    except ImportError:
        logger.warning("psycopg2 not installed — falling back to REST API")
        return None
    except Exception as e:
        logger.error("PG connection failed: %s", e)
        return None


class AsyncDLQWriter:
    """비블로킹 DLQ 쓰기 — 백그라운드 배치 삽입.

    메인 스레드는 큐에 넣기만 하고 즉시 반환.
    백그라운드 데몬 스레드가 1초 간격으로 배치 INSERT.

    스레드 고갈 방지:
    - 메인 스레드는 queue.put() 만 수행 (< 1μs)
    - DB I/O는 전용 스레드 1개에서만 발생
    - 큐 상한(maxsize=10000)으로 메모리 폭발 방지
    """

    def __init__(self, batch_interval: float = 1.0, max_queue: int = 10000) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._batch_interval = batch_interval
        self._stop_event = threading.Event()
        self._memory_fallback: List[Dict] = []  # DB 미연결 시
        self._persisted_count = 0
        self._error_count = 0

        # 백그라운드 데몬 스레드 시작
        self._thread = threading.Thread(
            target=self._batch_worker,
            name="dlq-writer",
            daemon=True,
        )
        self._thread.start()
        logger.info("📥 DLQ writer started (batch=%ss, max_queue=%d)", batch_interval, max_queue)

    def enqueue(self, decision_id: str, payload: Dict, reason: str = "breaker_tripped") -> None:
        """메인 스레드에서 호출 — 즉시 반환 (Non-blocking).

        큐가 가득 차면 인메모리 폴백에 저장.
        """
        entry = {
            "decision_id": decision_id,
            "reason": reason,
            "payload": payload,
            "timestamp": time.time(),
        }
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self._memory_fallback.append(entry)
            logger.warning("⚠️ DLQ queue full — memory fallback (size=%d)", len(self._memory_fallback))

    def _batch_worker(self) -> None:
        """백그라운드 배치 삽입 루프."""
        while not self._stop_event.is_set():
            time.sleep(self._batch_interval)
            batch = self._drain_queue()
            if not batch:
                continue

            # 네이티브 PG 삽입 시도
            if self._batch_insert_pg(batch):
                self._persisted_count += len(batch)
                continue

            # REST API 폴백
            if self._batch_insert_rest(batch):
                self._persisted_count += len(batch)
                continue

            # 최종 폴백: 인메모리 보존
            self._memory_fallback.extend(batch)
            self._error_count += 1
            logger.error("❌ DLQ batch failed — %d entries in memory fallback", len(batch))

    def _drain_queue(self, max_batch: int = 100) -> List[Dict]:
        """큐에서 항목 꺼내기."""
        batch = []
        while len(batch) < max_batch:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _batch_insert_pg(self, batch: List[Dict]) -> bool:
        """psycopg2 네이티브 배치 INSERT."""
        conn = _get_pg_connection()
        if not conn:
            return False

        try:
            with conn:
                with conn.cursor() as cur:
                    args = [
                        (e["decision_id"], e["reason"], json.dumps(e["payload"]))
                        for e in batch
                    ]
                    cur.executemany(
                        "INSERT INTO audit_dlq (decision_id, reason, payload) "
                        "VALUES (%s, %s, %s::jsonb)",
                        args,
                    )
            logger.info("📥 DLQ batch persisted via PG: %d entries", len(batch))
            return True
        except Exception as e:
            logger.warning("⚠️ PG batch insert failed: %s", e)
            return False
        finally:
            conn.close()

    def _batch_insert_rest(self, batch: List[Dict]) -> bool:
        """REST API 폴백 배치 INSERT."""
        if not (SUPABASE_URL and SUPABASE_KEY):
            return False

        try:
            import urllib.request
            url = f"{SUPABASE_URL}/rest/v1/audit_dlq"
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            }
            body = json.dumps([
                {"decision_id": e["decision_id"], "reason": e["reason"], "payload": e["payload"]}
                for e in batch
            ]).encode()
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status in (200, 201):
                    logger.info("📥 DLQ batch persisted via REST: %d entries", len(batch))
                    return True
        except Exception as e:
            logger.warning("⚠️ REST batch insert failed: %s", e)
        return False

    def shutdown(self, timeout: float = 5.0) -> None:
        """종료 — 잔여 항목 플러시."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        # 잔여 큐 처리
        remaining = self._drain_queue(max_batch=10000)
        if remaining:
            if not self._batch_insert_pg(remaining):
                self._batch_insert_rest(remaining)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "queue_size": self._queue.qsize(),
            "persisted_count": self._persisted_count,
            "memory_fallback_size": len(self._memory_fallback),
            "error_count": self._error_count,
            "thread_alive": self._thread.is_alive(),
        }


class IntegrityHashWriter:
    """무결성 해시 DB 저장 (네이티브 PG)."""

    @staticmethod
    def store(hash_entry: Dict, date_str: str) -> bool:
        """integrity_hashes 테이블에 UPSERT."""
        conn = _get_pg_connection()
        if not conn:
            return False

        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO integrity_hashes 
                            (rollup_date, sha256_hash, record_count, data_bytes)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (rollup_date) DO UPDATE SET
                            sha256_hash = EXCLUDED.sha256_hash,
                            record_count = EXCLUDED.record_count,
                            data_bytes = EXCLUDED.data_bytes
                        """,
                        (
                            date_str,
                            hash_entry["sha256"],
                            hash_entry["record_count"],
                            hash_entry["bytes"],
                        ),
                    )
            logger.info("✅ Integrity hash stored via PG: %s", date_str)
            return True
        except Exception as e:
            logger.warning("⚠️ PG hash write failed: %s", e)
            return False
        finally:
            conn.close()
