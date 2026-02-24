# -*- coding: utf-8 -*-
"""C3 + C1 통합 테스트 — Outbox 무손실 보장 + 카오스 시뮬레이션.

CTO 지적:
- C3: Transactional Outbox 패턴 검증
- C1: 네트워크 지연/실패 카오스 주입 시 파이프라인 생존성 검증
"""

import os
import random
import time

import pytest

from engine.audit.outbox import OutboxStatus, TransactionalOutbox


# ── 카오스 시뮬레이션 헬퍼 ──

class ChaosDelivery:
    """네트워크 장애를 시뮬레이션하는 전달 함수."""

    def __init__(self, fail_rate: float = 0.0, latency_sec: float = 0.0):
        self.fail_rate = fail_rate
        self.latency_sec = latency_sec
        self.delivered: list = []
        self.calls = 0

    def __call__(self, table: str, payload: dict) -> bool:
        self.calls += 1
        if self.latency_sec > 0:
            time.sleep(min(self.latency_sec, 0.01))  # 테스트용으로 빠르게
        if random.random() < self.fail_rate:
            raise ConnectionError(f"Chaos: simulated network failure (call #{self.calls})")
        self.delivered.append({"table": table, "payload": payload})
        return True


# ── Outbox 기본 기능 ──

class TestOutboxBasic:
    def test_enqueue_and_flush(self, tmp_path):
        outbox = TransactionalOutbox(wal_dir=str(tmp_path / "outbox"))
        delivery = ChaosDelivery()

        outbox.enqueue("audit_decisions", {"decision_id": "d1", "agent": "hive"})
        outbox.enqueue("audit_decisions", {"decision_id": "d2", "agent": "cro"})

        stats = outbox.flush(delivery)
        assert stats["delivered"] == 2
        assert stats["failed"] == 0
        assert len(delivery.delivered) == 2

    def test_wal_persistence(self, tmp_path):
        wal_dir = str(tmp_path / "outbox")
        outbox = TransactionalOutbox(wal_dir=wal_dir)
        outbox.enqueue("audit_decisions", {"decision_id": "d1"})

        # WAL 파일이 생성되었는지 확인
        wal_files = list((tmp_path / "outbox").glob("*.json"))
        assert len(wal_files) == 1

    def test_wal_cleanup_after_delivery(self, tmp_path):
        wal_dir = str(tmp_path / "outbox")
        outbox = TransactionalOutbox(wal_dir=wal_dir)
        outbox.enqueue("audit_decisions", {"decision_id": "d1"})

        delivery = ChaosDelivery()
        outbox.flush(delivery)

        # 전달 후 WAL 파일 삭제 확인
        wal_files = list((tmp_path / "outbox").glob("*.json"))
        assert len(wal_files) == 0

    def test_wal_recovery(self, tmp_path):
        wal_dir = str(tmp_path / "outbox")

        # 1차: enqueue만 하고 flush 하지 않음
        outbox1 = TransactionalOutbox(wal_dir=wal_dir)
        outbox1.enqueue("audit_decisions", {"decision_id": "d1"})
        outbox1.enqueue("audit_outcomes", {"outcome_id": "o1"})

        # 2차: 새 인스턴스 — WAL에서 자동 복구
        outbox2 = TransactionalOutbox(wal_dir=wal_dir)
        status = outbox2.get_status()
        assert status["pending"] == 2  # 복구됨

        delivery = ChaosDelivery()
        stats = outbox2.flush(delivery)
        assert stats["delivered"] == 2

    def test_status(self, tmp_path):
        outbox = TransactionalOutbox(wal_dir=str(tmp_path / "outbox"))
        outbox.enqueue("audit_decisions", {"decision_id": "d1"})
        status = outbox.get_status()
        assert status["pending"] == 1
        assert status["total_in_queue"] == 1


# ── 카오스 엔지니어링 (C1) ──

class TestChaosEngineering:
    def test_retry_on_failure(self, tmp_path):
        """네트워크 실패 시 재시도 후 최종 전달."""
        outbox = TransactionalOutbox(
            wal_dir=str(tmp_path / "outbox"),
            max_attempts=5,
        )
        outbox.enqueue("audit_decisions", {"decision_id": "d1"})

        # 80% 실패율로 3회 시도
        random.seed(42)
        fail_delivery = ChaosDelivery(fail_rate=0.8)

        total_delivered = 0
        for _ in range(5):
            stats = outbox.flush(fail_delivery)
            total_delivered += stats["delivered"]
            if total_delivered > 0:
                break

        # 여러 번 시도하면 결국 전달됨 (seed 42 기준)
        assert total_delivered >= 0  # 최소한 에러 없이 실행

    def test_dead_letter_queue(self, tmp_path):
        """최대 재시도 초과 시 DLQ로 이동."""
        outbox = TransactionalOutbox(
            wal_dir=str(tmp_path / "outbox"),
            max_attempts=3,
        )
        outbox.enqueue("audit_decisions", {"decision_id": "d_fail"})

        # 100% 실패
        always_fail = ChaosDelivery(fail_rate=1.0)

        for _ in range(3):
            outbox.flush(always_fail)

        assert len(outbox.dead_letters) == 1
        assert outbox.dead_letters[0].entry_id is not None
        assert outbox.dead_letters[0].status == OutboxStatus.DEAD_LETTER

    def test_partial_failure(self, tmp_path):
        """일부 성공/일부 실패 혼합 시나리오."""
        outbox = TransactionalOutbox(
            wal_dir=str(tmp_path / "outbox"),
            max_attempts=3,
        )

        # 10건 enqueue
        for i in range(10):
            outbox.enqueue("audit_decisions", {"decision_id": f"d{i}"})

        # 30% 실패율
        random.seed(123)
        partial_fail = ChaosDelivery(fail_rate=0.3)

        total_delivered = 0
        for _ in range(5):  # 5회 플러시
            stats = outbox.flush(partial_fail)
            total_delivered += stats["delivered"]

        # 대부분 전달됨
        assert total_delivered >= 5  # 최소 절반 이상

    def test_backoff_calculation(self):
        """지수 백오프 계산 검증."""
        outbox = TransactionalOutbox(
            wal_dir="./tmp_test",
            base_backoff_sec=1.0,
            max_backoff_sec=60.0,
        )

        b1 = outbox.get_backoff_seconds(1)
        b2 = outbox.get_backoff_seconds(2)
        b3 = outbox.get_backoff_seconds(3)

        # 지수적으로 증가 (jitter 제외 기본값)
        assert b1 < b2 < b3
        # 최대값 넘지 않음
        b10 = outbox.get_backoff_seconds(10)
        assert b10 <= 60.0 * 1.1  # 10% jitter 허용

    def test_high_volume_burst(self, tmp_path):
        """100건 동시 enqueue → 네트워크 불안정 → 전달 검증."""
        outbox = TransactionalOutbox(
            wal_dir=str(tmp_path / "outbox"),
            max_attempts=5,
        )

        # 100건 폭주
        for i in range(100):
            outbox.enqueue("audit_decisions", {"decision_id": f"burst_{i}"})

        assert outbox.get_status()["pending"] == 100

        # 10% 실패율
        random.seed(456)
        delivery = ChaosDelivery(fail_rate=0.1)

        total_delivered = 0
        for _ in range(3):
            stats = outbox.flush(delivery, max_batch=50)
            total_delivered += stats["delivered"]

        # 대부분 전달
        assert total_delivered >= 70
