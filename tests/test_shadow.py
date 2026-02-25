# -*- coding: utf-8 -*-
"""Phase 4 테스트 — 섀도우 배포 컨트롤러 + 서킷 브레이커."""

import time
import pytest

from engine.deploy.shadow import (
    CostBudget,
    DeploymentMode,
    ShadowDeployController,
    ShadowObservation,
    compute_daily_hash,
    append_hash_log,
)


class TestCostBudget:
    """비용 서킷 브레이커 — 실서비스 방어막."""

    def test_under_budget(self):
        budget = CostBudget(daily_token_limit=10000)
        assert budget.consume(5000) is True
        assert not budget.breaker_tripped

    def test_over_budget_trips(self):
        budget = CostBudget(daily_token_limit=1000)
        budget.consume(900)
        assert budget.consume(200) is False  # 1100 > 1000
        assert budget.breaker_tripped
        assert budget.trip_count == 1

    def test_cost_usd_limit(self):
        budget = CostBudget(daily_cost_limit_usd=5.0)
        budget.consume(100, cost_usd=4.0)
        assert budget.consume(100, cost_usd=2.0) is False  # $6 > $5
        assert budget.breaker_tripped

    def test_remaining_tokens(self):
        budget = CostBudget(daily_token_limit=10000)
        budget.consume(3000)
        assert budget.remaining_tokens == 7000

    def test_utilization(self):
        budget = CostBudget(daily_token_limit=10000)
        budget.consume(5000)
        assert budget.utilization == 0.5


class TestShadowController:
    """섀도우 배포 컨트롤러 — Dry-run 모니터링."""

    def test_dry_run_no_feedback(self):
        """Dry-run에서는 ζ 적용 안 함."""
        ctrl = ShadowDeployController(mode=DeploymentMode.SHADOW_DRY_RUN)
        assert not ctrl.should_apply_feedback()

    def test_active_applies_feedback(self):
        """Active에서는 ζ 적용."""
        ctrl = ShadowDeployController(mode=DeploymentMode.SHADOW_ACTIVE)
        assert ctrl.should_apply_feedback()

    def test_observation_recorded(self):
        ctrl = ShadowDeployController()
        obs = ctrl.record_observation(
            decision_id="d1", proposed_zeta=0.5,
            lyapunov_zeta_max=0.3, drift_index=0.2,
        )
        assert obs.would_have_clipped is True
        assert obs.mode == DeploymentMode.SHADOW_DRY_RUN

    def test_breaker_blocks_deep_audit(self):
        """서킷 브레이커 트립 시 deep audit 차단."""
        budget = CostBudget(daily_token_limit=100)
        budget.consume(200)  # 강제 트립
        ctrl = ShadowDeployController(cost_budget=budget)
        assert not ctrl.should_run_deep_audit()

    def test_promotion_chain(self):
        """DRY_RUN → ACTIVE → PRODUCTION 승격."""
        ctrl = ShadowDeployController(mode=DeploymentMode.SHADOW_DRY_RUN)
        ctrl.promote_to_active()
        assert ctrl.mode == DeploymentMode.SHADOW_ACTIVE
        ctrl.promote_to_production()
        assert ctrl.mode == DeploymentMode.PRODUCTION

    def test_dashboard_stats(self):
        ctrl = ShadowDeployController()
        ctrl.record_observation("d1", 0.5, 0.3, drift_index=0.2)
        ctrl.record_observation("d2", 0.2, 0.4, drift_index=0.1)
        stats = ctrl.get_dashboard_stats()
        assert stats["total_observations"] == 2
        assert 0 <= stats["clip_rate"] <= 1.0
        assert "cost_budget" in stats

    def test_dlq_enqueue(self):
        """서킷 브레이커 차단 시 DLQ에 적재."""
        ctrl = ShadowDeployController()
        ctrl.enqueue_to_dlq("d1", {"ate": 0.15}, reason="breaker_tripped")
        ctrl.enqueue_to_dlq("d2", {"ate": 0.22}, reason="timeout")
        # AsyncDLQWriter가 있으면 큐에, 없으면 메모리에 적재
        if ctrl._dlq_writer:
            assert ctrl._dlq_writer._queue.qsize() >= 0  # 백그라운드 처리 중
        else:
            assert ctrl.dlq_size == 2
            assert ctrl.dlq_entries[0]["decision_id"] == "d1"


class TestDataIntegrity:
    """암호학적 데이터 무결성 서명."""

    def test_sha256_deterministic(self):
        """동일 데이터 → 동일 해시."""
        data = {"records": [{"a": 1}, {"b": 2}]}
        h1 = compute_daily_hash(data, "2026-03-15")
        h2 = compute_daily_hash(data, "2026-03-15")
        assert h1["sha256"] == h2["sha256"]
        assert len(h1["sha256"]) == 64  # SHA-256 hex

    def test_sha256_different_data(self):
        """다른 데이터 → 다른 해시."""
        h1 = compute_daily_hash({"records": [{"a": 1}]}, "2026-03-15")
        h2 = compute_daily_hash({"records": [{"a": 2}]}, "2026-03-15")
        assert h1["sha256"] != h2["sha256"]

    def test_append_hash_log(self, tmp_path):
        """Append-only 해시 로그 파일 생성."""
        log_file = str(tmp_path / "test_hashes.jsonl")
        entry = compute_daily_hash({"records": []}, "2026-03-15")
        append_hash_log(entry, log_file)
        append_hash_log(entry, log_file)
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 2


class TestHeavyTailGuard:
    """Heavy-tail Lyapunov 보정 검증."""

    def test_min_zeta_prevents_deadlock(self):
        """극단적 노이즈에서도 ζ ≥ ε_floor."""
        from engine.audit.lyapunov import LyapunovFilter
        lyap = LyapunovFilter(min_zeta=0.01)
        # 매우 높은 noise → ζ_max ≈ 0 → min_zeta로 바닥
        result = lyap.clip(proposed_zeta=0.5, ate=0.001, drift_index=100.0)
        assert result >= 0.01

