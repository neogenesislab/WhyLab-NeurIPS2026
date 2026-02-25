# -*- coding: utf-8 -*-
"""Phase 4 테스트 — 섀도우 배포 컨트롤러 + 서킷 브레이커."""

import time
import pytest

from engine.deploy.shadow import (
    CostBudget,
    DeploymentMode,
    ShadowDeployController,
    ShadowObservation,
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
