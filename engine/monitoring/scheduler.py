# -*- coding: utf-8 -*-
"""인과 모니터링 스케줄러.

주기적으로 파이프라인을 실행하고, 드리프트를 감지하여 알림을 보냅니다.
"""

import logging
import time
from typing import Optional

from engine.config import WhyLabConfig
from engine.monitoring.drift_detector import DriftDetector, DriftResult
from engine.monitoring.alerter import Alerter, Alert, AlertLevel

logger = logging.getLogger("whylab.monitoring.scheduler")


class MonitoringScheduler:
    """인과 모니터링 스케줄러.

    주기적으로:
    1. 데이터 소스에서 최신 데이터를 가져오고
    2. WhyLab 파이프라인을 실행하여 ATE/CATE를 갱신하고
    3. 이전 결과와 비교하여 드리프트를 감지하고
    4. 드리프트 발생 시 알림을 전송합니다.

    사용법:
        scheduler = MonitoringScheduler(
            config=WhyLabConfig(),
            alerter=Alerter(slack_webhook_url="..."),
            interval_minutes=60,
        )
        scheduler.start()  # 블로킹 루프
        # 또는
        scheduler.run_once()  # 1회 실행
    """

    def __init__(
        self,
        config: WhyLabConfig,
        alerter: Optional[Alerter] = None,
        interval_minutes: int = 60,
        scenario: str = "A",
    ):
        """
        Args:
            config: WhyLab 설정.
            alerter: 알림 발송기 (None이면 콘솔 로그만).
            interval_minutes: 모니터링 주기 (분).
            scenario: 기본 시나리오.
        """
        self.config = config
        self.alerter = alerter or Alerter(log_alerts=True)
        self.interval_minutes = interval_minutes
        self.scenario = scenario

        self.detector = DriftDetector()
        self._running = False
        self._run_count = 0

    def run_once(self) -> DriftResult:
        """1회 파이프라인 실행 + 드리프트 체크.

        Returns:
            DriftResult: 드리프트 탐지 결과.
        """
        self._run_count += 1
        logger.info("="*50)
        logger.info("모니터링 실행 #%d", self._run_count)

        # 1. 파이프라인 실행
        try:
            from engine.orchestrator import Orchestrator
            orchestrator = Orchestrator(config=self.config)
            result = orchestrator.run_pipeline(scenario=self.scenario)
        except Exception as e:
            logger.error("파이프라인 실행 실패: %s", e)
            self.alerter.send(Alert(
                level=AlertLevel.CRITICAL,
                title="파이프라인 실행 실패",
                message=str(e),
            ))
            return DriftResult(drifted=False, metric="pipeline_error")

        # 2. ATE/CATE 추출
        ate = result.get("ate", 0)
        if isinstance(ate, dict):
            ate = ate.get("point_estimate", 0)

        cate = result.get("cate_predictions")
        cate_list = cate.tolist() if hasattr(cate, "tolist") else []

        # 3. 스냅샷 추가
        self.detector.add_snapshot(
            ate=ate,
            cate_distribution=cate_list,
            metadata={"run": self._run_count, "scenario": self.scenario},
        )

        # 4. 드리프트 판단
        drift_result = self.detector.check_drift()

        if drift_result.drifted:
            level = (
                AlertLevel.CRITICAL
                if drift_result.score > 1.0
                else AlertLevel.WARNING
            )
            self.alerter.send(Alert(
                level=level,
                title=f"Causal Drift Detected ({drift_result.metric})",
                message=(
                    f"드리프트 점수: {drift_result.score:.4f} "
                    f"(임계값: {drift_result.threshold:.4f}). "
                    f"ATE: {ate:.4f}."
                ),
                metadata={
                    "run": self._run_count,
                    "ate": ate,
                    "drift_score": drift_result.score,
                    "metric": drift_result.metric,
                },
            ))
        else:
            logger.info(
                "✅ 드리프트 미감지 (ATE=%.4f, 체크=%s)",
                ate, drift_result.metric,
            )

        return drift_result

    def start(self, max_runs: Optional[int] = None) -> None:
        """모니터링 루프를 시작합니다 (블로킹).

        Args:
            max_runs: 최대 실행 횟수 (None이면 무한).
        """
        self._running = True
        logger.info(
            "🔄 모니터링 시작 (주기: %d분, 시나리오: %s)",
            self.interval_minutes, self.scenario,
        )

        try:
            while self._running:
                self.run_once()

                if max_runs and self._run_count >= max_runs:
                    logger.info("최대 실행 횟수 도달. 종료.")
                    break

                logger.info(
                    "다음 실행까지 %d분 대기...", self.interval_minutes
                )
                time.sleep(self.interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("사용자에 의해 모니터링 중단.")
        finally:
            self._running = False

    def stop(self) -> None:
        """모니터링 루프를 중단합니다."""
        self._running = False
        logger.info("모니터링 중단 요청.")

    @property
    def status(self) -> dict:
        """현재 모니터링 상태."""
        return {
            "running": self._running,
            "run_count": self._run_count,
            "snapshot_count": self.detector.snapshot_count,
            "interval_minutes": self.interval_minutes,
            "alert_history_count": len(self.alerter.history),
        }
