# -*- coding: utf-8 -*-
"""알림 시스템 (Alerter).

드리프트 감지 시 다양한 채널(콘솔, Slack, 이메일)로 알림을 전송합니다.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("whylab.monitoring.alerter")


class AlertLevel(Enum):
    """알림 레벨."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """알림 메시지.

    Attributes:
        level: 알림 레벨 (INFO/WARNING/CRITICAL).
        title: 알림 제목.
        message: 상세 메시지.
        timestamp: 발생 시각.
        metadata: 추가 정보 (ATE, score 등).
    """
    level: AlertLevel
    title: str
    message: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class Alerter:
    """다중 채널 알림 발송기.

    사용법:
        alerter = Alerter(slack_webhook_url="https://hooks.slack.com/...")
        alerter.send(Alert(
            level=AlertLevel.CRITICAL,
            title="Causal Drift Detected",
            message="ATE가 50% 이상 변동했습니다.",
        ))
    """

    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None,
        log_alerts: bool = True,
    ):
        """
        Args:
            slack_webhook_url: Slack 웹훅 URL (환경변수 사용 권장).
            email_config: 이메일 설정 {"smtp_host", "port", "sender", "recipients"}.
            log_alerts: 콘솔 로그로도 알림을 출력할지.
        """
        self.slack_webhook_url = slack_webhook_url
        self.email_config = email_config
        self.log_alerts = log_alerts
        self._history: List[Alert] = []

    def send(self, alert: Alert) -> None:
        """알림을 모든 활성 채널로 발송합니다."""
        self._history.append(alert)

        # 1. 콘솔 로그
        if self.log_alerts:
            self._send_log(alert)

        # 2. Slack
        if self.slack_webhook_url:
            self._send_slack(alert)

        # 3. 이메일 (향후 구현)
        if self.email_config:
            self._send_email(alert)

    def _send_log(self, alert: Alert) -> None:
        """콘솔 로그로 알림을 출력합니다."""
        level_map = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
            alert.level.value, "📢"
        )
        log_fn = level_map.get(alert.level, logger.info)
        log_fn(
            "%s [%s] %s: %s",
            emoji,
            alert.level.value.upper(),
            alert.title,
            alert.message,
        )

    def _send_slack(self, alert: Alert) -> None:
        """Slack 웹훅으로 알림을 발송합니다."""
        try:
            import urllib.request

            emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
                alert.level.value, "📢"
            )
            color = {
                "info": "#36a64f",
                "warning": "#ffcc00",
                "critical": "#ff0000",
            }.get(alert.level.value, "#999999")

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{emoji} {alert.title}",
                        "text": alert.message,
                        "footer": f"WhyLab Monitor | {alert.timestamp}",
                        "fields": [
                            {"title": k, "value": str(v), "short": True}
                            for k, v in alert.metadata.items()
                        ],
                    }
                ]
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.slack_webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info("Slack 알림 전송 완료")

        except Exception as e:
            logger.error("Slack 발송 실패: %s", e)

    def _send_email(self, alert: Alert) -> None:
        """이메일 알림 (향후 구현 예정)."""
        # TODO: smtplib을 사용한 이메일 발송
        logger.info("이메일 알림은 향후 구현 예정입니다.")

    @property
    def history(self) -> List[Alert]:
        """발송된 알림 이력."""
        return list(self._history)

    def clear_history(self) -> None:
        """알림 이력을 초기화합니다."""
        self._history.clear()
