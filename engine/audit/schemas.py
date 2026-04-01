# -*- coding: utf-8 -*-
"""Causal Audit 스키마 정의 — Decision/Outcome 이벤트 모델.

에이전트의 결정(Decision)과 비즈니스 결과(Outcome)를 표준화하여
인과 감사 파이프라인에 투입할 수 있는 데이터 구조를 정의합니다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# 에이전트 식별
# ──────────────────────────────────────────────

class AgentType(str, Enum):
    """neo-genesis 에코시스템의 에이전트 유형."""

    HIVE_MIND = "hive_mind"          # 콘텐츠 생성 오케스트레이터
    CRO_AGENT = "cro_agent"          # 전환율 최적화 에이전트
    FARMING_BOT = "farming_bot"      # 에어드롭/DeFi 파밍 봇
    CONTENT_AGENT = "content_agent"  # SEO/콘텐츠 에이전트
    CUSTOM = "custom"                # 사용자 정의 에이전트


class DecisionType(str, Enum):
    """에이전트 결정의 유형."""

    CONTENT_STRATEGY = "content_strategy"      # 콘텐츠 주제/키워드 변경
    UI_CHANGE = "ui_change"                    # UI/UX 변경 (CTA 위치, 색상 등)
    POSTING_FREQUENCY = "posting_frequency"    # 포스팅 빈도 변경
    SEO_OPTIMIZATION = "seo_optimization"      # SEO 전략 변경
    AD_PLACEMENT = "ad_placement"              # 광고 위치 변경
    FEATURE_ROLLOUT = "feature_rollout"        # 기능 배포
    PROTOCOL_INTERACTION = "protocol_interaction"  # DeFi 프로토콜 상호작용
    CUSTOM = "custom"


class OutcomeMetric(str, Enum):
    """측정 가능한 비즈니스 결과 지표."""

    ORGANIC_TRAFFIC = "organic_traffic"    # 오가닉 트래픽 (PV/UV)
    BOUNCE_RATE = "bounce_rate"            # 이탈률
    SESSION_DURATION = "session_duration"  # 평균 세션 지속 시간
    CLICK_RATE = "click_rate"              # 클릭률 (CTR)
    CONVERSION_RATE = "conversion_rate"    # 전환율
    REVENUE = "revenue"                    # 수익
    PAGE_VIEWS = "page_views"             # 페이지뷰
    SUBSCRIBERS = "subscribers"           # 구독자 수
    CUSTOM = "custom"


class AuditVerdict(str, Enum):
    """감사 판결 결과."""

    CAUSAL = "CAUSAL"               # 인과관계 확인
    NOT_CAUSAL = "NOT_CAUSAL"       # 인과관계 부정
    UNCERTAIN = "UNCERTAIN"         # 불확실 (추가 데이터 필요)
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # 데이터 부족
    PENDING = "PENDING"             # 감사 대기


# ──────────────────────────────────────────────
# Decision Event — 에이전트가 내린 결정
# ──────────────────────────────────────────────

@dataclass
class DecisionEvent:
    """에이전트의 결정 이벤트.

    Attributes:
        decision_id: 고유 식별자
        agent_type: 결정을 내린 에이전트 유형
        agent_name: 에이전트 구체 이름 (예: "hive_mind_toolpick")
        decision_type: 결정 유형
        treatment: 처치 변수 설명 (예: "키워드 전략을 AI Tools로 변경")
        treatment_value: 처치 변수 값 (수치 또는 카테고리)
        target_sbu: 대상 SBU (예: "toolpick", "ur-wrong")
        target_metric: 기대 영향 지표
        timestamp: 결정 시점 (UTC)
        context: 추가 맥락 정보
        expected_effect: 에이전트가 예측한 효과 방향 (+/-)
        observation_window_days: 효과 관측 기간 (일)
    """

    agent_type: AgentType
    agent_name: str
    decision_type: DecisionType
    treatment: str
    target_sbu: str
    target_metric: OutcomeMetric

    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    treatment_value: Any = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    context: Dict[str, Any] = field(default_factory=dict)
    expected_effect: str = "positive"  # positive | negative | neutral
    observation_window_days: int = 7

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리로 변환."""
        d = asdict(self)
        d["agent_type"] = self.agent_type.value
        d["decision_type"] = self.decision_type.value
        d["target_metric"] = self.target_metric.value
        return d


# ──────────────────────────────────────────────
# Outcome Event — 관측된 비즈니스 결과
# ──────────────────────────────────────────────

@dataclass
class OutcomeEvent:
    """비즈니스 결과 관측 이벤트.

    Attributes:
        outcome_id: 고유 식별자
        metric: 측정 지표
        value: 측정값
        timestamp: 관측 시점 (UTC)
        sbu: 관측 대상 SBU
        source: 데이터 소스 (ga4, posthog, manual 등)
        period: 집계 기간 (daily, weekly 등)
        metadata: 추가 메타데이터
    """

    metric: OutcomeMetric
    value: float
    sbu: str

    outcome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "ga4"
    period: str = "daily"  # daily | weekly | monthly
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리로 변환."""
        d = asdict(self)
        d["metric"] = self.metric.value
        return d


# ──────────────────────────────────────────────
# Audit Result — 감사 결과
# ──────────────────────────────────────────────

@dataclass
class AuditResult:
    """인과 감사 결과.

    Attributes:
        audit_id: 고유 식별자
        decision_id: 감사 대상 결정 ID
        verdict: 감사 판결
        confidence: 확신도 (0~1)
        ate: 평균 처치 효과 (Average Treatment Effect)
        ate_ci: ATE 신뢰구간 [lower, upper]
        p_value: 통계적 유의성
        method: 사용된 인과추론 방법론
        refutation_passed: 반증 테스트 통과 여부
        recommendation: 1-Pager 보고서 (마크다운)
        timestamp: 감사 시점
        pipeline_results: WhyLab 파이프라인 전체 결과 (raw)
    """

    decision_id: str
    verdict: AuditVerdict
    confidence: float

    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ate: float = 0.0
    ate_ci: List[float] = field(default_factory=lambda: [0.0, 0.0])
    p_value: Optional[float] = None
    method: str = "CausalImpact"
    refutation_passed: bool = False
    recommendation: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리로 변환."""
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


# ──────────────────────────────────────────────
# Decision-Outcome Pair — 매칭된 쌍
# ──────────────────────────────────────────────

@dataclass
class DecisionOutcomePair:
    """감사를 위해 매칭된 결정-결과 쌍.

    Attributes:
        decision: 에이전트 결정
        pre_outcomes: 결정 이전 관측값 (시계열)
        post_outcomes: 결정 이후 관측값 (시계열)
        audit_result: 감사 결과 (감사 후 채워짐)
    """

    decision: DecisionEvent
    pre_outcomes: List[OutcomeEvent] = field(default_factory=list)
    post_outcomes: List[OutcomeEvent] = field(default_factory=list)
    audit_result: Optional[AuditResult] = None

    @property
    def is_ready_for_audit(self) -> bool:
        """감사에 필요한 최소 데이터가 확보되었는지 확인."""
        min_pre = 7   # 최소 7일 사전 관측
        min_post = 3  # 최소 3일 사후 관측
        return len(self.pre_outcomes) >= min_pre and len(self.post_outcomes) >= min_post

    @property
    def pre_values(self) -> List[float]:
        """사전 관측값 리스트."""
        return [o.value for o in sorted(self.pre_outcomes, key=lambda x: x.timestamp)]

    @property
    def post_values(self) -> List[float]:
        """사후 관측값 리스트."""
        return [o.value for o in sorted(self.post_outcomes, key=lambda x: x.timestamp)]
