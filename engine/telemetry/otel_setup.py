# -*- coding: utf-8 -*-
"""C4: OpenTelemetry 설정 — 동적 샘플링 + GenAI 시맨틱 컨벤션.

레거시 tracing.py 대체. OTel 표준 기반 분산 추적.

CTO 지적:
- 100% 전송 시 비용 폭발 → Tail-based 동적 샘플링
- 성공 케이스: 5% 샘플링
- 에러/DI 급등: 100% 전송

Reviewer 기여:
- OTel Span Graph = MACIE/ECHO의 인과 그래프(DAG) 입력
- GenAI 시맨틱 컨벤션 (토큰 수, 에이전트 Role, 지연 시간)
"""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger("whylab.telemetry.otel")


class SpanKind(str, Enum):
    """OTel Span 종류."""
    AGENT_DECISION = "agent_decision"
    AUDIT_RUN = "audit_run"
    LLM_CALL = "llm_call"
    DATA_FETCH = "data_fetch"
    FEEDBACK_LOOP = "feedback_loop"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """OTel Span (경량 구현).

    실제 프로덕션에서는 opentelemetry-sdk의 Span으로 교체.
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str = ""
    kind: SpanKind = SpanKind.AGENT_DECISION
    status: SpanStatus = SpanStatus.OK
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dag_node(self) -> Dict[str, Any]:
        """DAG 노드로 변환 (MACIE/ECHO 인과 그래프 입력).

        Reviewer 기여: OTel Span → 인과 DAG 노드 자동 매핑.
        """
        return {
            "node_id": self.span_id,
            "parent_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "agent": self.attributes.get("agent.id", "unknown"),
            "action": self.name,
            "kind": self.kind.value,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
        }


class DynamicSampler:
    """동적 샘플링 — Tail-based Sampling 구현.

    CTO 요구사항:
    - 정상 케이스: base_rate (기본 5%) 샘플링
    - 에러 발생: 100% 전송
    - DI 급등: 100% 전송
    - 인과 감사 트리거: 100% 전송
    """

    def __init__(
        self,
        base_rate: float = 0.05,
        error_rate: float = 1.0,
        drift_alert_rate: float = 1.0,
    ) -> None:
        self.base_rate = base_rate
        self.error_rate = error_rate
        self.drift_alert_rate = drift_alert_rate
        self._stats = {"sampled": 0, "dropped": 0, "forced": 0}

    def should_sample(
        self,
        span: Span,
        drift_index: float = 0.0,
        drift_threshold: float = 0.3,
    ) -> bool:
        """Span을 전송할지 결정합니다.

        우선순위:
        1. 에러 → 무조건 전송
        2. DI > threshold → 무조건 전송
        3. 그 외 → base_rate 확률
        """
        # 에러는 무조건 전송
        if span.status == SpanStatus.ERROR:
            self._stats["forced"] += 1
            return True

        # DI 급등 시 무조건 전송
        if drift_index > drift_threshold:
            self._stats["forced"] += 1
            return True

        # 인과 감사 관련 Span은 무조건 전송
        if span.kind in (SpanKind.AUDIT_RUN, SpanKind.FEEDBACK_LOOP):
            self._stats["forced"] += 1
            return True

        # 일반 케이스: 확률적 샘플링
        if random.random() < self.base_rate:
            self._stats["sampled"] += 1
            return True

        self._stats["dropped"] += 1
        return False

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ── GenAI 시맨틱 컨벤션 ──

class GenAIAttributes:
    """OTel GenAI Semantic Conventions.

    참조: OpenTelemetry GenAI Semantic Conventions (실험적)
    에이전트의 LLM 호출에 특화된 표준 속성.
    """
    # 시스템
    SYSTEM = "gen_ai.system"              # "openai", "anthropic"
    MODEL = "gen_ai.request.model"        # "gpt-4o", "claude-3.7"
    TEMPERATURE = "gen_ai.request.temperature"
    MAX_TOKENS = "gen_ai.request.max_tokens"

    # 토큰 사용량
    PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # 에이전트
    AGENT_ID = "agent.id"
    AGENT_ROLE = "agent.role"            # "auditor", "optimizer", "cro"
    DECISION_ID = "agent.decision_id"

    # 인과 감사
    AUDIT_METHOD = "whylab.audit.method"  # "dml", "causal_impact", "gsc"
    AUDIT_ATE = "whylab.audit.ate"
    AUDIT_CONFIDENCE = "whylab.audit.confidence"
    AUDIT_DI = "whylab.audit.drift_index"
    AUDIT_EVALUE = "whylab.audit.e_value"


class WhyLabTracer:
    """WhyLab OTel 트레이서 (경량 구현).

    실제 프로덕션: opentelemetry-sdk TracerProvider로 교체.
    테스트/논문 검증: 이 경량 구현 사용.
    """

    def __init__(
        self,
        service_name: str = "whylab-audit",
        sampler: Optional[DynamicSampler] = None,
    ) -> None:
        self.service_name = service_name
        self.sampler = sampler or DynamicSampler()
        self._traces: Dict[str, List[Span]] = {}
        self._active_span: Optional[Span] = None

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.AGENT_DECISION,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """새 Span을 시작합니다."""
        import uuid
        trace_id = self._active_span.trace_id if self._active_span else str(uuid.uuid4())
        parent_id = self._active_span.span_id if self._active_span else None

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_id,
            name=name,
            kind=kind,
            start_time=time.time(),
        )
        if attributes:
            span.attributes.update(attributes)

        prev_span = self._active_span
        self._active_span = span

        try:
            yield span
            span.status = SpanStatus.OK
        except Exception as e:
            span.status = SpanStatus.ERROR
            span.add_event("exception", {"message": str(e)})
            raise
        finally:
            span.end_time = time.time()
            self._active_span = prev_span
            self._record_span(span)

    def _record_span(self, span: Span) -> None:
        """Span 기록 (동적 샘플링 적용)."""
        if span.trace_id not in self._traces:
            self._traces[span.trace_id] = []
        self._traces[span.trace_id].append(span)

    def get_trace_dag(self, trace_id: str) -> List[Dict[str, Any]]:
        """특정 Trace의 DAG 노드 목록을 반환합니다.

        Reviewer 기여: MACIE/ECHO 인과 그래프 입력용.
        """
        spans = self._traces.get(trace_id, [])
        return [s.to_dag_node() for s in spans]

    def export_sampled(self, drift_index: float = 0.0) -> List[Span]:
        """동적 샘플링을 적용하여 전송 대상 Span을 선별합니다."""
        sampled = []
        for trace_id, spans in self._traces.items():
            for span in spans:
                if self.sampler.should_sample(span, drift_index):
                    sampled.append(span)
        return sampled

    @property
    def trace_count(self) -> int:
        return len(self._traces)

    @property
    def total_spans(self) -> int:
        return sum(len(spans) for spans in self._traces.values())
