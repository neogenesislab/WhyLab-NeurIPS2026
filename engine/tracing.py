# -*- coding: utf-8 -*-
"""WhyLab 경량 Tracing 모듈 — 제로 의존성 파이프라인 추적.

외부 SaaS(LangFuse, Phoenix) 없이, Python stdlib만으로
OpenTelemetry 호환 가능한 Span/Trace 구조를 제공합니다.

사용법:
    from engine.tracing import trace_cell, trace_llm, TraceCollector

    @trace_cell
    def execute(self, inputs):
        ...

    @trace_llm
    def generate(self, prompt, max_tokens=2048):
        ...

    # 수집된 트레이스 조회
    traces = TraceCollector.get_instance().get_traces()
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("whylab.tracing")


# ──────────────────────────────────────────────
# 데이터 모델 (OpenTelemetry Span 호환)
# ──────────────────────────────────────────────

@dataclass
class Span:
    """단일 작업 단위의 추적 레코드."""

    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    name: str = ""
    kind: str = "INTERNAL"  # INTERNAL | LLM | CELL | AGENT
    start_time: str = ""
    end_time: str = ""
    duration_ms: float = 0.0
    status: str = "OK"  # OK | ERROR
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리로 변환."""
        return asdict(self)


@dataclass
class Trace:
    """하나의 파이프라인 실행을 나타내는 트레이스."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:32])
    name: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_ms: float = 0.0
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리로 변환."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }


# ──────────────────────────────────────────────
# TraceCollector — 싱글톤 트레이스 수집기
# ──────────────────────────────────────────────

class TraceCollector:
    """파이프라인 실행 트레이스를 수집하고 저장하는 싱글톤."""

    _instance: Optional[TraceCollector] = None

    def __init__(self) -> None:
        self._traces: List[Trace] = []
        self._active_trace: Optional[Trace] = None
        self._active_spans: List[Span] = []
        self._log_dir = Path(
            os.environ.get("WHYLAB_TRACE_DIR", "logs/traces")
        )

    @classmethod
    def get_instance(cls) -> TraceCollector:
        """싱글톤 인스턴스 반환."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """테스트용 리셋."""
        cls._instance = None

    # ── Trace 라이프사이클 ──

    def start_trace(self, name: str = "pipeline") -> Trace:
        """새로운 트레이스를 시작합니다."""
        trace = Trace(
            name=name,
            start_time=_now_iso(),
        )
        self._active_trace = trace
        logger.debug("🔍 Trace 시작: %s [%s]", name, trace.trace_id[:8])
        return trace

    def end_trace(self) -> Optional[Trace]:
        """현재 활성 트레이스를 종료하고 저장합니다."""
        if self._active_trace is None:
            return None

        trace = self._active_trace
        trace.end_time = _now_iso()
        trace.duration_ms = _duration_ms(trace.start_time, trace.end_time)
        self._traces.append(trace)
        self._active_trace = None

        logger.debug(
            "✅ Trace 종료: %s [%s] %.1fms, %d spans",
            trace.name, trace.trace_id[:8],
            trace.duration_ms, len(trace.spans),
        )

        # 자동 파일 저장
        self._save_trace(trace)
        return trace

    # ── Span 라이프사이클 ──

    def start_span(
        self,
        name: str,
        kind: str = "INTERNAL",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """새로운 Span을 시작합니다."""
        trace_id = self._active_trace.trace_id if self._active_trace else ""
        parent_id = self._active_spans[-1].span_id if self._active_spans else None

        span = Span(
            trace_id=trace_id,
            parent_span_id=parent_id,
            name=name,
            kind=kind,
            start_time=_now_iso(),
            attributes=attributes or {},
        )
        self._active_spans.append(span)
        return span

    def end_span(
        self,
        span: Span,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Span을 종료하고 부모 Trace에 추가합니다."""
        span.end_time = _now_iso()
        span.duration_ms = _duration_ms(span.start_time, span.end_time)
        span.status = status
        if attributes:
            span.attributes.update(attributes)

        # 스택에서 제거
        if self._active_spans and self._active_spans[-1].span_id == span.span_id:
            self._active_spans.pop()

        # 활성 Trace에 추가
        if self._active_trace is not None:
            self._active_trace.spans.append(span)

        return span

    # ── 조회 ──

    def get_traces(self) -> List[Dict[str, Any]]:
        """수집된 모든 트레이스를 딕셔너리 리스트로 반환합니다."""
        return [t.to_dict() for t in self._traces]

    def get_last_trace(self) -> Optional[Dict[str, Any]]:
        """가장 최근 트레이스를 반환합니다."""
        if not self._traces:
            return None
        return self._traces[-1].to_dict()

    # ── 저장 ──

    def _save_trace(self, trace: Trace) -> Optional[Path]:
        """트레이스를 JSON 파일로 저장합니다."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            filename = f"trace_{trace.trace_id[:8]}_{trace.name}.json"
            filepath = self._log_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug("💾 Trace 저장: %s", filepath)
            return filepath
        except Exception as e:
            logger.warning("⚠️ Trace 저장 실패: %s", e)
            return None


# ──────────────────────────────────────────────
# 데코레이터
# ──────────────────────────────────────────────

def trace_cell(func):
    """셀 실행을 추적하는 데코레이터.

    사용법:
        class MyCell(BaseCell):
            @trace_cell
            def execute(self, inputs):
                ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        collector = TraceCollector.get_instance()

        # 셀 이름 추출 (self.name 또는 클래스명)
        cell_name = "unknown_cell"
        if args and hasattr(args[0], "name"):
            cell_name = args[0].name
        elif args and hasattr(args[0], "__class__"):
            cell_name = args[0].__class__.__name__

        span = collector.start_span(
            name=f"cell:{cell_name}",
            kind="CELL",
            attributes={"cell.name": cell_name},
        )

        try:
            result = func(*args, **kwargs)
            collector.end_span(span, status="OK", attributes={
                "cell.output_keys": list(result.keys()) if isinstance(result, dict) else [],
            })
            return result
        except Exception as e:
            collector.end_span(span, status="ERROR", attributes={
                "error.type": type(e).__name__,
                "error.message": str(e)[:200],
            })
            raise

    return wrapper


def trace_llm(func):
    """LLM 호출을 추적하는 데코레이터.

    프롬프트 길이, 응답 길이, 지연시간, 추정 토큰 수를 기록합니다.

    사용법:
        class GeminiClient:
            @trace_llm
            def generate(self, prompt, max_tokens=2048):
                ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        collector = TraceCollector.get_instance()

        # 프롬프트 추출 (첫 번째 positional arg 또는 'prompt' kwarg)
        prompt = ""
        if len(args) > 1:
            prompt = str(args[1])
        elif "prompt" in kwargs:
            prompt = str(kwargs["prompt"])

        max_tokens = kwargs.get("max_tokens", 2048)

        span = collector.start_span(
            name="llm:generate",
            kind="LLM",
            attributes={
                "llm.prompt_length": len(prompt),
                "llm.prompt_tokens_est": len(prompt) // 4,  # 대략적 토큰 추정
                "llm.max_tokens": max_tokens,
            },
        )

        try:
            result = func(*args, **kwargs)

            response_text = str(result) if result else ""
            collector.end_span(span, status="OK", attributes={
                "llm.response_length": len(response_text),
                "llm.response_tokens_est": len(response_text) // 4,
                "llm.success": result is not None,
            })
            return result
        except Exception as e:
            collector.end_span(span, status="ERROR", attributes={
                "llm.success": False,
                "error.type": type(e).__name__,
                "error.message": str(e)[:200],
            })
            raise

    return wrapper


def trace_agent(agent_role: str):
    """에이전트 호출을 추적하는 데코레이터.

    사용법:
        class AdvocateAgent:
            @trace_agent("advocate")
            def gather_evidence(self, results):
                ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = TraceCollector.get_instance()
            span = collector.start_span(
                name=f"agent:{agent_role}",
                kind="AGENT",
                attributes={"agent.role": agent_role},
            )

            try:
                result = func(*args, **kwargs)
                result_count = len(result) if isinstance(result, list) else 1
                collector.end_span(span, status="OK", attributes={
                    "agent.output_count": result_count,
                })
                return result
            except Exception as e:
                collector.end_span(span, status="ERROR", attributes={
                    "error.type": type(e).__name__,
                    "error.message": str(e)[:200],
                })
                raise

        return wrapper
    return decorator


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def _now_iso() -> str:
    """현재 시각을 ISO 8601 형식으로 반환."""
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(start_iso: str, end_iso: str) -> float:
    """두 ISO 타임스탬프 사이의 밀리초 차이를 계산."""
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        return (end - start).total_seconds() * 1000
    except (ValueError, TypeError):
        return 0.0
