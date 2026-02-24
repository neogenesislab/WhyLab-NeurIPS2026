# -*- coding: utf-8 -*-
"""Tracing 모듈 테스트.

engine/tracing.py의 핵심 기능을 검증합니다:
1. Span/Trace 데이터 모델
2. TraceCollector 싱글톤
3. @trace_cell 데코레이터
4. @trace_llm 데코레이터
5. @trace_agent 데코레이터
6. JSON 파일 저장
"""

import json
import os
import time
import uuid
from pathlib import Path

import pytest

from engine.tracing import (
    Span,
    Trace,
    TraceCollector,
    trace_cell,
    trace_llm,
    trace_agent,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_collector(tmp_path):
    """각 테스트 전후로 TraceCollector를 리셋합니다."""
    TraceCollector.reset()
    os.environ["WHYLAB_TRACE_DIR"] = str(tmp_path / "traces")
    yield
    TraceCollector.reset()
    os.environ.pop("WHYLAB_TRACE_DIR", None)


# ──────────────────────────────────────────────
# Span/Trace 데이터 모델 테스트
# ──────────────────────────────────────────────

class TestDataModel:
    """Span/Trace 데이터 모델."""

    def test_span_default_values(self):
        """Span 기본값 확인."""
        span = Span()
        assert len(span.span_id) == 16
        assert span.kind == "INTERNAL"
        assert span.status == "OK"
        assert span.attributes == {}
        assert span.events == []

    def test_span_to_dict(self):
        """Span → dict 변환."""
        span = Span(name="test", kind="LLM")
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["kind"] == "LLM"
        assert isinstance(d, dict)

    def test_trace_to_dict(self):
        """Trace → dict 변환 (spans 포함)."""
        trace = Trace(name="pipeline")
        trace.spans.append(Span(name="cell:data"))
        d = trace.to_dict()
        assert d["name"] == "pipeline"
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "cell:data"


# ──────────────────────────────────────────────
# TraceCollector 테스트
# ──────────────────────────────────────────────

class TestTraceCollector:
    """TraceCollector 싱글톤 라이프사이클."""

    def test_singleton(self):
        """동일 인스턴스 반환."""
        a = TraceCollector.get_instance()
        b = TraceCollector.get_instance()
        assert a is b

    def test_reset(self):
        """reset 후 새 인스턴스 생성."""
        a = TraceCollector.get_instance()
        TraceCollector.reset()
        b = TraceCollector.get_instance()
        assert a is not b

    def test_trace_lifecycle(self):
        """start_trace → start_span → end_span → end_trace 전체 흐름."""
        c = TraceCollector.get_instance()
        trace = c.start_trace("test_pipeline")
        assert trace.name == "test_pipeline"

        span = c.start_span("cell:data", kind="CELL")
        assert span.trace_id == trace.trace_id

        c.end_span(span, status="OK", attributes={"rows": 100})
        assert span.status == "OK"
        assert span.attributes["rows"] == 100
        assert span.duration_ms >= 0

        result = c.end_trace()
        assert result is not None
        assert len(result.spans) == 1
        assert result.duration_ms >= 0

    def test_parent_span_tracking(self):
        """중첩 Span에서 parent_span_id가 올바르게 설정."""
        c = TraceCollector.get_instance()
        c.start_trace("nested")

        parent = c.start_span("parent")
        child = c.start_span("child")

        assert child.parent_span_id == parent.span_id
        c.end_span(child)
        c.end_span(parent)
        c.end_trace()

    def test_get_traces(self):
        """수집된 트레이스 조회."""
        c = TraceCollector.get_instance()
        c.start_trace("one")
        c.end_trace()
        c.start_trace("two")
        c.end_trace()

        traces = c.get_traces()
        assert len(traces) == 2
        assert traces[0]["name"] == "one"
        assert traces[1]["name"] == "two"

    def test_get_last_trace(self):
        """가장 최근 트레이스 반환."""
        c = TraceCollector.get_instance()
        assert c.get_last_trace() is None  # 비어있을 때

        c.start_trace("x")
        c.end_trace()
        last = c.get_last_trace()
        assert last["name"] == "x"

    def test_json_file_saved(self, tmp_path):
        """end_trace 시 JSON 파일 자동 저장."""
        c = TraceCollector.get_instance()
        c.start_trace("save_test")
        span = c.start_span("cell:test", kind="CELL")
        c.end_span(span)
        c.end_trace()

        trace_dir = tmp_path / "traces"
        files = list(trace_dir.glob("trace_*.json"))
        assert len(files) == 1

        with open(files[0], encoding="utf-8") as f:
            data = json.load(f)
        assert data["name"] == "save_test"
        assert len(data["spans"]) == 1


# ──────────────────────────────────────────────
# 데코레이터 테스트
# ──────────────────────────────────────────────

class TestTraceCellDecorator:
    """@trace_cell 데코레이터."""

    def test_cell_traced(self):
        """셀 실행이 Span으로 기록됨."""
        c = TraceCollector.get_instance()
        c.start_trace("decorator_test")

        class MockCell:
            name = "mock_cell"

            @trace_cell
            def execute(self, inputs):
                return {"key": "value"}

        cell = MockCell()
        result = cell.execute({"input": True})

        assert result == {"key": "value"}

        trace = c.end_trace()
        assert len(trace.spans) == 1
        span = trace.spans[0]
        assert span.name == "cell:mock_cell"
        assert span.kind == "CELL"
        assert span.status == "OK"
        assert "cell.output_keys" in span.attributes

    def test_cell_error_tracked(self):
        """셀 에러가 Span에 기록됨."""
        c = TraceCollector.get_instance()
        c.start_trace("error_test")

        class FailCell:
            name = "fail_cell"

            @trace_cell
            def execute(self, inputs):
                raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            FailCell().execute({})

        trace = c.end_trace()
        assert trace.spans[0].status == "ERROR"
        assert "ValueError" in trace.spans[0].attributes["error.type"]


class TestTraceLLMDecorator:
    """@trace_llm 데코레이터."""

    def test_llm_call_traced(self):
        """LLM 호출이 Span으로 기록됨."""
        c = TraceCollector.get_instance()
        c.start_trace("llm_test")

        class MockLLM:
            @trace_llm
            def generate(self, prompt, max_tokens=1024):
                return "Mock response text"

        llm = MockLLM()
        result = llm.generate("Tell me about causality", max_tokens=512)

        assert result == "Mock response text"

        trace = c.end_trace()
        span = trace.spans[0]
        assert span.name == "llm:generate"
        assert span.kind == "LLM"
        assert span.attributes["llm.prompt_length"] == len("Tell me about causality")
        assert span.attributes["llm.response_length"] == len("Mock response text")
        assert span.attributes["llm.max_tokens"] == 512
        assert span.attributes["llm.success"] is True

    def test_llm_none_response(self):
        """LLM이 None 반환 시 success=False."""
        c = TraceCollector.get_instance()
        c.start_trace("llm_none")

        class MockLLM:
            @trace_llm
            def generate(self, prompt, max_tokens=1024):
                return None

        MockLLM().generate("test")
        trace = c.end_trace()
        assert trace.spans[0].attributes["llm.success"] is False


class TestTraceAgentDecorator:
    """@trace_agent 데코레이터."""

    def test_agent_traced(self):
        """에이전트 호출이 Span으로 기록됨."""
        c = TraceCollector.get_instance()
        c.start_trace("agent_test")

        class MockAdvocate:
            @trace_agent("advocate")
            def gather_evidence(self, results):
                return [{"evidence": "test1"}, {"evidence": "test2"}]

        result = MockAdvocate().gather_evidence({})
        assert len(result) == 2

        trace = c.end_trace()
        span = trace.spans[0]
        assert span.name == "agent:advocate"
        assert span.kind == "AGENT"
        assert span.attributes["agent.role"] == "advocate"
        assert span.attributes["agent.output_count"] == 2


# ──────────────────────────────────────────────
# 통합 테스트
# ──────────────────────────────────────────────

class TestIntegration:
    """DebateCell 수준의 다중 Span 통합."""

    def test_full_debate_tracing(self):
        """여러 데코레이터가 하나의 Trace에 올바르게 쌓임."""
        c = TraceCollector.get_instance()
        c.start_trace("debate_integration")

        class MockAdvocate:
            @trace_agent("advocate")
            def gather(self, r):
                return [1, 2, 3]

        class MockCritic:
            @trace_agent("critic")
            def challenge(self, r):
                return [4]

        class MockLLM:
            @trace_llm
            def generate(self, prompt, max_tokens=2048):
                return "LLM output"

        # 시뮬레이션
        MockAdvocate().gather({})
        MockCritic().challenge({})
        MockLLM().generate("prompt1")
        MockLLM().generate("prompt2")

        trace = c.end_trace()
        assert len(trace.spans) == 4

        kinds = [s.kind for s in trace.spans]
        assert kinds == ["AGENT", "AGENT", "LLM", "LLM"]

        names = [s.name for s in trace.spans]
        assert "agent:advocate" in names
        assert "agent:critic" in names
