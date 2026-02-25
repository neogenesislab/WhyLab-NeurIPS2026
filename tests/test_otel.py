# -*- coding: utf-8 -*-
"""C4 테스트 — OTel 동적 샘플링 + GenAI 시맨틱 컨벤션."""

import random
import time

import pytest

from engine.telemetry.otel_setup import (
    DynamicSampler,
    GenAIAttributes,
    Span,
    SpanKind,
    SpanStatus,
    WhyLabTracer,
)


class TestDynamicSampler:
    """동적 샘플링 검증."""

    def test_error_always_sampled(self):
        """에러 Span → 100% 전송."""
        sampler = DynamicSampler(base_rate=0.0)
        span = Span(trace_id="t1", span_id="s1", status=SpanStatus.ERROR)
        assert sampler.should_sample(span)

    def test_drift_alert_always_sampled(self):
        """DI > threshold → 100% 전송."""
        sampler = DynamicSampler(base_rate=0.0)
        span = Span(trace_id="t1", span_id="s1")
        assert sampler.should_sample(span, drift_index=0.5, drift_threshold=0.3)

    def test_audit_span_always_sampled(self):
        """감사 Span → 100% 전송."""
        sampler = DynamicSampler(base_rate=0.0)
        span = Span(trace_id="t1", span_id="s1", kind=SpanKind.AUDIT_RUN)
        assert sampler.should_sample(span)

    def test_normal_probabilistic(self):
        """정상 Span → base_rate 확률."""
        random.seed(42)
        sampler = DynamicSampler(base_rate=0.1)
        samples = 0
        for i in range(1000):
            span = Span(trace_id=f"t{i}", span_id=f"s{i}")
            if sampler.should_sample(span, drift_index=0.0):
                samples += 1
        # 10% ± 3% 이내
        assert 70 < samples < 130

    def test_stats_tracking(self):
        """통계 추적."""
        sampler = DynamicSampler(base_rate=0.0)
        span_err = Span(trace_id="t1", span_id="s1", status=SpanStatus.ERROR)
        span_ok = Span(trace_id="t2", span_id="s2")
        sampler.should_sample(span_err)
        sampler.should_sample(span_ok)
        assert sampler.stats["forced"] == 1
        assert sampler.stats["dropped"] == 1


class TestWhyLabTracer:
    """WhyLab 트레이서 검증."""

    def test_span_context_manager(self):
        """Span 컨텍스트 매니저."""
        tracer = WhyLabTracer()
        with tracer.start_span("test_op") as span:
            span.set_attribute("key", "value")
        assert tracer.total_spans == 1

    def test_parent_child_spans(self):
        """부모-자식 Span 관계."""
        tracer = WhyLabTracer()
        with tracer.start_span("parent", kind=SpanKind.AUDIT_RUN) as parent:
            with tracer.start_span("child", kind=SpanKind.LLM_CALL) as child:
                assert child.parent_span_id == parent.span_id
                assert child.trace_id == parent.trace_id

    def test_error_span(self):
        """에러 발생 시 Span 상태."""
        tracer = WhyLabTracer()
        with pytest.raises(ValueError):
            with tracer.start_span("failing_op") as span:
                raise ValueError("test error")
        # 에러 후에도 Span 기록됨
        assert tracer.total_spans == 1

    def test_dag_extraction(self):
        """Span → DAG 노드 변환 (MACIE/ECHO 입력)."""
        tracer = WhyLabTracer()
        with tracer.start_span("audit", attributes={
            GenAIAttributes.AGENT_ID: "agent_A",
        }) as span:
            trace_id = span.trace_id

        dag = tracer.get_trace_dag(trace_id)
        assert len(dag) == 1
        assert dag[0]["agent"] == "agent_A"
        assert dag[0]["node_id"] is not None


class TestGenAIAttributes:
    """GenAI 시맨틱 컨벤션 검증."""

    def test_span_with_genai_attrs(self):
        """GenAI 속성이 Span에 올바르게 기록."""
        tracer = WhyLabTracer()
        with tracer.start_span("llm_call", kind=SpanKind.LLM_CALL, attributes={
            GenAIAttributes.SYSTEM: "openai",
            GenAIAttributes.MODEL: "gpt-4o",
            GenAIAttributes.PROMPT_TOKENS: 150,
            GenAIAttributes.COMPLETION_TOKENS: 50,
            GenAIAttributes.AGENT_ROLE: "auditor",
        }) as span:
            pass

        assert span.attributes[GenAIAttributes.SYSTEM] == "openai"
        assert span.attributes[GenAIAttributes.PROMPT_TOKENS] == 150

    def test_audit_attributes(self):
        """WhyLab 감사 속성."""
        tracer = WhyLabTracer()
        with tracer.start_span("audit", attributes={
            GenAIAttributes.AUDIT_METHOD: "dml",
            GenAIAttributes.AUDIT_ATE: 15.3,
            GenAIAttributes.AUDIT_CONFIDENCE: 0.92,
            GenAIAttributes.AUDIT_DI: 0.15,
            GenAIAttributes.AUDIT_EVALUE: 3.2,
        }) as span:
            pass

        assert span.attributes[GenAIAttributes.AUDIT_ATE] == 15.3
        assert span.attributes[GenAIAttributes.AUDIT_EVALUE] == 3.2


class TestExportSampled:
    """동적 샘플링 + 내보내기 검증."""

    def test_export_with_drift(self):
        """DI 높을 때 더 많은 Span 내보내기."""
        tracer = WhyLabTracer(sampler=DynamicSampler(base_rate=0.0))
        # 정상 Span (전송 안 됨)
        with tracer.start_span("normal") as span:
            pass
        # DI=0일 때: 전송 0건
        low_drift = tracer.export_sampled(drift_index=0.0)
        # DI=0.5일 때: 전송됨
        high_drift = tracer.export_sampled(drift_index=0.5)
        assert len(high_drift) >= len(low_drift)
