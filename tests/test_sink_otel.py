"""OTel export: span shape, nesting, and what happens to unfinished requests."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from inferscope import EventKind as K  # noqa: E402
from inferscope import Tracer  # noqa: E402
from inferscope.sinks.otel import OTelSink  # noqa: E402
from inferscope_lab.pathologies import SCENARIOS, run_workload  # noqa: E402


@pytest.fixture()
def exported():
    """Yields (make_tracer, get_spans)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter


def spans_by_name(spans):
    out: dict[str, list] = {}
    for s in spans:
        out.setdefault(s.name, []).append(s)
    return out


def test_one_span_tree_per_request(exported) -> None:
    provider, exporter = exported
    t = Tracer(OTelSink(provider), autostart=False)
    with t.trace_request("r1", prompt_tokens=64) as span:
        span.mark(K.SCHEDULED)
        span.mark(K.PREFILL_START)
        span.mark(K.PREFILL_END, 64)
        for _ in range(4):
            span.decode_step(batch_size=2)
    t.close()

    named = spans_by_name(exporter.get_finished_spans())
    assert set(named) == {"inference.request", "queue", "prefill", "decode"}
    root = named["inference.request"][0]
    assert root.attributes["inferscope.request_id"] == "r1"
    assert root.attributes["inferscope.prompt_tokens"] == 64
    assert root.attributes["inferscope.output_tokens"] == 4
    assert root.attributes["inferscope.complete"] is True

    for child in named["prefill"] + named["decode"]:
        assert child.parent.span_id == root.context.span_id


def test_spans_carry_wall_clock_time(exported) -> None:
    """Monotonic timestamps would put every trace in 1970."""
    import time

    provider, exporter = exported
    before = time.time_ns()
    t = Tracer(OTelSink(provider), autostart=False)
    with t.trace_request("r1") as span:
        span.decode_step(batch_size=1)
    t.close()
    root = spans_by_name(exporter.get_finished_spans())["inference.request"][0]
    assert before <= root.start_time <= time.time_ns()


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_children_nest_within_their_root(scenario: str, exported) -> None:
    provider, exporter = exported
    t = Tracer(OTelSink(provider), autostart=False)
    engine = run_workload(SCENARIOS[scenario](), t)
    t.close()

    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.name == "inference.request"]
    assert len(roots) == len(engine.finished)

    by_trace: dict[int, list] = {}
    for s in spans:
        by_trace.setdefault(s.context.trace_id, []).append(s)

    for group in by_trace.values():
        root = next(s for s in group if s.name == "inference.request")
        for child in group:
            if child is root:
                continue
            assert root.start_time <= child.start_time
            assert child.end_time <= root.end_time


def test_preemption_becomes_span_events_and_a_requeue_span(exported) -> None:
    provider, exporter = exported
    t = Tracer(OTelSink(provider), autostart=False)
    engine = run_workload(SCENARIOS["kv-thrash"](), t)
    t.close()

    named = spans_by_name(exporter.get_finished_spans())
    assert len(named["requeue"]) == engine.preemptions, (
        "one requeue span per preemption"
    )
    assert len(named["prefill"]) == len(engine.finished) + engine.preemptions, (
        "a resumed request prefills again"
    )
    assert len(named["queue"]) == len(engine.finished), (
        "the initial queue is emitted exactly once per request"
    )

    events = [e.name for s in named["decode"] for e in s.events]
    assert "preempted" in events
    assert "resumed" in events


def test_stalls_become_span_events(exported) -> None:
    provider, exporter = exported
    t = Tracer(OTelSink(provider), autostart=False)
    run_workload(SCENARIOS["prefill-starvation"](), t)
    t.close()

    decode = spans_by_name(exporter.get_finished_spans())["decode"]
    stalls = [e for s in decode for e in s.events if e.name == "stall"]
    assert stalls, "the starvation workload must export stall events"
    assert max(e.attributes["inferscope.stall_ms"] for e in stalls) > 10


def test_a_failed_request_is_marked_as_an_error(exported) -> None:
    from opentelemetry.trace import StatusCode

    provider, exporter = exported
    t = Tracer(OTelSink(provider), autostart=False)
    with pytest.raises(ValueError):
        with t.trace_request("r1") as span:
            span.decode_step(batch_size=1)
            raise ValueError("boom")
    t.close()
    root = spans_by_name(exporter.get_finished_spans())["inference.request"][0]
    assert root.status.status_code is StatusCode.ERROR


def test_unfinished_requests_are_flushed_at_close_not_dropped(exported) -> None:
    provider, exporter = exported
    sink = OTelSink(provider)
    t = Tracer(sink, autostart=False)
    span = t.trace_request("never-finishes", prompt_tokens=32)
    span.mark(K.PREFILL_START)
    span.mark(K.PREFILL_END, 32)
    span.decode_step(batch_size=1)
    t.close()

    roots = spans_by_name(exporter.get_finished_spans())["inference.request"]
    assert len(roots) == 1
    assert roots[0].attributes["inferscope.complete"] is False


def test_pending_requests_are_capped(exported) -> None:
    """A request that never terminates must not pin memory forever."""
    provider, exporter = exported
    sink = OTelSink(provider, max_pending=5)
    t = Tracer(sink, autostart=False)
    for i in range(20):
        s = t.trace_request(f"r{i}")
        s.decode_step(batch_size=1)
        t.flush()
    assert sink.requests_abandoned >= 15
    assert len(sink._pending) <= 5
    t.close()
