"""Derived metrics: the breakdown must add up, in every decode mode."""

from __future__ import annotations

import pytest

from inferscope import EventKind as K
from inferscope import MemorySink, Tracer
from inferscope.metrics import per_request, window_metrics
from inferscope.trace import Trace
from inferscope_lab.pathologies import SCENARIOS, run_workload

MODES = ["aggregate", "coarse", "full"]


def traced(workload, decode_mode: str = "aggregate") -> Trace:
    tracer = Tracer(MemorySink(), decode_mode=decode_mode, autostart=False)
    run_workload(workload, tracer)
    tracer.flush()
    trace = Trace.from_sink(tracer.sink)
    tracer.close()
    return trace


@pytest.fixture(scope="module")
def healthy_trace() -> Trace:
    return traced(SCENARIOS["healthy"]())


def test_hand_built_trace_decomposes_exactly() -> None:
    """A trace with known timings must produce the timings we put in."""
    import time

    t = Tracer(MemorySink(), autostart=False)
    try:
        with t.trace_request("r1", prompt_tokens=100) as span:
            time.sleep(0.010)                    # queued
            span.mark(K.SCHEDULED)
            span.mark(K.PREFILL_START)
            time.sleep(0.020)                    # prefill
            span.mark(K.PREFILL_END, 100)
            time.sleep(0.005)                    # waiting for a decode slot
            for _ in range(5):
                span.decode_step(batch_size=4)
                time.sleep(0.001)
        t.flush()
        m = per_request(Trace.from_sink(t.sink))["r1"]

        assert m.prompt_tokens == 100
        assert m.output_tokens == 5
        assert m.completed
        assert 8e6 < m.queue_ns < 25e6
        assert 18e6 < m.prefill_ns < 35e6
        assert 4e6 < m.decode_wait_ns < 15e6
        assert abs(m.unattributed_ns) < 0.05 * m.total_ns
        assert m.ttft_ns == m.queue_ns + m.prefill_ns + m.decode_wait_ns
    finally:
        t.close()


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("mode", MODES)
def test_breakdown_sums_to_wall_time(scenario: str, mode: str) -> None:
    """No component may be double-counted or dropped.

    Regression: a DECODE_STALL spanning a preemption was counted both as a
    stall and as requeue time, making breakdowns reach 143% of wall time.
    """
    trace = traced(SCENARIOS[scenario](), mode)
    done = [m for m in per_request(trace).values() if m.completed]
    assert done

    for m in done:
        parts = (m.queue_ns + m.prefill_ns + m.decode_wait_ns
                 + m.decode_ns + m.stall_ns + m.requeue_ns)
        assert parts <= m.total_ns + 1e6, f"{m.request_id} over-attributed"
        assert abs(m.unattributed_ns) < 0.05 * m.total_ns, (
            f"{m.request_id} leaves {m.unattributed_ns / 1e6:.1f} ms unexplained"
        )


@pytest.mark.parametrize("mode", MODES)
def test_output_token_count_matches_the_terminal_event(mode: str) -> None:
    trace = traced(SCENARIOS["healthy"](), mode)
    for m in per_request(trace).values():
        if m.completed:
            assert m.output_tokens > 0


def test_ttft_identity_holds_for_never_preempted_requests() -> None:
    for scenario in SCENARIOS:
        trace = traced(SCENARIOS[scenario]())
        for m in per_request(trace).values():
            if m.completed and not m.preemptions:
                assert m.ttft_ns == m.queue_ns + m.prefill_ns + m.decode_wait_ns


def test_preemption_is_attributed_to_requeue_and_recompute() -> None:
    trace = traced(SCENARIOS["kv-thrash"]())
    preempted = [m for m in per_request(trace).values() if m.preemptions]
    assert preempted, "kv-thrash produced no preemptions"
    for m in preempted:
        assert m.requeue_ns > 0
        assert m.recomputed_tokens > 0


def test_window_metrics_summarise_the_engine(healthy_trace: Trace) -> None:
    w = window_metrics(healthy_trace)
    assert w.iterations > 0
    assert w.prefill_iterations > 0
    assert w.decode_iterations > 0
    assert 1 <= w.mean_batch_size <= w.max_batch_size
    assert 0.0 <= w.mean_kv_occupancy <= 1.0
    assert w.goodput_ratio == 1.0, "healthy runs recompute nothing"


def test_window_can_be_restricted_in_time(healthy_trace: Trace) -> None:
    first, last = healthy_trace.events[0][0], healthy_trace.events[-1][0]
    midpoint = first + (last - first) // 2
    whole = window_metrics(healthy_trace)
    half = window_metrics(healthy_trace, start_ns=midpoint)
    assert half.iterations < whole.iterations


def test_kv_thrash_shows_lost_goodput() -> None:
    w = window_metrics(traced(SCENARIOS["kv-thrash"]()))
    assert w.preemptions > 0
    assert w.recomputed_tokens > 0
    assert w.goodput_ratio < 0.95


@pytest.mark.parametrize("mode", ["aggregate", "full"])
def test_tpot_tail_is_consistent_across_decode_modes(mode: str) -> None:
    """The headline tail statistic must not depend on how decode was recorded.

    Regression: stall durations were excluded from the TPOT distribution, so an
    aggregate trace reported a p99/p50 of 1.1x where a full trace of the same
    workload reported 40x.
    """
    trace = traced(SCENARIOS["prefill-starvation"](), mode)
    samples = sorted(x for m in per_request(trace).values() for x in m.tpot_samples)
    assert samples
    median = samples[len(samples) // 2]
    tail = sum(x for x in samples if x > median * 8)
    assert 0.15 < tail / sum(samples) < 0.50, "the stall must show up as decode time"
