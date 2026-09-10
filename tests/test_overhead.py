"""Guards on instrumentation cost.

Thresholds here are deliberately loose -- roughly 5x the measured cost on the
development machine. They are not the benchmark (that lives in
``benchmarks/overhead.py``); they exist to catch an order-of-magnitude
regression, such as someone moving derivation onto the hot path, without
turning into a flaky test on a noisy CI box.
"""

from __future__ import annotations

import time

import pytest
from benchmarks.overhead import bench_micro, bench_throughput

from inferscope import EventKind as K
from inferscope import Tracer
from inferscope.sinks.null import NullSink


def _ns_per_call(fn, arg, calls: int = 50_000, rounds: int = 5) -> float:
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(calls):
            fn(arg)
        best = min(best, (time.perf_counter_ns() - t0) / calls)
    return best


@pytest.fixture()
def span():
    tracer = Tracer(NullSink(), capacity=1 << 30, autostart=False)
    try:
        yield tracer.trace_request("bench")
    finally:
        tracer.close()


def test_mark_stays_cheap(span) -> None:
    assert _ns_per_call(span.mark, K.QUEUED) < 500


def test_decode_step_stays_cheap(span) -> None:
    """The per-token path must not become more expensive than a generic mark."""
    assert _ns_per_call(span.decode_step, 8) < 500


def test_disabled_tracer_is_nearly_free() -> None:
    tracer = Tracer(NullSink(), enabled=False, autostart=False)
    try:
        s = tracer.trace_request("bench")
        assert _ns_per_call(s.mark, K.QUEUED) < 200
    finally:
        tracer.close()


def test_bench_micro_runs_and_reports_every_case() -> None:
    import benchmarks.overhead as mod

    mod.CALLS, mod.ROUNDS = 2000, 2  # keep the test fast
    results = bench_micro()
    names = [name for name, _ in results]
    assert "span.mark()" in names
    assert "span.decode_step() aggregate" in names
    assert all(ns > 0 for name, ns in results if name != "empty loop (baseline)")


def test_bench_throughput_runs_and_reports_a_ci() -> None:
    r = bench_throughput(qps=200, output_tokens=20, prompt_tokens=64,
                         batch_size=8, seconds=0.05, pairs=3)
    assert r["uninstrumented_rps"] > 0
    assert r["instrumented_rps"] > 0
    assert r["ci95"] >= 0
    assert r["hook_calls_per_request"] == 25
