"""Measure what inferscope costs the engine it observes.

An observability library that is expensive is a library nobody turns on in
production, so this number is the project's credibility. Two modes:

``--micro``
    Nanoseconds per instrumentation call, measured with the buffer drained
    between rounds so we time the call and not the allocator. Reports the
    minimum across rounds (the least-noisy estimate of true cost).

``--qps N``
    Throughput delta. A synthetic driver replays the exact hook-call pattern of
    a serving workload -- N requests/s, each with a prefill and a decode loop --
    against an instrumented and an uninstrumented engine stub, and reports the
    difference in requests completed per second.

    This measures *instrumentation cost*, not real GPU serving. It is run on a
    stub precisely so that the model's own cost does not swamp the signal we
    are trying to measure; the end-to-end number on a real model lives in
    ``benchmarks/e2e.py`` once ``inferscope_lab`` exists.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

from inferscope import EventKind as K
from inferscope import Tracer
from inferscope.sinks.null import NullSink

ROUNDS = 7
CALLS = 200_000

#: Two-sided 95% t critical values by degrees of freedom, so the benchmark
#: does not need scipy just to put an error bar on one number.
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
        9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13,
        16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09, 20: 2.09, 24: 2.06, 29: 2.05}


def _time_calls(setup: Callable[[], Any], body: Callable[[Any, int], None],
                calls: int = CALLS, rounds: int = ROUNDS) -> float:
    """Return nanoseconds per call, minimum across rounds."""
    best = float("inf")
    for _ in range(rounds):
        ctx = setup()
        t0 = time.perf_counter_ns()
        body(ctx, calls)
        elapsed = time.perf_counter_ns() - t0
        best = min(best, elapsed / calls)
    return best


def _drained_tracer(**kwargs: Any) -> Tracer:
    """A tracer with a null sink and no flush thread -- pure hot-path cost."""
    return Tracer(NullSink(), capacity=1 << 30, autostart=False, **kwargs)


def bench_micro() -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []

    # Baseline: an empty loop, so every number below is net of loop overhead.
    def loop_body(_ctx: Any, n: int) -> None:
        for _ in range(n):
            pass

    baseline = _time_calls(lambda: None, loop_body)
    results.append(("empty loop (baseline)", baseline))

    def mark_setup() -> Any:
        t = _drained_tracer()
        span = t.trace_request("bench")
        t.collector.buffer().events.clear()
        return span

    def mark_body(span: Any, n: int) -> None:
        mark = span.mark
        queued = K.QUEUED
        for _ in range(n):
            mark(queued)

    results.append(("span.mark()", _time_calls(mark_setup, mark_body) - baseline))

    def mark_payload_body(span: Any, n: int) -> None:
        mark = span.mark
        for _ in range(n):
            mark(K.PREFILL_END, 512, 0, 3)

    results.append(
        ("span.mark(kind, a, b, batch)", _time_calls(mark_setup, mark_payload_body) - baseline)
    )

    def step_body(span: Any, n: int) -> None:
        step = span.decode_step
        for _ in range(n):
            step(8)

    results.append(
        ("span.decode_step() aggregate", _time_calls(mark_setup, step_body) - baseline)
    )

    def coarse_setup() -> Any:
        t = _drained_tracer(decode_mode="coarse")
        span = t.trace_request("bench")
        t.collector.buffer().events.clear()
        return span

    results.append(
        ("span.decode_step() coarse", _time_calls(coarse_setup, step_body) - baseline)
    )

    def full_setup() -> Any:
        t = _drained_tracer(decode_mode="full")
        span = t.trace_request("bench")
        t.collector.buffer().events.clear()
        return span

    results.append(
        ("span.decode_step() full", _time_calls(full_setup, step_body) - baseline)
    )

    def churn_body(span: Any, n: int) -> None:
        """Worst case: batch size changes every step, so every step emits a run."""
        step = span.decode_step
        for i in range(n):
            step(i & 7)

    results.append(
        ("span.decode_step() run churn", _time_calls(mark_setup, churn_body) - baseline)
    )

    def disabled_setup() -> Any:
        t = Tracer(NullSink(), enabled=False, autostart=False)
        return t.trace_request("bench")

    results.append(
        ("span.mark() when disabled", _time_calls(disabled_setup, mark_body) - baseline)
    )

    def tracer_setup() -> Any:
        t = _drained_tracer()
        t.collector.buffer().events.clear()
        return t

    def request_body(t: Any, n: int) -> None:
        for _ in range(n):
            with t.trace_request("bench") as span:
                span.mark(K.QUEUED)
                span.mark(K.PREFILL_START)
                span.mark(K.PREFILL_END, 512)

    results.append(
        ("full request span (5 events)", _time_calls(tracer_setup, request_body, calls=50_000)
         - baseline)
    )

    def batch_body(t: Any, n: int) -> None:
        members = [f"r{i}" for i in range(8)]
        for _ in range(n):
            t.record_batch("b", members, prefill_tokens=512, decode_tokens=8)

    results.append(
        ("record_batch(8 members)", _time_calls(tracer_setup, batch_body, calls=50_000) - baseline)
    )

    return results


class _EngineStub:
    """Stands in for an inference engine: burns a calibrated amount of CPU."""

    __slots__ = ("work",)

    def __init__(self, work: int) -> None:
        self.work = work

    def step(self) -> int:
        total = 0
        for i in range(self.work):
            total += i * i
        return total


def _calibrate(target_step_ns: float) -> _EngineStub:
    """Size the stub so one step costs ``target_step_ns``.

    Without this the stub would be far cheaper than a real decode step and the
    overhead ratio would be flattering to us by an order of magnitude.
    """
    probe = _EngineStub(work=1000)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(200):
            probe.step()
        best = min(best, (time.perf_counter_ns() - t0) / 200)
    ns_per_unit = best / 1000
    return _EngineStub(work=max(1, round(target_step_ns / ns_per_unit)))


def bench_throughput(qps: int, output_tokens: int, prompt_tokens: int,
                     batch_size: int, seconds: float, pairs: int = 12) -> dict[str, Any]:
    """Instrumented vs uninstrumented requests/sec at a calibrated operating point.

    The synthetic engine is first calibrated so that the *uninstrumented*
    pipeline sustains ``qps`` requests/s at ``output_tokens`` tokens each --
    that is, so each step costs what it would have to cost for a real engine to
    hit that number. Only then do we turn instrumentation on and measure what
    it takes away.
    """
    steps_per_request = 1 + output_tokens
    target_step_ns = 1e9 / (qps * steps_per_request)
    engine = _calibrate(target_step_ns)

    def run(tracer: Tracer | None) -> float:
        deadline = time.perf_counter() + seconds
        completed = 0
        i = 0
        step = engine.step
        while time.perf_counter() < deadline:
            i += 1
            if tracer is None:
                step()
                for _ in range(output_tokens):
                    step()
            else:
                with tracer.trace_request(f"r{i}", prompt_tokens=prompt_tokens) as span:
                    span.mark(K.QUEUED)
                    span.mark(K.PREFILL_START)
                    step()
                    span.mark(K.PREFILL_END, prompt_tokens)
                    decode_step = span.decode_step
                    for _ in range(output_tokens):
                        step()
                        decode_step(batch_size)
            completed += 1
        return completed / seconds

    # Paired sampling: each pair runs the baseline and the instrumented
    # pipeline back to back, and we take the CI over per-pair deltas. Machine
    # drift (thermal, other processes) moves both halves of a pair together and
    # so cancels, which matters here because the drift is several times larger
    # than the effect we are trying to resolve.
    tracer = Tracer(NullSink(), capacity=1 << 22, flush_interval=0.05)
    deltas: list[float] = []
    samples_off: list[float] = []
    samples_on: list[float] = []
    try:
        run(None)  # warm up
        for _ in range(pairs):
            off_i = run(None)
            on_i = run(tracer)
            samples_off.append(off_i)
            samples_on.append(on_i)
            deltas.append((off_i - on_i) / off_i * 100.0)
    finally:
        tracer.close()

    mean = statistics.fmean(deltas)
    half_width = 0.0
    if len(deltas) > 1:
        sem = statistics.stdev(deltas) / math.sqrt(len(deltas))
        half_width = _T95.get(len(deltas) - 1, 1.96) * sem

    return {
        "target_qps": qps,
        "calibrated_step_ns": target_step_ns,
        "stub_work": engine.work,
        "uninstrumented_rps": statistics.median(samples_off),
        "instrumented_rps": statistics.median(samples_on),
        "overhead_pct": mean,
        "ci95": half_width,
        "pairs": len(deltas),
        "hook_calls_per_request": 5 + output_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro", action="store_true", help="per-call nanosecond costs")
    parser.add_argument("--qps", type=int, default=0, help="synthetic throughput delta")
    parser.add_argument("--output-tokens", type=int, default=200)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--pairs", type=int, default=12,
                        help="paired baseline/instrumented runs for the CI")
    args = parser.parse_args()

    if not args.micro and not args.qps:
        parser.error("pass --micro and/or --qps N")

    if args.micro:
        print(f"\n  per-call cost (min of {ROUNDS} rounds x {CALLS} calls,"
              " net of loop overhead)\n")
        for name, ns in bench_micro():
            print(f"    {name:<32} {ns:8.1f} ns")

    if args.qps:
        r = bench_throughput(args.qps, args.output_tokens, args.prompt_tokens,
                             args.batch_size, args.seconds, args.pairs)
        print("\n  synthetic throughput delta"
              f" ({r['hook_calls_per_request']} hook calls/request,"
              f" {args.output_tokens} output tokens)")
        print(f"  engine stub calibrated to {r['calibrated_step_ns']:.0f} ns/step"
              f" so the baseline sustains {args.qps} req/s\n")
        print(f"    uninstrumented   {r['uninstrumented_rps']:10.1f} req/s (median)")
        print(f"    instrumented     {r['instrumented_rps']:10.1f} req/s (median)")
        print(f"    overhead         {r['overhead_pct']:10.2f} %"
              f"  +/- {r['ci95']:.2f} (95% CI, {r['pairs']} paired runs)")
    print()


if __name__ == "__main__":
    main()
