"""Each scenario must produce the pathology it claims to.

These are the tests that keep the Phase 3 diagnostic queries honest: if a
scenario stops exhibiting its signature, a query that still "detects" it is
detecting noise.
"""

from __future__ import annotations

import pytest

from inferscope import EventKind as K
from inferscope import MemorySink, Tracer
from inferscope_lab.pathologies import (
    SCENARIOS,
    batch_starvation,
    healthy,
    kv_thrash,
    prefill_starvation,
    run_workload,
)


def trace(workload, decode_mode: str = "aggregate"):
    tracer = Tracer(MemorySink(), decode_mode=decode_mode, autostart=False)
    engine = run_workload(workload, tracer)
    tracer.flush()
    events = tracer.sink.snapshot()
    names = dict(tracer.sink.names)
    tracer.close()
    return engine, events, names


def stalls_ms(events, names, prefix: str = "") -> list[float]:
    return [
        e[4] / 1e6
        for e in events
        if e[1] == K.DECODE_STALL and names.get(e[2], "").startswith(prefix)
    ]


def mean_decode_batch(events) -> float:
    sizes = [e[5] for e in events if e[1] == K.BATCH and e[5] > 0]
    return sum(sizes) / len(sizes) if sizes else 0.0


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_drains_and_finishes_its_requests(name: str) -> None:
    workload = SCENARIOS[name]()
    engine, _events, _names = trace(workload)
    assert engine.stats()["finished"] == len(workload.arrivals)
    assert engine.allocator.used_blocks == 0, "leaked KV blocks"


def test_healthy_has_no_preemption_and_no_long_stalls() -> None:
    engine, events, names = trace(healthy())
    assert engine.preemptions == 0
    assert engine.stats()["recomputed_tokens"] == 0
    worst = max(stalls_ms(events, names), default=0.0)
    assert worst < 10.0, f"healthy run stalled for {worst:.1f} ms"


def test_prefill_starvation_stalls_decoding_requests() -> None:
    """The short requests should stop emitting tokens while the burst prefills.

    End-to-end latency does *not* show this -- p99/p50 over request duration
    stays near 1.1x -- which is exactly why the tool records per-token timing.
    """
    engine, events, names = trace(prefill_starvation())
    short_stalls = stalls_ms(events, names, prefix="r")
    assert short_stalls, "no stalls recorded at all"
    assert max(short_stalls) > 10.0, (
        f"expected a long stall from the prompt burst, worst was {max(short_stalls):.1f} ms"
    )
    assert engine.preemptions == 0, "this scenario is about scheduling, not KV pressure"


def test_prefill_starvation_is_invisible_in_end_to_end_latency() -> None:
    """Guards the premise of the demo, not just its conclusion."""
    engine, _events, _names = trace(prefill_starvation())
    latencies = sorted((r.finish_ns - r.arrival_ns) for r in engine.finished)
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[max(0, int(len(latencies) * 0.99) - 1)]
    assert p99 / p50 < 3.0, (
        "request-level latency should look unremarkable here; if it does not, "
        "the scenario no longer demonstrates what it claims"
    )


def test_kv_thrash_preempts_and_recomputes() -> None:
    engine, events, _names = trace(kv_thrash())
    assert engine.preemptions > 0
    assert engine.stats()["recomputed_tokens"] > 0
    kinds = {e[1] for e in events}
    assert K.PREEMPTED in kinds
    assert K.KV_EVICT in kinds
    assert K.RESUMED in kinds


def test_batch_starvation_keeps_batches_far_below_capacity() -> None:
    workload = batch_starvation()
    engine, events, _names = trace(workload)
    mean = mean_decode_batch(events)
    assert mean < workload.config.max_batch_size / 2, (
        f"batches averaged {mean:.1f} of a possible {workload.config.max_batch_size}"
    )
    assert engine.preemptions == 0


def test_healthy_batches_are_fuller_than_starved_ones() -> None:
    _e1, healthy_events, _n1 = trace(healthy())
    _e2, starved_events, _n2 = trace(batch_starvation())
    assert mean_decode_batch(healthy_events) > mean_decode_batch(starved_events)


def test_coarse_mode_hides_the_starvation_stall() -> None:
    """Documents the cost of the cheap mode, so the trade stays deliberate."""
    _e1, aggregate_events, names = trace(prefill_starvation(), decode_mode="aggregate")
    _e2, coarse_events, _n = trace(prefill_starvation(), decode_mode="coarse")
    assert max(stalls_ms(aggregate_events, names, "r"), default=0) > 10.0
    assert not [e for e in coarse_events if e[1] == K.DECODE_STALL]
