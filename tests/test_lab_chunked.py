"""Chunked prefill: correctness, and that it actually fixes the pathology."""

from __future__ import annotations

from dataclasses import replace

import pytest

from inferscope import MemorySink, Tracer
from inferscope.metrics import window_metrics
from inferscope.query import diagnose, latency_regime
from inferscope.trace import Trace
from inferscope_lab.config import EngineConfig
from inferscope_lab.engine import Engine
from inferscope_lab.pathologies import SCENARIOS, prefill_starvation, run_workload
from inferscope_lab.runner import FakeRunner


def traced(workload) -> Trace:
    tracer = Tracer(MemorySink(), autostart=False)
    run_workload(workload, tracer)
    tracer.flush()
    trace = Trace.from_sink(tracer.sink)
    tracer.close()
    return trace


def test_chunked_prefill_drains_and_generates_the_same_tokens() -> None:
    results = {}
    for chunked in (False, True):
        engine = Engine(
            FakeRunner(speedup=5000), None,
            EngineConfig(num_blocks=512, max_batch_size=8,
                         chunked_prefill=chunked, chunk_tokens=64),
        )
        for i in range(6):
            engine.add_request(f"r{i}", prompt_tokens=200, max_new_tokens=5)
        done = engine.run_until_idle()
        assert len(done) == 6
        assert all(r.generated == 5 for r in done)
        assert engine.allocator.used_blocks == 0
        results[chunked] = engine.stats()

    assert results[True]["prefill_iterations"] > results[False]["prefill_iterations"], (
        "chunking splits prefill across more iterations"
    )


def test_chunking_interleaves_decode_with_prefill() -> None:
    """The whole point: decode must keep running while a long prompt prefills."""
    engine = Engine(
        FakeRunner(speedup=5000), None,
        EngineConfig(num_blocks=1024, max_batch_size=8,
                     chunked_prefill=True, chunk_tokens=64),
    )
    engine.add_request("short", prompt_tokens=32, max_new_tokens=40)
    for _ in range(6):
        engine.step()          # get "short" decoding
    engine.add_request("long", prompt_tokens=512, max_new_tokens=2)
    before = engine.finished, next(r for r in engine.running if r.request_id == "short")
    generated_before = before[1].generated

    for _ in range(4):
        engine.step()
    assert before[1].generated > generated_before, (
        "the short request stopped emitting tokens while the long prompt prefilled"
    )
    engine.run_until_idle()


def test_a_chunked_prefill_covers_the_whole_prompt_exactly() -> None:
    engine = Engine(
        FakeRunner(speedup=5000), None,
        EngineConfig(num_blocks=512, max_batch_size=4,
                     chunked_prefill=True, chunk_tokens=48),
    )
    req = engine.add_request("r0", prompt_tokens=200, max_new_tokens=1)
    engine.run_until_idle()
    assert req.prefilled_tokens >= 200
    assert req.generated == 1


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_chunking_does_not_break_any_scenario(scenario: str) -> None:
    workload = SCENARIOS[scenario]()
    workload.config = replace(workload.config, chunked_prefill=True, chunk_tokens=256)
    engine = run_workload(workload, None)
    assert engine.stats()["finished"] == len(workload.arrivals)
    assert engine.allocator.used_blocks == 0, "leaked KV blocks"


def test_mid_prefill_requests_can_be_preempted() -> None:
    """They hold blocks, so leaving them out of victim selection deadlocks the pool."""
    workload = SCENARIOS["kv-thrash"]()
    workload.config = replace(workload.config, chunked_prefill=True, chunk_tokens=128)
    engine = run_workload(workload, None)
    assert engine.stats()["finished"] == len(workload.arrivals)


def test_run_workload_raises_rather_than_spinning_forever() -> None:
    workload = SCENARIOS["healthy"]()
    with pytest.raises(RuntimeError, match="did not drain"):
        run_workload(workload, None, max_iterations=3)


# -- the point of the whole exercise ---------------------------------------


def test_chunked_prefill_cures_the_starvation_it_was_chosen_for() -> None:
    baseline = traced(prefill_starvation())
    assert diagnose(baseline).pathology == "prefill-starvation"
    baseline_regime = latency_regime(baseline)

    fixed_workload = prefill_starvation()
    fixed_workload.config = replace(
        fixed_workload.config, chunked_prefill=True, chunk_tokens=512
    )
    fixed = traced(fixed_workload)

    assert diagnose(fixed).pathology == "healthy"
    fixed_regime = latency_regime(fixed)
    assert fixed_regime.tail_time_share < 0.05
    assert fixed_regime.tpot_max_ns < baseline_regime.tpot_max_ns / 4, (
        "the worst decode step must come down by a large factor"
    )
    # and it must not have been bought by wrecking batching
    assert window_metrics(fixed).mean_batch_size > (
        window_metrics(baseline).mean_batch_size * 0.8
    )


def test_the_obvious_fix_trades_one_pathology_for_another() -> None:
    """decode-priority removes the stalls and destroys TTFT. The tool says so."""
    workload = prefill_starvation()
    workload.config = replace(workload.config, policy="decode-priority")
    trace = traced(workload)

    assert latency_regime(trace).tail_time_share < 0.05, "stalls are indeed gone"
    assert diagnose(trace).pathology == "admission-starvation"
    assert window_metrics(trace).mean_batch_size < 4


def test_over_chunking_starves_admission() -> None:
    """Chunk size is a real trade-off, not a free win."""
    workload = prefill_starvation()
    workload.config = replace(
        workload.config, chunked_prefill=True, chunk_tokens=64
    )
    trace = traced(workload)
    assert latency_regime(trace).ttft_share > latency_regime(
        traced(prefill_starvation())
    ).ttft_share
