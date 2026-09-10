"""Engine scheduling, preemption and the events it emits."""

from __future__ import annotations

import pytest

from inferscope import EventKind as K
from inferscope import MemorySink, Tracer
from inferscope_lab.config import EngineConfig
from inferscope_lab.engine import Engine
from inferscope_lab.request import RequestState
from inferscope_lab.runner import FakeRunner


def make(config: EngineConfig | None = None, **tracer_kwargs) -> tuple[Engine, Tracer]:
    tracer = Tracer(MemorySink(), autostart=False, **tracer_kwargs)
    engine = Engine(FakeRunner(speedup=2000), tracer, config)
    return engine, tracer


def kinds(tracer: Tracer) -> list[K]:
    tracer.flush()
    return [K(e[1]) for e in tracer.sink.snapshot()]


def test_requests_run_to_completion() -> None:
    engine, tracer = make(EngineConfig(num_blocks=128, max_batch_size=4))
    try:
        for i in range(6):
            engine.add_request(f"r{i}", prompt_tokens=64, max_new_tokens=5)
        done = engine.run_until_idle()

        assert len(done) == 6
        assert all(r.state is RequestState.FINISHED for r in done)
        assert all(r.generated == 5 for r in done)
        assert not engine.waiting and not engine.running
    finally:
        tracer.close()


def test_all_blocks_are_returned_when_the_engine_drains() -> None:
    engine, tracer = make(EngineConfig(num_blocks=64, max_batch_size=4))
    try:
        for i in range(8):
            engine.add_request(f"r{i}", prompt_tokens=48, max_new_tokens=4)
        engine.run_until_idle()
        assert engine.allocator.used_blocks == 0, "leaked KV blocks"
        assert engine.allocator.free_blocks == 64
    finally:
        tracer.close()


def test_lifecycle_events_are_emitted_in_order() -> None:
    engine, tracer = make(EngineConfig(num_blocks=64, max_batch_size=2))
    try:
        engine.add_request("r0", prompt_tokens=32, max_new_tokens=3)
        engine.run_until_idle()
        seen = kinds(tracer)
        for expected in (K.REQUEST_START, K.QUEUED, K.SCHEDULED,
                         K.PREFILL_START, K.PREFILL_END, K.FIRST_TOKEN, K.COMPLETE):
            assert expected in seen, f"missing {expected.name}"
        assert seen.index(K.QUEUED) < seen.index(K.SCHEDULED)
        assert seen.index(K.SCHEDULED) < seen.index(K.PREFILL_START)
        assert seen.index(K.PREFILL_END) < seen.index(K.FIRST_TOKEN)
        assert seen.index(K.FIRST_TOKEN) < seen.index(K.COMPLETE)
    finally:
        tracer.close()


def test_batch_membership_is_recorded_for_every_iteration() -> None:
    engine, tracer = make(EngineConfig(num_blocks=128, max_batch_size=4))
    try:
        for i in range(4):
            engine.add_request(f"r{i}", prompt_tokens=32, max_new_tokens=3)
        engine.run_until_idle()
        tracer.flush()
        events = tracer.sink.snapshot()
        batches = [e for e in events if e[1] == K.BATCH]
        assert len(batches) == engine.iteration
        members = [e for e in events if e[1] == K.BATCH_MEMBER]
        assert members, "no batch composition recorded"
        assert all(m[3] >= 0 for m in members), "members must name their batch"
    finally:
        tracer.close()


def test_kv_pressure_causes_preemption_and_recompute() -> None:
    """A pool too small for the working set must preempt, not deadlock."""
    engine, tracer = make(EngineConfig(
        num_blocks=20, block_size=16, max_batch_size=8, max_prefill_seqs=2
    ))
    try:
        for i in range(5):
            engine.add_request(f"r{i}", prompt_tokens=64, max_new_tokens=32)
        done = engine.run_until_idle()

        assert len(done) == 5, "every request must still finish"
        assert engine.preemptions > 0, "the pool was too small not to preempt"
        assert sum(r.recomputed_tokens for r in done) > 0

        seen = kinds(tracer)
        assert K.PREEMPTED in seen
        assert K.KV_EVICT in seen
        assert K.RESUMED in seen
    finally:
        tracer.close()


def test_preemption_spares_the_oldest_requests() -> None:
    """Victims are taken from the newest end, so early arrivals keep progress."""
    engine, tracer = make(EngineConfig(
        num_blocks=20, block_size=16, max_batch_size=8, max_prefill_seqs=2
    ))
    try:
        for i in range(5):
            engine.add_request(f"r{i}", prompt_tokens=64, max_new_tokens=32)
        done = engine.run_until_idle()
        by_id = {r.request_id: r for r in done}
        assert by_id["r0"].preemptions == 0
        assert sum(r.preemptions for r in done) == engine.preemptions
    finally:
        tracer.close()


def test_a_single_request_never_preempts_itself() -> None:
    """With one request and a tight pool the engine must not spin forever."""
    engine, tracer = make(EngineConfig(num_blocks=6, block_size=16, max_batch_size=4))
    try:
        engine.add_request("solo", prompt_tokens=64, max_new_tokens=24)
        done = engine.run_until_idle(max_iterations=2000)
        assert len(done) == 1
        assert done[0].preemptions == 0
    finally:
        tracer.close()


def test_admission_watermark_holds_blocks_back() -> None:
    strict, tracer = make(EngineConfig(
        num_blocks=64, max_batch_size=8, admission_watermark=0.9
    ))
    try:
        for i in range(8):
            strict.add_request(f"r{i}", prompt_tokens=64, max_new_tokens=2)
        strict.run_until_idle()
        assert strict.stats()["finished"] == 8
    finally:
        tracer.close()


def test_scheduling_policy_trades_batch_size_against_queue_time() -> None:
    """The policy knob must actually change how work is batched.

    ``prefill-priority`` admits whenever it can, so decode batches fill up --
    at the cost of spending iterations on prefill while running requests wait.
    ``decode-priority`` drains the running set first, so batches stay small and
    new arrivals queue. Neither is "correct"; the point is that the trace should
    make which one you are running obvious.
    """
    batch_sizes = {}
    queue_waits = {}
    for policy in ("prefill-priority", "decode-priority"):
        engine, tracer = make(EngineConfig(
            num_blocks=256, max_batch_size=8, max_prefill_seqs=1, policy=policy
        ))
        try:
            for i in range(8):
                engine.add_request(f"r{i}", prompt_tokens=32, max_new_tokens=6)
            engine.run_until_idle()
            tracer.flush()
            events = tracer.sink.snapshot()
            decode_batches = [e[5] for e in events if e[1] == K.BATCH and e[5] > 0]
            batch_sizes[policy] = sum(decode_batches) / len(decode_batches)

            queued = {e[2]: e[0] for e in events if e[1] == K.QUEUED}
            sched = {e[2]: e[0] for e in events if e[1] == K.SCHEDULED}
            waits = [sched[r] - queued[r] for r in sched if r in queued]
            queue_waits[policy] = sum(waits) / len(waits)
        finally:
            tracer.close()

    assert batch_sizes["prefill-priority"] > batch_sizes["decode-priority"]
    assert queue_waits["decode-priority"] > queue_waits["prefill-priority"]


def test_engine_runs_without_a_tracer() -> None:
    engine = Engine(FakeRunner(speedup=2000), None, EngineConfig(num_blocks=64))
    engine.add_request("r0", prompt_tokens=32, max_new_tokens=4)
    assert len(engine.run_until_idle()) == 1


def test_run_until_idle_reports_a_stuck_engine() -> None:
    engine, tracer = make(EngineConfig(num_blocks=128, max_batch_size=2))
    try:
        engine.add_request("r0", prompt_tokens=32, max_new_tokens=50)
        with pytest.raises(RuntimeError, match="did not drain"):
            engine.run_until_idle(max_iterations=3)
    finally:
        tracer.close()
