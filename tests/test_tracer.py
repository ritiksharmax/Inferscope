"""Span semantics: ordering, terminal events, and decode aggregation."""

from __future__ import annotations

import time

import pytest

from inferscope import EventKind as K
from inferscope import MemorySink, Tracer


@pytest.fixture()
def tracer() -> Tracer:
    t = Tracer(MemorySink(), autostart=False)
    yield t
    t.close()


def kinds(t: Tracer) -> list[K]:
    t.flush()
    assert isinstance(t.sink, MemorySink)
    return [K(e[1]) for e in t.sink.snapshot()]


def test_span_emits_start_and_complete(tracer: Tracer) -> None:
    with tracer.trace_request("r1", prompt_tokens=7) as span:
        span.mark(K.QUEUED)
    assert kinds(tracer) == [K.REQUEST_START, K.QUEUED, K.COMPLETE]

    tracer.flush()
    start = tracer.sink.snapshot()[0]
    assert start[4] == 7, "prompt_tokens travels in slot a"


def test_exception_in_body_emits_failed_not_complete(tracer: Tracer) -> None:
    with pytest.raises(ValueError):
        with tracer.trace_request("r1") as span:
            span.mark(K.PREFILL_START)
            raise ValueError("engine blew up")
    assert kinds(tracer)[-1] == K.FAILED


def test_close_is_idempotent(tracer: Tracer) -> None:
    with tracer.trace_request("r1") as span:
        pass
    span.close()
    span.close()
    assert kinds(tracer).count(K.COMPLETE) == 1


def test_timestamps_are_monotonic_within_a_span(tracer: Tracer) -> None:
    with tracer.trace_request("r1") as span:
        for _ in range(20):
            span.mark(K.QUEUED)
    tracer.flush()
    ts = [e[0] for e in tracer.sink.snapshot()]
    assert ts == sorted(ts)


def test_first_token_emitted_once(tracer: Tracer) -> None:
    with tracer.trace_request("r1") as span:
        for _ in range(5):
            span.decode_step(batch_size=2)
    assert kinds(tracer).count(K.FIRST_TOKEN) == 1


def test_decode_runs_collapse_contiguous_equal_batch_sizes(tracer: Tracer) -> None:
    with tracer.trace_request("r1") as span:
        for _ in range(5):
            span.decode_step(batch_size=2)
        for _ in range(3):
            span.decode_step(batch_size=4)
        assert span.output_tokens == 8

    tracer.flush()
    runs = [e for e in tracer.sink.snapshot() if e[1] == K.DECODE_RUN]
    # The first token is carried by FIRST_TOKEN, so the run that follows it
    # reports 4, not 5: 1 + 4 + 3 == 8 tokens.
    assert [(e[4], e[5]) for e in runs] == [(4, 2), (3, 4)], "(n_steps, batch_size) per run"


def test_decode_run_closes_at_last_step_timestamp(tracer: Tracer) -> None:
    """A run's end must be when its last token landed, not when close() ran."""
    with tracer.trace_request("r1") as span:
        for _ in range(3):
            span.decode_step(batch_size=1)
        last_step_ts = span._state[0]
    tracer.flush()
    run = next(e for e in tracer.sink.snapshot() if e[1] == K.DECODE_RUN)
    assert run[0] == last_step_ts


def test_full_mode_emits_one_event_per_token() -> None:
    t = Tracer(MemorySink(), decode_mode="full", autostart=False)
    try:
        with t.trace_request("r1") as span:
            for _ in range(4):
                span.decode_step(batch_size=3)
        t.flush()
        events = t.sink.snapshot()
        steps = [e for e in events if e[1] == K.DECODE_STEP]
        # Token 0 is FIRST_TOKEN; DECODE_STEP covers tokens 1..3.
        assert [(e[4], e[5]) for e in steps] == [(1, 3), (2, 3), (3, 3)]
        assert next(e for e in events if e[1] == K.FIRST_TOKEN)[5] == 3
        assert not [e for e in events if e[1] == K.DECODE_RUN]
    finally:
        t.close()


def test_terminal_event_carries_output_token_count(tracer: Tracer) -> None:
    with tracer.trace_request("r1") as span:
        for _ in range(6):
            span.decode_step(batch_size=1)
    tracer.flush()
    complete = next(e for e in tracer.sink.snapshot() if e[1] == K.COMPLETE)
    assert complete[4] == 6


def test_record_batch_emits_membership(tracer: Tracer) -> None:
    tracer.record_batch("b1", ["r1", "r2"], prefill_tokens=512, decode_tokens=2,
                        padding_tokens=9)
    tracer.flush()
    events = tracer.sink.snapshot()
    batch = next(e for e in events if e[1] == K.BATCH)
    members = [e for e in events if e[1] == K.BATCH_MEMBER]
    padding = next(e for e in events if e[1] == K.BATCH_PADDING)

    assert (batch[4], batch[5]) == (512, 2)
    assert len(members) == 2
    assert {m[3] for m in members} == {batch[3]}, "members reference the batch index"
    assert padding[4] == 9


def test_kv_events_and_usage(tracer: Tracer) -> None:
    tracer.record_kv_event(K.KV_EVICT, "r1", blocks=4)
    tracer.record_kv_usage(blocks_used=90, blocks_total=128)
    tracer.flush()
    events = tracer.sink.snapshot()
    evict = next(e for e in events if e[1] == K.KV_EVICT)
    usage = next(e for e in events if e[1] == K.KV_USAGE)
    assert evict[4] == 4
    assert (usage[4], usage[5]) == (90, 128)


def test_disabled_tracer_records_nothing_and_still_works() -> None:
    t = Tracer(MemorySink(), enabled=False, autostart=False)
    try:
        with t.trace_request("r1", prompt_tokens=5) as span:
            span.mark(K.QUEUED)
            span.decode_step(batch_size=2)
        t.record_batch("b1", ["r1"])
        t.record_kv_usage(1, 2)
        t.flush()
        assert t.sink.snapshot() == []
    finally:
        t.close()


def test_stall_ends_the_run_and_is_emitted_as_its_own_event() -> None:
    """A gap far longer than the run's mean must survive aggregation.

    This is the regression test for the bug that made the default mode useless:
    closing the run at a stall is not sufficient, because the gap then lands
    inside the *next* run and is averaged across its tokens.
    """
    t = Tracer(MemorySink(), stall_threshold_ns=2_000_000, autostart=False)
    try:
        with t.trace_request("r1") as span:
            for _ in range(4):
                span.decode_step(batch_size=8)
            time.sleep(0.01)  # 10 ms >> the 2 ms bootstrap threshold
            for _ in range(4):
                span.decode_step(batch_size=8)
        t.flush()
        events = t.sink.snapshot()

        stalls = [e for e in events if e[1] == K.DECODE_STALL]
        assert len(stalls) == 1
        assert stalls[0][4] >= 9_000_000, "stall event carries the gap in ns"
        assert stalls[0][5] == 8, "and the batch size it resumed at"

        runs = [e for e in events if e[1] == K.DECODE_RUN]
        # 1 (FIRST_TOKEN) + 3 + 1 (the token ending the stall) + 3 == 8.
        assert [r[4] for r in runs] == [3, 3], "the stall split one run into two"
    finally:
        t.close()


def test_coarse_mode_does_not_split_on_stalls() -> None:
    t = Tracer(MemorySink(), decode_mode="coarse", stall_threshold_ns=2_000_000,
               autostart=False)
    try:
        with t.trace_request("r1") as span:
            for _ in range(4):
                span.decode_step(batch_size=8)
            time.sleep(0.01)
            for _ in range(4):
                span.decode_step(batch_size=8)
        t.flush()
        events = t.sink.snapshot()
        assert not [e for e in events if e[1] == K.DECODE_STALL]
        # 1 (FIRST_TOKEN) + 7 == 8 tokens, in one unbroken run across the stall.
        assert [e[4] for e in events if e[1] == K.DECODE_RUN] == [7]
    finally:
        t.close()


def test_steady_decode_does_not_produce_false_stalls() -> None:
    """The adaptive threshold must not fire on ordinary token-to-token jitter.

    Regression test: measuring a run's mean from the step that *opened* it
    rather than from the reader's anchor undercounts by one interval, so a
    single-step run reports a mean of zero, the threshold collapses to its
    floor, and every subsequent token looks like a stall.
    """
    t = Tracer(MemorySink(), stall_threshold_ns=5_000_000, autostart=False)
    try:
        with t.trace_request("r1") as span:
            # Batch size changes constantly, so runs are short -- exactly the
            # case that produced a zero mean.
            for i in range(60):
                span.decode_step(batch_size=1 + (i % 3))
                time.sleep(0.0005)
        t.flush()
        stalls = [e for e in t.sink.snapshot() if e[1] == K.DECODE_STALL]
        assert stalls == [], f"expected no stalls in steady decode, got {len(stalls)}"
    finally:
        t.close()


def test_decode_run_mean_is_recoverable_from_the_anchor() -> None:
    """A run of k steps spans exactly k intervals from its anchor."""
    t = Tracer(MemorySink(), autostart=False)
    try:
        with t.trace_request("r1") as span:
            for _ in range(5):
                span.decode_step(batch_size=2)
                time.sleep(0.001)
        t.flush()
        events = t.sink.snapshot()
        first_token = next(e for e in events if e[1] == K.FIRST_TOKEN)
        run = next(e for e in events if e[1] == K.DECODE_RUN)
        assert run[4] == 4, "5 tokens = FIRST_TOKEN + a 4-step run"
        mean_ns = (run[0] - first_token[0]) / run[4]
        # 1 ms of sleep per step: the mean must land near it, not at 4/5 of it.
        assert 0.8e6 < mean_ns < 3e6, f"implausible mean step time {mean_ns} ns"
    finally:
        t.close()


@pytest.mark.parametrize("mode", ["aggregate", "coarse", "full"])
@pytest.mark.parametrize("n_tokens", [1, 2, 7, 33])
def test_token_count_reconstructs_in_every_mode(mode: str, n_tokens: int) -> None:
    """The reader must recover the exact token count from any decode mode.

    one FIRST_TOKEN + one per DECODE_STALL + the DECODE_RUN lengths
    + the DECODE_STEP count == the terminal event's token total.
    """
    t = Tracer(MemorySink(), decode_mode=mode, stall_threshold_ns=2_000_000,
               autostart=False)
    try:
        with t.trace_request("r1") as span:
            for i in range(n_tokens):
                span.decode_step(batch_size=2 if i % 5 else 4)
                if i and i % 11 == 0:
                    time.sleep(0.005)  # provoke a stall split
        t.flush()
        events = t.sink.snapshot()

        reconstructed = (
            sum(1 for e in events if e[1] == K.FIRST_TOKEN)
            + sum(1 for e in events if e[1] == K.DECODE_STALL)
            + sum(e[4] for e in events if e[1] == K.DECODE_RUN)
            + sum(1 for e in events if e[1] == K.DECODE_STEP)
        )
        terminal = next(e for e in events if e[1] == K.COMPLETE)
        assert reconstructed == n_tokens == terminal[4]
    finally:
        t.close()


def test_no_zero_length_decode_runs_are_emitted() -> None:
    """An empty run shares a timestamp with the real one and sorts ahead of it.

    Regression: after a stall split or a preemption the run counter sits at
    zero, and the next batch-size change emitted a DECODE_RUN carrying no
    tokens. Events sort by payload, so the empty run landed first, moved the
    reader's anchor onto the real run's own timestamp, and charged 9 ms of
    genuine decoding as instantaneous.
    """
    for mode in ("aggregate", "coarse"):
        t = Tracer(MemorySink(), decode_mode=mode, stall_threshold_ns=2_000_000,
                   autostart=False)
        try:
            with t.trace_request("r1") as span:
                for _ in range(4):
                    span.decode_step(batch_size=8)
                time.sleep(0.01)          # stall -> splits the run
                span.decode_step(batch_size=8)
                span.decode_step(batch_size=4)   # batch change right after
                span.preempted(blocks_freed=2, tokens_discarded=10)
                span.decode_step(batch_size=4)
            t.flush()
            runs = [e for e in t.sink.snapshot() if e[1] == K.DECODE_RUN]
            assert all(r[4] > 0 for r in runs), f"{mode}: empty run {runs}"
        finally:
            t.close()


def test_preemption_closes_the_open_decode_run() -> None:
    """Otherwise tokens generated before eviction are timed from after it."""
    t = Tracer(MemorySink(), decode_mode="coarse", autostart=False)
    try:
        with t.trace_request("r1") as span:
            for _ in range(5):
                span.decode_step(batch_size=4)
            span.preempted(blocks_freed=3, tokens_discarded=20)
            span.mark(K.PREFILL_START)
            span.mark(K.PREFILL_END, 20)
            for _ in range(3):
                span.decode_step(batch_size=4)
        t.flush()
        events = t.sink.snapshot()
        runs = [e for e in events if e[1] == K.DECODE_RUN]
        preempt_ts = next(e[0] for e in events if e[1] == K.PREEMPTED)

        assert len(runs) == 2, "one run before the eviction, one after"
        assert runs[0][0] <= preempt_ts, "the first run must close at the eviction"
        assert runs[0][4] == 4, "1 (FIRST_TOKEN) + 4 == 5 tokens before eviction"
        assert runs[1][4] == 3
    finally:
        t.close()
