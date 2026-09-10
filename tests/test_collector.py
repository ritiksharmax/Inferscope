"""Collector concurrency, backpressure, and shutdown draining.

These are the tests that matter: the hot path is lock-free by construction, so
the risk is not slowness but silently losing or duplicating events.
"""

from __future__ import annotations

import threading
import time

from inferscope import EventKind as K
from inferscope import MemorySink, Tracer
from inferscope.collector import Collector
from inferscope.sinks.null import NullSink


def test_concurrent_producers_lose_no_events_while_flushing() -> None:
    """8 threads emitting while the flush thread drains in place."""
    sink = MemorySink()
    tracer = Tracer(sink, flush_interval=0.001)
    n_threads, per_thread = 8, 2000
    barrier = threading.Barrier(n_threads)

    def worker(i: int) -> None:
        barrier.wait()
        for j in range(per_thread):
            with tracer.trace_request(f"r{i}-{j}") as span:
                span.mark(K.QUEUED)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    tracer.close()

    events = sink.snapshot()
    # 3 events per request: REQUEST_START, QUEUED, COMPLETE
    assert len(events) == n_threads * per_thread * 3
    assert tracer.dropped_events == 0

    starts = [e for e in events if e[1] == K.REQUEST_START]
    assert len({e[2] for e in starts}) == n_threads * per_thread, "no duplicate request ids"


def test_drain_in_place_preserves_events_appended_during_the_drain() -> None:
    """The `chunk = ev[:n]; del ev[:n]` dance must not eat concurrent appends."""
    sink = MemorySink()
    tracer = Tracer(sink, autostart=False)
    stop = threading.Event()
    produced = 0

    def producer() -> None:
        nonlocal produced
        while not stop.is_set():
            span = tracer.trace_request("r")
            span.close()
            produced += 1

    t = threading.Thread(target=producer)
    t.start()
    for _ in range(200):
        tracer.flush()
    stop.set()
    t.join()
    tracer.flush()

    # 2 events per request (REQUEST_START, COMPLETE); allow the final in-flight one.
    got = len(sink.snapshot())
    assert got in (produced * 2, produced * 2 + 1), f"produced={produced} collected={got}"
    tracer.close()


def test_overflow_drops_and_counts_instead_of_blocking() -> None:
    tracer = Tracer(MemorySink(), capacity=10, autostart=False)
    try:
        with tracer.trace_request("r1") as span:
            for _ in range(100):
                span.mark(K.QUEUED)
        assert tracer.collector.buffered_events == 10
        assert tracer.dropped_events > 0
    finally:
        tracer.close()


def test_overflow_in_decode_run_path_is_counted() -> None:
    tracer = Tracer(MemorySink(), capacity=2, autostart=False)
    try:
        with tracer.trace_request("r1") as span:
            for i in range(50):
                span.decode_step(batch_size=i)  # every step starts a new run
        assert tracer.dropped_events > 0
    finally:
        tracer.close()


def test_close_drains_buffered_events() -> None:
    sink = MemorySink()
    tracer = Tracer(sink, autostart=False)
    with tracer.trace_request("r1") as span:
        span.mark(K.QUEUED)
    assert sink.snapshot() == [], "nothing written before flush"
    tracer.close()
    assert len(sink.snapshot()) == 3


def test_flush_thread_survives_a_broken_sink() -> None:
    class BrokenSink(NullSink):
        def write(self, events, names):  # type: ignore[no-untyped-def]
            raise RuntimeError("disk is on fire")

    collector = Collector(BrokenSink(), flush_interval=0.001)
    collector.start()
    buf = collector.buffer()
    buf.events.append((1, 1, 0, -1, 0, 0))
    deadline = time.monotonic() + 2.0
    while collector.flush_errors == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert collector.flush_errors > 0
    assert collector._thread is not None and collector._thread.is_alive()
    collector._stop.set()


def test_buffers_of_dead_threads_are_reaped_but_their_drops_survive() -> None:
    collector = Collector(NullSink(), flush_interval=10.0)

    def worker() -> None:
        buf = collector.buffer()
        buf.capacity = 0
        buf.drops = 5

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert len(collector._buffers) == 1
    collector.flush()
    assert len(collector._buffers) == 0, "dead thread's drained buffer is forgotten"
    assert collector.dropped_events == 5, "its drop count is retained"


def test_names_reach_the_sink_once() -> None:
    sink = MemorySink()
    tracer = Tracer(sink, autostart=False)
    for i in range(3):
        tracer.trace_request(f"r{i}").close()
    tracer.flush()
    tracer.trace_request("r0").close()
    tracer.flush()
    assert sink.names == {0: "r0", 1: "r1", 2: "r2"}
    tracer.close()
