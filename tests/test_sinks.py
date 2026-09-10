"""Sink round-trips."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from inferscope import EventKind as K
from inferscope import SQLiteSink, Tracer


def test_sqlite_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "traces.db"
    tracer = Tracer(f"sqlite://{db}", autostart=False)
    with tracer.trace_request("req-abc", prompt_tokens=512) as span:
        span.mark(K.PREFILL_START)
        span.mark(K.PREFILL_END, 512)
        for _ in range(3):
            span.decode_step(batch_size=2)
    tracer.close()

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT kind, a, b FROM events ORDER BY ts_ns").fetchall()
    names = dict(conn.execute("SELECT idx, name FROM names").fetchall())

    assert [K(r[0]) for r in rows] == [
        K.REQUEST_START, K.PREFILL_START, K.PREFILL_END,
        K.FIRST_TOKEN, K.DECODE_RUN, K.COMPLETE,
    ]
    assert names == {0: "req-abc"}
    conn.close()


def test_sqlite_creates_indices_on_close(tmp_path: Path) -> None:
    db = tmp_path / "traces.db"
    tracer = Tracer(f"sqlite://{db}", autostart=False)
    tracer.trace_request("r1").close()
    tracer.flush()

    conn = sqlite3.connect(db)
    before = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'events%'"
    ).fetchall()
    assert before == [], "indices must not slow the insert path"
    conn.close()

    tracer.close()
    conn = sqlite3.connect(db)
    after = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'events%'"
    )}
    assert after == {"events_req_ts", "events_ts", "events_batch"}
    conn.close()


def test_sqlite_creates_parent_directories(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "dir" / "traces.db"
    tracer = Tracer(SQLiteSink(db), autostart=False)
    tracer.trace_request("r1").close()
    tracer.close()
    assert db.exists()


def test_sink_spec_resolution(tmp_path: Path) -> None:
    from inferscope.sinks import MemorySink, NullSink
    from inferscope.tracer import resolve_sink

    assert isinstance(resolve_sink("memory"), MemorySink)
    assert isinstance(resolve_sink("null"), NullSink)
    assert isinstance(resolve_sink(str(tmp_path / "x.db")), SQLiteSink)
    assert resolve_sink("sqlite:///tmp/x.db").path == "/tmp/x.db"
    existing = MemorySink()
    assert resolve_sink(existing) is existing


def test_sqlite_survives_the_flush_thread_handoff(tmp_path: Path) -> None:
    """Regression: the final drain runs on the caller's thread, not the flush thread.

    ``Collector.stop()`` joins the flush thread and then flushes once more from
    whoever called ``close()``. With a connection pinned to its creating thread
    that raises ``ProgrammingError`` -- and since SQLite is the default sink and
    a running flush thread is the default configuration, this broke every
    ordinary use. It went unnoticed because every other sqlite test here builds
    its tracer with ``autostart=False`` and so never crosses threads.
    """
    db = tmp_path / "traces.db"
    tracer = Tracer(f"sqlite://{db}", flush_interval=0.005)
    for i in range(50):
        with tracer.trace_request(f"r{i}", prompt_tokens=8) as span:
            span.mark(K.QUEUED)
            span.decode_step(batch_size=2)
    time.sleep(0.05)  # let the flush thread open the connection and write
    tracer.close()

    conn = sqlite3.connect(db)
    (count,) = conn.execute("SELECT count(*) FROM events").fetchone()
    (names,) = conn.execute("SELECT count(*) FROM names").fetchone()
    conn.close()
    assert count == 50 * 4, "REQUEST_START, QUEUED, FIRST_TOKEN, COMPLETE per request"
    assert names == 50
