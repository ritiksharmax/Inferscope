"""SQLite sink.

Chosen because it needs no server, survives the process, and is directly
queryable by the dashboard and by ad-hoc analysis. All writes happen on the
collector's flush thread via ``executemany``; indices are deferred to
``finalize()`` so they do not slow the insert path.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from inferscope.events import Event
from inferscope.sinks.base import Sink

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    ts_ns INTEGER NOT NULL,
    kind  INTEGER NOT NULL,
    req   INTEGER NOT NULL,
    batch INTEGER NOT NULL,
    a     INTEGER NOT NULL,
    b     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS names (
    idx  INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_INDICES = """
CREATE INDEX IF NOT EXISTS events_req_ts ON events (req, ts_ns);
CREATE INDEX IF NOT EXISTS events_ts     ON events (ts_ns);
CREATE INDEX IF NOT EXISTS events_batch  ON events (batch) WHERE batch >= 0;
"""


class SQLiteSink(Sink):
    """Append events to a SQLite database file.

    Writes normally arrive on the collector's flush thread, but not always: the
    final drain in ``Collector.stop()`` runs on whichever thread called
    ``close()``, after the flush thread has been joined. So the connection is
    opened with ``check_same_thread=False`` and every use is serialized behind a
    lock. Contention is irrelevant -- there is at most one writer at a time by
    construction, and none of this is on the instrumented hot path.
    """

    def __init__(self, path: str | Path, *, synchronous: str = "NORMAL") -> None:
        self.path = str(path)
        self._synchronous = synchronous
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = self._conn
        if conn is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self.path, isolation_level=None, check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA synchronous={self._synchronous}")
            conn.executescript(_SCHEMA)
            self._conn = conn
        return conn

    def describe(self, meta: dict[str, str]) -> None:
        with self._lock:
            conn = self._connect()
            conn.executemany(
                "INSERT OR REPLACE INTO meta VALUES (?,?)", sorted(meta.items())
            )

    def write(self, events: list[Event], names: list[tuple[int, str]]) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute("BEGIN")
            try:
                if names:
                    conn.executemany("INSERT OR REPLACE INTO names VALUES (?,?)", names)
                if events:
                    conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", events)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def finalize(self) -> None:
        """Build query indices. Called once at close, not per write."""
        with self._lock:
            if self._conn is not None:
                self._conn.executescript(_INDICES)

    def close(self) -> None:
        self.finalize()
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
