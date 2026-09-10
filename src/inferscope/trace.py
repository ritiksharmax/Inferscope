"""Loading a recorded event stream back for analysis.

The hot path writes flat tuples and derives nothing. This is the other half of
that bargain: everything interesting gets reconstructed here, by walking the
stream in timestamp order.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from inferscope.events import NONE_IDX, Event, EventKind
from inferscope.sinks.memory import MemorySink


@dataclass
class Trace:
    """A recorded event stream, plus the id table needed to read it."""

    events: list[Event]
    names: dict[int, str] = field(default_factory=dict)
    #: Add to an event timestamp to get wall-clock ns. Zero when unknown, in
    #: which case timestamps are monotonic-only and correlate with nothing.
    epoch_offset_ns: int = 0

    def __post_init__(self) -> None:
        self.events.sort()
        self._by_request: dict[int, list[Event]] | None = None

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_sqlite(cls, path: str | Path) -> Trace:
        conn = sqlite3.connect(str(path))
        try:
            events = [
                (int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]))
                for r in conn.execute(
                    "SELECT ts_ns, kind, req, batch, a, b FROM events"
                )
            ]
            names = {int(i): str(n) for i, n in conn.execute("SELECT idx, name FROM names")}
            meta = dict(conn.execute("SELECT key, value FROM meta"))
        finally:
            conn.close()
        return cls(events, names, int(meta.get("epoch_offset_ns", 0)))

    @classmethod
    def from_sink(cls, sink: MemorySink) -> Trace:
        return cls(
            list(sink.events), dict(sink.names),
            int(sink.meta.get("epoch_offset_ns", 0)),
        )

    def wall_ns(self, ts_ns: int) -> int:
        """Convert an event timestamp to wall-clock nanoseconds."""
        return ts_ns + self.epoch_offset_ns

    @classmethod
    def from_tracer(cls, tracer: object) -> Trace:
        """Flush a live tracer with a ``MemorySink`` and read what it has so far."""
        flush = getattr(tracer, "flush", None)
        if callable(flush):
            flush()
        sink = getattr(tracer, "sink", None)
        if not isinstance(sink, MemorySink):
            raise TypeError("Trace.from_tracer needs a tracer backed by a MemorySink")
        return cls.from_sink(sink)

    # -- access -------------------------------------------------------------

    def name(self, idx: int) -> str:
        return self.names.get(idx, f"<{idx}>")

    def index_of(self, request_id: str) -> int:
        for idx, name in self.names.items():
            if name == request_id:
                return idx
        raise KeyError(request_id)

    @property
    def by_request(self) -> dict[int, list[Event]]:
        """Events grouped by request index, each list in timestamp order.

        ``BATCH_MEMBER`` events carry a request index too, so a request's list
        includes the iterations it participated in.
        """
        if self._by_request is None:
            grouped: dict[int, list[Event]] = {}
            for event in self.events:
                req = event[2]
                if req != NONE_IDX:
                    grouped.setdefault(req, []).append(event)
            self._by_request = grouped
        return self._by_request

    def requests(self) -> list[int]:
        """Request indices that have a REQUEST_START, in arrival order."""
        return [e[2] for e in self.events if e[1] == EventKind.REQUEST_START]

    @property
    def span_ns(self) -> int:
        return self.events[-1][0] - self.events[0][0] if self.events else 0

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return (
            f"Trace({len(self.events)} events, {len(self.requests())} requests, "
            f"{self.span_ns / 1e6:.0f} ms)"
        )
