"""In-memory sink, for tests and for short-lived local analysis."""

from __future__ import annotations

import threading

from inferscope.events import Event
from inferscope.sinks.base import Sink


class MemorySink(Sink):
    """Keeps every event in a list. Not bounded -- for tests only."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.names: dict[int, str] = {}
        self.meta: dict[str, str] = {}
        self._lock = threading.Lock()

    def describe(self, meta: dict[str, str]) -> None:
        with self._lock:
            self.meta.update(meta)

    def write(self, events: list[Event], names: list[tuple[int, str]]) -> None:
        with self._lock:
            self.events.extend(events)
            self.names.update(names)

    def snapshot(self) -> list[Event]:
        """A stable copy of what has been written so far, sorted by time."""
        with self._lock:
            return sorted(self.events)
