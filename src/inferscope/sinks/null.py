"""Sink that discards everything -- the baseline for overhead benchmarking."""

from __future__ import annotations

from inferscope.events import Event
from inferscope.sinks.base import Sink


class NullSink(Sink):
    """Counts what it was given and drops it."""

    def __init__(self) -> None:
        self.events_written = 0

    def write(self, events: list[Event], names: list[tuple[int, str]]) -> None:
        self.events_written += len(events)
