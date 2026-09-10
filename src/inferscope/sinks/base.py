"""Sink interface.

A sink receives batches of events on the collector's flush thread. It is never
called from the instrumented hot path, so sinks may block, do I/O, and
allocate freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from inferscope.events import Event


class Sink(ABC):
    """Destination for collected events."""

    @abstractmethod
    def write(self, events: list[Event], names: list[tuple[int, str]]) -> None:
        """Persist a batch of events and any newly interned id -> name pairs.

        ``names`` carries only mappings not previously handed to this sink.
        """

    def describe(self, meta: dict[str, str]) -> None:
        """Record trace-level metadata, once, before the first write.

        Carries ``epoch_offset_ns``: add it to any event timestamp to get wall
        time. Optional -- sinks that do not care may ignore it.
        """
        return None

    def flush(self) -> None:
        """Force any buffered state out. Called on explicit flush and close.

        Optional: sinks that write through on every ``write`` need no override.
        """
        return None

    def close(self) -> None:
        """Release resources. Implies ``flush()``."""
        self.flush()
