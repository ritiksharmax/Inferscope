"""Event collection: per-thread buffers drained by a single flush thread.

Design constraints, in priority order:

1. The hot path must never block. It appends to a plain list owned by the
   calling thread and does nothing else -- no locks, no I/O, no serialization,
   no allocation beyond the event tuple itself.
2. The library must never stall the engine it observes. When a buffer is full
   the event is dropped and counted, never queued behind a slow sink.
3. Draining must not swap the buffer object out from under a producer. The
   flush thread drains *in place* with ``chunk = ev[:n]; del ev[:n]``, which
   under CPython is two atomic list operations; appends racing between them
   land at indices >= n and survive the delete.
"""

from __future__ import annotations

import threading
import time
import weakref
from typing import TYPE_CHECKING

from inferscope.events import Event, Interner

if TYPE_CHECKING:
    from inferscope.sinks.base import Sink

DEFAULT_CAPACITY = 1 << 20
DEFAULT_FLUSH_INTERVAL = 0.05


class _Buffer:
    """One thread's event buffer."""

    __slots__ = ("events", "capacity", "drops", "owner", "__weakref__")

    def __init__(self, capacity: int) -> None:
        self.events: list[Event] = []
        self.capacity = capacity
        self.drops = 0
        self.owner: weakref.ref[threading.Thread] = weakref.ref(threading.current_thread())

    def owner_alive(self) -> bool:
        t = self.owner()
        return t is not None and t.is_alive()


class Collector:
    """Owns the buffers, the interner, and the flush thread."""

    def __init__(
        self,
        sink: Sink,
        *,
        capacity: int = DEFAULT_CAPACITY,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        self.sink = sink
        self.capacity = capacity
        self.flush_interval = flush_interval
        self.interner = Interner()

        # Event timestamps are perf_counter_ns: monotonic, with an arbitrary
        # origin. That is the right clock to measure with and a useless one to
        # correlate with anything else, so capture the offset to wall time once
        # here. Anything needing real timestamps -- OTel export, lining a trace
        # up against an incident -- adds this.
        self.epoch_offset_ns = time.time_ns() - time.perf_counter_ns()

        self._local = threading.local()
        self._buffers: list[_Buffer] = []
        self._registry_lock = threading.Lock()

        self._retired_drops = 0
        self._flushed = 0
        self._announced = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._flush_lock = threading.Lock()
        self.flush_errors = 0
        self.last_error: BaseException | None = None

    # -- buffers ------------------------------------------------------------

    def buffer(self) -> _Buffer:
        """The calling thread's buffer, creating and registering it on first use."""
        buf: _Buffer | None = getattr(self._local, "buf", None)
        if buf is None:
            buf = _Buffer(self.capacity)
            self._local.buf = buf
            with self._registry_lock:
                self._buffers.append(buf)
        return buf

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="inferscope-flush", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Stop the flush thread, drain everything, and close the sink."""
        thread, self._thread = self._thread, None
        self._stop.set()
        if thread is not None:
            thread.join(timeout)
        self.flush()
        self.sink.close()

    def _run(self) -> None:
        while not self._stop.wait(self.flush_interval):
            try:
                self.flush()
            except Exception as exc:  # a broken sink must not kill the thread
                self.flush_errors += 1
                self.last_error = exc

    # -- draining -----------------------------------------------------------

    def flush(self) -> int:
        """Drain all buffers into the sink. Returns the number of events written."""
        with self._flush_lock:
            with self._registry_lock:
                buffers = list(self._buffers)

            chunk: list[Event] = []
            for buf in buffers:
                ev = buf.events
                n = len(ev)
                if n:
                    chunk.extend(ev[:n])
                    del ev[:n]

            names = self.interner.drain_new()
            if chunk:
                chunk.sort()
            if chunk or names:
                if not self._announced:
                    self._announced = True
                    describe = getattr(self.sink, "describe", None)
                    if callable(describe):
                        describe({"epoch_offset_ns": str(self.epoch_offset_ns)})
                self.sink.write(chunk, names)
                self._flushed += len(chunk)

            self._reap(buffers)
            return len(chunk)

    def _reap(self, buffers: list[_Buffer]) -> None:
        """Forget buffers belonging to threads that have exited and are drained."""
        dead = [b for b in buffers if not b.events and not b.owner_alive()]
        if not dead:
            return
        with self._registry_lock:
            for b in dead:
                try:
                    self._buffers.remove(b)
                except ValueError:
                    continue
                self._retired_drops += b.drops

    # -- introspection ------------------------------------------------------

    @property
    def dropped_events(self) -> int:
        """Events discarded because a buffer was at capacity."""
        with self._registry_lock:
            return self._retired_drops + sum(b.drops for b in self._buffers)

    @property
    def flushed_events(self) -> int:
        return self._flushed

    @property
    def buffered_events(self) -> int:
        with self._registry_lock:
            return sum(len(b.events) for b in self._buffers)
