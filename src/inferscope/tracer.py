"""The public instrumentation API.

The whole surface is three things: a request span, a batch record, and a KV
record. That is deliberate -- this gets bolted onto engines written by other
people, so it has to be small enough to add in an afternoon.

Hot-path note: ``span.mark`` and ``span.decode_step`` are *closures built per
span*, not methods. Binding the buffer, its ``append``, the capacity and the
request index as closure defaults turns every operation into a ``LOAD_FAST``
instead of an attribute chain. Measured on an M4 / CPython 3.14 that is ~74 ns
per ``mark()`` versus ~88 ns for the equivalent method -- worth the slight
oddity on the single hottest call in the library, and nowhere else.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter_ns
from types import TracebackType
from typing import Any, Literal

from inferscope.collector import DEFAULT_CAPACITY, DEFAULT_FLUSH_INTERVAL, Collector, _Buffer
from inferscope.events import NONE_IDX, EventKind, Interner
from inferscope.sinks.base import Sink
from inferscope.sinks.memory import MemorySink
from inferscope.sinks.null import NullSink
from inferscope.sinks.sqlite import SQLiteSink

DecodeMode = Literal["aggregate", "full", "coarse"]

# Indices into the per-span decode state list. A list rather than attributes
# because list stores are cheaper than slot stores on the per-token path.
#
# _RUN_START holds the value of _TOTAL when the current decode run began, so
# the run's length is a subtraction at run close rather than a counter
# incremented on every single token.
#
# _RUN_ANCHOR is the timestamp the *reader* will treat as the run's start: the
# previous run's end, the stall event, or FIRST_TOKEN. A run of k steps spans
# exactly k intervals from its anchor, so mean step time is
# (last_step - anchor) / k.
#
# Keeping that invariant true costs one subtlety. When the anchor is the
# *current* token's timestamp -- which is the case for FIRST_TOKEN and for the
# token that ends a stall -- that token is already accounted for by the anchor
# event itself, so the next run must not count it again: _RUN_START advances to
# _TOTAL + 1. When the anchor is the previous run's end, the current token is
# genuinely the new run's first, and _RUN_START advances to _TOTAL. Get this
# wrong and a k-token run reports k intervals where it spans k-1: single-step
# runs then report a mean of zero, the adaptive threshold collapses to its
# floor, and every ordinary token registers as a stall.
#
# A reader therefore reconstructs the token count as
# 1 (FIRST_TOKEN) + one per DECODE_STALL + the sum of DECODE_RUN lengths.
_LAST_NS, _RUN_START, _RUN_BS, _TOTAL, _THRESH, _RUN_ANCHOR = 0, 1, 2, 3, 4, 5

#: Until a run has closed once there is no measured step time to scale from, so
#: the first stall threshold is this. Deliberately loose: a false split costs
#: one extra event, a missed split costs the stall.
BOOTSTRAP_STALL_NS = 20_000_000  # 20 ms

#: A step this many times slower than the current run's mean is a stall, not a
#: step, and ends the run.
DEFAULT_STALL_MULTIPLIER = 8

#: Floor on the adaptive threshold, so a fast engine does not split on jitter.
MIN_STALL_NS = 500_000  # 0.5 ms


def _noop(*args: Any, **kwargs: Any) -> None:
    """Stand-in for the hot-path closures when tracing is disabled."""


def resolve_sink(spec: str | Path | Sink) -> Sink:
    """Turn a sink spec into a sink.

    Accepts a ``Sink`` instance, ``"memory"``, ``"null"``, ``"otel"``,
    ``"sqlite:///path"``, or any filesystem path (treated as SQLite).
    """
    if isinstance(spec, Sink):
        return spec
    text = str(spec)
    if text == "memory":
        return MemorySink()
    if text == "otel":
        # Imported here, not at module scope: opentelemetry is an optional
        # extra and the core library must not need it.
        from inferscope.sinks.otel import OTelSink

        return OTelSink()
    if text in ("null", "none"):
        return NullSink()
    if text.startswith("sqlite://"):
        rest = text[len("sqlite://") :]
        # sqlite:///abs/path -> /abs/path ; sqlite://:memory: -> :memory:
        return SQLiteSink(rest or ":memory:")
    return SQLiteSink(text)


class RequestSpan:
    """Tracks one request through the engine.

    Use as a context manager; the terminal event is emitted on exit, including
    when the body raises::

        with tracer.trace_request("r1", prompt_tokens=512) as span:
            span.mark(EventKind.QUEUED)
    """

    __slots__ = (
        "tracer", "request_id", "req", "prompt_tokens",
        "mark", "decode_step", "_state", "_buf", "_closed",
    )

    def __init__(
        self,
        tracer: Tracer,
        request_id: str,
        req: int,
        prompt_tokens: int,
        buf: _Buffer | None,
        decode_mode: DecodeMode,
        stall_threshold_ns: int = BOOTSTRAP_STALL_NS,
        stall_multiplier: int = DEFAULT_STALL_MULTIPLIER,
    ) -> None:
        self.tracer = tracer
        self.request_id = request_id
        self.req = req
        self.prompt_tokens = prompt_tokens
        self._buf = buf
        self._closed = False
        # See the _LAST_NS..._RUN_T0 index constants above.
        self._state: list[int] = [0, 0, -1, 0, stall_threshold_ns, 0]

        if buf is None:
            self.mark = _noop
            self.decode_step = _noop
        else:
            self.mark, self.decode_step = _build_ops(
                buf, req, decode_mode, self._state, stall_multiplier
            )

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> RequestSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close(failed=exc_type is not None)
        return False

    def close(self, *, failed: bool = False) -> None:
        """Emit the pending decode run and the terminal event. Idempotent."""
        if self._closed:
            return
        self._closed = True
        st = self._state
        pending = st[_TOTAL] - st[_RUN_START]
        if pending:
            # Close the open decode run at the timestamp of its last step,
            # not at "now" -- close() may be called well after the last token.
            self._emit_at(st[_LAST_NS], EventKind.DECODE_RUN, pending, st[_RUN_BS])
        self.mark(EventKind.FAILED if failed else EventKind.COMPLETE, st[_TOTAL])

    def preempted(self, blocks_freed: int = 0, tokens_discarded: int = 0) -> None:
        """Mark eviction, closing the open decode run first.

        Use this rather than ``mark(EventKind.PREEMPTED, ...)``: an eviction
        interrupts decoding, and a run left open across it reports tokens
        generated *before* the eviction as though they were generated after the
        recompute. In ``coarse`` mode, where nothing else would close the run,
        that silently loses the pre-eviction decode time entirely.

        It is a separate method so that ``mark`` -- the hottest call in the
        library -- does not have to test for this on every single event.
        """
        st = self._state
        pending = st[_TOTAL] - st[_RUN_START]
        if pending:
            self._emit_at(st[_LAST_NS], EventKind.DECODE_RUN, pending, st[_RUN_BS])
            st[_RUN_START] = st[_TOTAL]
            st[_RUN_ANCHOR] = st[_LAST_NS]
        self.mark(EventKind.PREEMPTED, blocks_freed, tokens_discarded)

    def _emit_at(self, ts_ns: int, kind: int, a: int = 0, b: int = 0) -> None:
        """Emit with an explicit timestamp. Cold path only."""
        buf = self._buf
        if buf is None:
            return
        if len(buf.events) < buf.capacity:
            buf.events.append((ts_ns, kind, self.req, NONE_IDX, a, b))
        else:
            buf.drops += 1

    # -- convenience --------------------------------------------------------

    @property
    def output_tokens(self) -> int:
        """Decode steps recorded so far."""
        return self._state[_TOTAL]

    def prefill(self, tokens: int, batch: int = NONE_IDX) -> None:
        """Bracket a prefill. Prefer the explicit marks when you have both edges."""
        self.mark(EventKind.PREFILL_END, tokens, 0, batch)


def _build_ops(
    buf: _Buffer,
    req: int,
    decode_mode: DecodeMode,
    state: list[int],
    stall_multiplier: int = DEFAULT_STALL_MULTIPLIER,
) -> tuple[Any, Any]:
    """Build the per-span ``mark`` / ``decode_step`` closures.

    Everything the hot path touches is bound as a default argument so the
    bytecode is ``LOAD_FAST`` throughout. ``buf`` stays a real closure cell
    because it is only touched on the (rare) overflow path.
    """
    events = buf.events
    append = events.append
    capacity = buf.capacity

    FIRST_TOKEN = int(EventKind.FIRST_TOKEN)
    DECODE_RUN = int(EventKind.DECODE_RUN)
    DECODE_STEP = int(EventKind.DECODE_STEP)
    DECODE_STALL = int(EventKind.DECODE_STALL)

    def mark(
        kind: int,
        a: int = 0,
        b: int = 0,
        batch: int = NONE_IDX,
        _pcn: Any = perf_counter_ns,
        _ap: Any = append,
        _ev: list[Any] = events,
        _cap: int = capacity,
        _req: int = req,
    ) -> None:
        """Record an instant. ``a``/``b`` mean whatever ``kind`` says they mean."""
        if len(_ev) < _cap:
            _ap((_pcn(), kind, _req, batch, a, b))
        else:
            buf.drops += 1

    if decode_mode == "aggregate":

        def decode_step(
            batch_size: int = 0,
            _pcn: Any = perf_counter_ns,
            _ap: Any = append,
            _ev: list[Any] = events,
            _cap: int = capacity,
            _req: int = req,
            _st: list[int] = state,
            _ft: int = FIRST_TOKEN,
            _dr: int = DECODE_RUN,
            _sv: int = DECODE_STALL,
            _mult: int = stall_multiplier,
            _min: int = MIN_STALL_NS,
        ) -> None:
            """Record that this request produced one token.

            Call this *after* the step that produced the token.

            Contiguous steps at the same batch size collapse into a single
            ``DECODE_RUN`` carrying its *end* timestamp, so the sink sees
            O(runs) not O(tokens). A run also ends when a step takes far longer
            than the run's mean, and that gap is emitted as its own
            ``DECODE_STALL`` -- without it a 25 ms scheduling stall spread over
            a 48-token run raises mean TPOT by 0.5 ms and the pathology
            vanishes from the very data meant to expose it.

            Steady state is a clock read, two comparisons and two stores. The
            ``_RUN_BS`` sentinel of -1 routes a request's first token into the
            same cold branch, so first-token detection costs nothing extra.
            """
            now = _pcn()
            if batch_size != _st[2] or now - _st[0] > _st[4]:
                if _st[2] < 0:
                    if len(_ev) < _cap:
                        _ap((now, _ft, _req, NONE_IDX, 0, batch_size))
                    else:
                        buf.drops += 1
                    _st[5] = now
                    _st[1] = _st[3] + 1
                else:
                    gap = now - _st[0]
                    stalled = gap > _st[4]
                    steps = _st[3] - _st[1]
                    # A zero-length run carries no tokens and shares a timestamp
                    # with the real run beside it. Because events sort by their
                    # payload, the empty one lands first, moves the reader's
                    # anchor, and leaves the real run measured as instantaneous.
                    if steps > 0:
                        if len(_ev) < _cap:
                            _ap((_st[0], _dr, _req, NONE_IDX, steps, _st[2]))
                        else:
                            buf.drops += 1
                        # Re-arm the threshold from this run's mean step time.
                        # Cold path, so the division is free.
                        mean = (_st[0] - _st[5]) // steps
                        _st[4] = _min if mean * _mult < _min else mean * _mult
                    if stalled:
                        if len(_ev) < _cap:
                            _ap((now, _sv, _req, NONE_IDX, gap, batch_size))
                        else:
                            buf.drops += 1
                        _st[5] = now
                        _st[1] = _st[3] + 1
                    else:
                        _st[5] = _st[0]
                        _st[1] = _st[3]
                _st[2] = batch_size
            _st[0] = now
            _st[3] += 1

    elif decode_mode == "coarse":

        def decode_step(  # type: ignore[misc]
            batch_size: int = 0,
            _pcn: Any = perf_counter_ns,
            _ap: Any = append,
            _ev: list[Any] = events,
            _cap: int = capacity,
            _req: int = req,
            _st: list[int] = state,
            _ft: int = FIRST_TOKEN,
            _dr: int = DECODE_RUN,
        ) -> None:
            """Record one token, splitting runs only on batch-size changes.

            The cheapest mode, and the one that will lie to you: a stall inside
            a run is averaged across that run's tokens and disappears. Use it
            when you want throughput, batch-efficiency and KV metrics and are
            not chasing latency tails.
            """
            now = _pcn()
            if batch_size != _st[2]:
                if _st[2] < 0:
                    if len(_ev) < _cap:
                        _ap((now, _ft, _req, NONE_IDX, 0, batch_size))
                    else:
                        buf.drops += 1
                    _st[5] = now
                    _st[1] = _st[3] + 1
                else:
                    if _st[3] > _st[1]:  # never emit a zero-length run
                        if len(_ev) < _cap:
                            _ap((_st[0], _dr, _req, NONE_IDX, _st[3] - _st[1], _st[2]))
                        else:
                            buf.drops += 1
                    _st[5] = _st[0]
                    _st[1] = _st[3]
                _st[2] = batch_size
            _st[0] = now
            _st[3] += 1

    else:

        def decode_step(  # type: ignore[misc]
            batch_size: int = 0,
            _pcn: Any = perf_counter_ns,
            _ap: Any = append,
            _ev: list[Any] = events,
            _cap: int = capacity,
            _req: int = req,
            _st: list[int] = state,
            _ft: int = FIRST_TOKEN,
            _ds: int = DECODE_STEP,
        ) -> None:
            """Record one token, emitting exactly one event per token.

            Token 0 is reported as ``FIRST_TOKEN`` and tokens 1.. as
            ``DECODE_STEP``, so that across every mode the token count
            reconstructs as: one for ``FIRST_TOKEN``, one per ``DECODE_STALL``,
            plus the ``DECODE_RUN`` lengths and the ``DECODE_STEP`` count.
            """
            now = _pcn()
            if len(_ev) < _cap:
                if _st[3] == 0:
                    # FIRST_TOKEN *is* token 0. Emitting a DECODE_STEP for it as
                    # well would double-count it against the terminal event's
                    # token total.
                    _ap((now, _ft, _req, NONE_IDX, 0, batch_size))
                else:
                    _ap((now, _ds, _req, NONE_IDX, _st[3], batch_size))
            else:
                buf.drops += 1
            _st[0] = now
            _st[1] = _st[3] + 1
            _st[5] = now
            _st[3] += 1

    return mark, decode_step


class Tracer:
    """Entry point. One per process, usually created next to the engine."""

    def __init__(
        self,
        sink: str | Path | Sink = "memory",
        *,
        decode_mode: DecodeMode = "aggregate",
        stall_threshold_ns: int = BOOTSTRAP_STALL_NS,
        stall_multiplier: int = DEFAULT_STALL_MULTIPLIER,
        capacity: int = DEFAULT_CAPACITY,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        enabled: bool | None = None,
        autostart: bool = True,
    ) -> None:
        if enabled is None:
            enabled = os.environ.get("INFERSCOPE_DISABLED", "") not in ("1", "true", "yes")
        self.enabled = enabled
        self.decode_mode: DecodeMode = decode_mode
        self.stall_threshold_ns = stall_threshold_ns
        self.stall_multiplier = stall_multiplier
        self.sink = resolve_sink(sink)
        self.collector = Collector(
            self.sink, capacity=capacity, flush_interval=flush_interval
        )
        if enabled and autostart:
            self.collector.start()

    # -- requests -----------------------------------------------------------

    def trace_request(self, request_id: str, prompt_tokens: int = 0) -> RequestSpan:
        """Open a span for ``request_id``. Emits ``REQUEST_START`` immediately."""
        if not self.enabled:
            return RequestSpan(self, request_id, NONE_IDX, prompt_tokens, None, self.decode_mode)
        req = self.collector.interner.intern(request_id)
        buf = self.collector.buffer()
        span = RequestSpan(
            self, request_id, req, prompt_tokens, buf, self.decode_mode,
            self.stall_threshold_ns, self.stall_multiplier,
        )
        span.mark(EventKind.REQUEST_START, prompt_tokens)
        return span

    # -- batches ------------------------------------------------------------

    def record_batch(
        self,
        batch_id: str,
        request_ids: Iterable[str],
        *,
        prefill_tokens: int = 0,
        decode_tokens: int = 0,
        padding_tokens: int = 0,
    ) -> None:
        """Record a batch's composition. Called once per scheduler iteration."""
        if not self.enabled:
            return
        intern = self.collector.interner.intern
        batch = intern(batch_id)
        ts = perf_counter_ns()
        emit = self._emitter()
        emit((ts, EventKind.BATCH, NONE_IDX, batch, prefill_tokens, decode_tokens))
        for rid in request_ids:
            emit((ts, EventKind.BATCH_MEMBER, intern(rid), batch, 0, 0))
        if padding_tokens:
            emit((ts, EventKind.BATCH_PADDING, NONE_IDX, batch, padding_tokens, 0))

    # -- kv cache -----------------------------------------------------------

    def record_kv_event(
        self,
        kind: EventKind,
        request_id: str | None = None,
        *,
        blocks: int = 0,
        tokens: int = 0,
    ) -> None:
        """Record a KV-cache allocation, eviction or free."""
        if not self.enabled:
            return
        req = self.collector.interner.intern(request_id) if request_id else NONE_IDX
        self._emitter()((perf_counter_ns(), kind, req, NONE_IDX, blocks, tokens))

    def record_kv_usage(self, blocks_used: int, blocks_total: int) -> None:
        """Sample overall KV occupancy. Call once per scheduler iteration."""
        if not self.enabled:
            return
        self._emitter()(
            (perf_counter_ns(), EventKind.KV_USAGE, NONE_IDX, NONE_IDX, blocks_used, blocks_total)
        )

    def intern(self, name: str) -> int:
        """The dense integer index for ``name``.

        Engines need this to reference a batch from a request's ``mark(batch=...)``
        without re-deriving the mapping the tracer already keeps.
        """
        if not self.enabled:
            return NONE_IDX
        return self.collector.interner.intern(name)

    def _emitter(self) -> Any:
        buf = self.collector.buffer()
        events = buf.events
        if len(events) >= buf.capacity:
            def drop(_event: Any) -> None:
                buf.drops += 1
            return drop
        return events.append

    # -- lifecycle ----------------------------------------------------------

    @property
    def epoch_offset_ns(self) -> int:
        """Add to any event timestamp to convert monotonic ns to wall-clock ns."""
        return self.collector.epoch_offset_ns

    @property
    def interner(self) -> Interner:
        return self.collector.interner

    @property
    def dropped_events(self) -> int:
        return self.collector.dropped_events

    def flush(self) -> int:
        return self.collector.flush()

    def close(self) -> None:
        self.collector.stop()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        self.close()
        return False
