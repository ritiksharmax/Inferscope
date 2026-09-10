"""Event schema for inferscope.

Every observation is a fixed-arity tuple::

    (ts_ns, kind, req, batch, a, b)

``ts_ns`` is ``time.perf_counter_ns()``. ``req`` and ``batch`` are interned
integer ids (``-1`` when not applicable). ``a`` and ``b`` are two generic
integer payload slots whose meaning depends on ``kind`` (see ``EventKind``).

The arity is fixed so the collector's buffers stay homogeneous and the hot
path never branches on shape. Nothing here derives anything: TTFT, TPOT,
queue time and batch composition are all reconstructed offline by
``inferscope.metrics`` from this stream.
"""

from __future__ import annotations

import threading
from enum import IntEnum

#: An event as it lives in the collector buffers.
Event = tuple[int, int, int, int, int, int]

#: Sentinel for "this event has no request / no batch".
NONE_IDX = -1


class EventKind(IntEnum):
    """Event kinds and the meaning of their ``a`` / ``b`` payload slots."""

    # --- request lifecycle -------------------------------------------------
    REQUEST_START = 1   # a=prompt_tokens
    QUEUED = 2          #
    SCHEDULED = 3       # batch=batch it was admitted into
    PREFILL_START = 4   # batch=batch
    PREFILL_END = 5     # a=tokens_prefilled
    FIRST_TOKEN = 6     # b=batch_size; counts as the request's first token
    DECODE_STEP = 7     # a=step_index, b=batch_size          (decode_mode="full")
                        # emitted from step 1; step 0 is FIRST_TOKEN
    DECODE_RUN = 8      # a=n_steps,    b=batch_size          (decode_mode="aggregate")
    PREEMPTED = 9       # a=blocks_freed, b=tokens_discarded
    RESUMED = 10        # a=tokens_recomputed
    COMPLETE = 11       # a=output_tokens
    FAILED = 12         # a=output_tokens

    # --- batch composition -------------------------------------------------
    BATCH = 13          # batch=batch, a=prefill_tokens, b=decode_tokens
    BATCH_MEMBER = 14   # batch=batch, req=member
    BATCH_PADDING = 15  # batch=batch, a=padding_tokens

    # --- kv cache ----------------------------------------------------------
    KV_ALLOC = 16       # a=blocks
    KV_EVICT = 17       # a=blocks_freed
    KV_FREE = 18        # a=blocks
    KV_USAGE = 19       # a=blocks_used, b=blocks_total

    # --- scheduling stalls -------------------------------------------------
    DECODE_STALL = 20   # a=stall_ns, b=batch_size after the stall


#: Events that establish "when this request last produced a token".
#:
#: A ``DECODE_RUN`` carries only its *end* timestamp; its start is the most
#: recent preceding DECODE_ANCHOR event for the same request. It is emphatically
#: **not** simply the previous event of any kind -- a request also emits
#: ``KV_ALLOC``, ``BATCH_MEMBER`` and ``RESUMED`` events mid-flight, and
#: anchoring on those silently hides exactly the stalls this library exists to
#: find (measuring from a mid-iteration ``KV_ALLOC`` charges a 30 ms starvation
#: stall as a 0.8 ms decode step).
#:
#: ``PREFILL_END`` is in the set so that the first token after a preemption is
#: measured from the recompute finishing, not from the token before the request
#: was evicted.
DECODE_ANCHOR = frozenset({
    EventKind.FIRST_TOKEN,
    EventKind.DECODE_STEP,
    EventKind.DECODE_RUN,
    EventKind.DECODE_STALL,
    EventKind.PREFILL_END,
})

#: Terminal kinds for a request span.
TERMINAL = frozenset({EventKind.COMPLETE, EventKind.FAILED})


class Interner:
    """Maps string ids to dense integer indices.

    Lookups of an already-seen string take a single dict ``__getitem__`` and no
    lock. Only first sight of a string takes the lock, so this stays off the
    hot path in steady state.
    """

    __slots__ = ("_map", "_lock", "_pending", "_names")

    def __init__(self) -> None:
        self._map: dict[str, int] = {}
        self._names: list[str] = []
        self._lock = threading.Lock()
        self._pending: list[tuple[int, str]] = []

    def intern(self, name: str) -> int:
        idx = self._map.get(name)
        if idx is not None:
            return idx
        with self._lock:
            # Re-check: another thread may have won the race.
            idx = self._map.get(name)
            if idx is not None:
                return idx
            idx = len(self._names)
            self._map[name] = idx
            self._names.append(name)
            self._pending.append((idx, name))
            return idx

    def name(self, idx: int) -> str | None:
        if 0 <= idx < len(self._names):
            return self._names[idx]
        return None

    def drain_new(self) -> list[tuple[int, str]]:
        """Return (and clear) mappings interned since the last drain."""
        if not self._pending:
            return []
        with self._lock:
            out, self._pending = self._pending, []
        return out

    def __len__(self) -> int:
        return len(self._names)
