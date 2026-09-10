"""Derived metrics: everything the hot path deliberately did not compute.

Two levels. ``per_request`` reconstructs each request's timeline into a
breakdown that sums to its wall time. ``window_metrics`` summarises the engine
over a period: batch efficiency, KV occupancy, preemption.

All of it is mode-agnostic. ``aggregate`` traces carry explicit ``DECODE_STALL``
events; ``full`` traces carry every token's timestamp and no stall events, so
stalls are recovered here by outlier detection against the median step. Callers
should not have to know which mode produced the trace they were handed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from inferscope.events import EventKind as K
from inferscope.trace import Trace

#: A step this many times the median is treated as a stall when the trace does
#: not carry explicit stall events (i.e. ``decode_mode="full"``).
STALL_MULTIPLIER = 8.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


@dataclass
class RequestMetrics:
    """One request's timeline, decomposed."""

    request_id: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    completed: bool = False
    failed: bool = False

    arrival_ns: int = 0
    finish_ns: int = 0

    # -- the breakdown; these sum to total_ns together with unattributed_ns --
    queue_ns: int = 0        # arrival -> the engine first does work for it
    prefill_ns: int = 0      # time inside prefill, including recompute
    decode_wait_ns: int = 0  # prefilled, waiting for a decode slot
    decode_ns: int = 0       # time actually producing tokens
    stall_ns: int = 0        # scheduled-out while nominally running
    requeue_ns: int = 0      # preempted -> re-admitted

    # == queue_ns + prefill_ns + decode_wait_ns, for a request that was
    # never preempted. After a preemption prefill_ns also carries recompute
    # that happened long after the first token, and the identity stops holding.
    ttft_ns: int = 0
    tpot_samples: list[float] = field(default_factory=list, repr=False)
    stalls: list[float] = field(default_factory=list, repr=False)
    batch_sizes: list[int] = field(default_factory=list, repr=False)

    preemptions: int = 0
    recomputed_tokens: int = 0

    @property
    def total_ns(self) -> int:
        return self.finish_ns - self.arrival_ns

    @property
    def unattributed_ns(self) -> int:
        """Whatever the breakdown does not account for.

        Reported rather than hidden: a breakdown that silently fails to add up
        is worse than one that admits a gap.
        """
        return self.total_ns - (
            self.queue_ns + self.prefill_ns + self.decode_wait_ns
            + self.decode_ns + self.stall_ns + self.requeue_ns
        )

    @property
    def tpot_mean_ns(self) -> float:
        return statistics.fmean(self.tpot_samples) if self.tpot_samples else 0.0

    @property
    def tpot_p50_ns(self) -> float:
        return _percentile(self.tpot_samples, 0.50)

    @property
    def tpot_p99_ns(self) -> float:
        return _percentile(self.tpot_samples, 0.99)

    @property
    def max_stall_ns(self) -> float:
        return max(self.stalls, default=0.0)

    @property
    def mean_batch_size(self) -> float:
        return statistics.fmean(self.batch_sizes) if self.batch_sizes else 0.0


def per_request(trace: Trace) -> dict[str, RequestMetrics]:
    """Reconstruct every request's timeline from the event stream."""
    out: dict[str, RequestMetrics] = {}

    for req, events in trace.by_request.items():
        name = trace.name(req)
        m = RequestMetrics(request_id=name)

        anchor: int | None = None       # last DECODE_ANCHOR timestamp
        prefill_start: int | None = None
        preempted_at: int | None = None
        first_scheduled: int | None = None
        started_work = False
        saw_start = False
        raw_steps: list[tuple[float, int]] = []  # (duration_ns, batch_size)
        explicit_stalls = False

        for ts, kind, _req, _batch, a, b in events:
            if kind == K.REQUEST_START:
                m.arrival_ns = ts
                m.prompt_tokens = a
                saw_start = True
            elif kind == K.SCHEDULED:
                if first_scheduled is None:
                    first_scheduled = ts
            elif kind == K.PREFILL_START:
                # Work starting -- not admission -- is what closes the waiting
                # period. Between SCHEDULED and PREFILL_START an engine still
                # has bookkeeping to do (block allocation, here), and charging
                # that microsecond gap to nothing broke the TTFT identity.
                if not started_work:
                    started_work = True
                    m.queue_ns = ts - m.arrival_ns
                elif preempted_at is not None:
                    m.requeue_ns += ts - preempted_at
                    preempted_at = None
                prefill_start = ts
            elif kind == K.PREFILL_END:
                if prefill_start is not None:
                    m.prefill_ns += ts - prefill_start
                    prefill_start = None
                anchor = ts
            elif kind == K.FIRST_TOKEN:
                # Prefill finished but the scheduler had not yet given this
                # request a decode slot. Under prefill-priority scheduling this
                # is where a starved request's time actually goes -- 26 ms of a
                # 57 ms request in the reference workload -- so it gets its own
                # term rather than being left unattributed.
                if anchor is not None:
                    m.decode_wait_ns = ts - anchor
                m.ttft_ns = ts - m.arrival_ns
                m.output_tokens += 1
                if b:
                    m.batch_sizes.append(b)
                anchor = ts
            elif kind == K.DECODE_RUN:
                if a <= 0:
                    # Carries no tokens; advancing the anchor on it would zero
                    # out whatever real run shares its timestamp.
                    continue
                if anchor is not None:
                    duration = ts - anchor
                    m.decode_ns += duration
                    raw_steps.extend([(duration / a, b)] * a)
                    m.batch_sizes.extend([b] * a)
                m.output_tokens += a
                anchor = ts
            elif kind == K.DECODE_STEP:
                if anchor is not None:
                    duration = ts - anchor
                    m.decode_ns += duration
                    raw_steps.append((float(duration), b))
                m.output_tokens += 1
                if b:
                    m.batch_sizes.append(b)
                anchor = ts
            elif kind == K.DECODE_STALL:
                explicit_stalls = True
                # Measure from this walk's own anchor, not from the tracer's
                # reported gap `a`. The two agree for an ordinary scheduling
                # stall. They diverge when the request was preempted: the
                # tracer's gap runs from the last token before eviction and so
                # spans the requeue and the recompute, both of which are already
                # attributed. Trusting `a` there made breakdowns sum to 143% of
                # wall time.
                gap = ts - anchor if anchor is not None else a
                m.stall_ns += gap
                m.stalls.append(float(gap))
                # A stall is also a token that took `gap` to arrive, so it
                # belongs in the TPOT distribution. Leaving it out made the
                # headline p99/p50 depend on the decode mode: 1.1x from an
                # aggregate trace against 40x from a full one, same workload.
                raw_steps.append((float(gap), b))
                m.output_tokens += 1
                if b:
                    m.batch_sizes.append(b)
                anchor = ts
            elif kind == K.PREEMPTED:
                m.preemptions += 1
                preempted_at = ts
                anchor = None
                prefill_start = None
            elif kind == K.RESUMED:
                m.recomputed_tokens += a
            elif kind in (K.COMPLETE, K.FAILED):
                m.finish_ns = ts
                m.completed = kind == K.COMPLETE
                m.failed = kind == K.FAILED
                m.output_tokens = a or m.output_tokens

        if not saw_start:
            continue  # a bare BATCH_MEMBER reference, not a traced request

        if not started_work and first_scheduled is not None:
            # An engine that marks admission but not prefill still gets a
            # queue time, just a slightly coarser one.
            m.queue_ns = first_scheduled - m.arrival_ns

        m.tpot_samples = [d for d, _bs in raw_steps]

        if not explicit_stalls and m.tpot_samples:
            # A "full" trace records every token but no stall events; recover
            # them the same way the hot path would have, so the metric means
            # the same thing whichever mode produced the trace.
            median = statistics.median(m.tpot_samples)
            threshold = median * STALL_MULTIPLIER
            outliers = [d for d in m.tpot_samples if d > threshold]
            if outliers:
                m.stalls = outliers
                m.stall_ns = int(sum(outliers))
                m.decode_ns -= m.stall_ns

        out[name] = m
    return out


@dataclass
class WindowMetrics:
    """How the engine behaved over a period, independent of any one request."""

    iterations: int = 0
    prefill_iterations: int = 0
    decode_iterations: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    padding_tokens: int = 0

    batch_sizes: list[int] = field(default_factory=list, repr=False)
    kv_occupancy: list[float] = field(default_factory=list, repr=False)

    evictions: int = 0
    evicted_blocks: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0

    span_ns: int = 0

    @property
    def mean_batch_size(self) -> float:
        return statistics.fmean(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def max_batch_size(self) -> int:
        return max(self.batch_sizes, default=0)

    @property
    def padding_ratio(self) -> float:
        """Share of attended tokens that were padding rather than context."""
        attended = self.decode_tokens + self.padding_tokens
        return self.padding_tokens / attended if attended else 0.0

    @property
    def mean_kv_occupancy(self) -> float:
        return statistics.fmean(self.kv_occupancy) if self.kv_occupancy else 0.0

    @property
    def peak_kv_occupancy(self) -> float:
        return max(self.kv_occupancy, default=0.0)

    @property
    def goodput_ratio(self) -> float:
        """Fraction of prefill work that was not thrown away and redone.

        1.0 means no recompute; 0.5 means half the prefill tokens the engine
        processed were re-processing context it had already built and evicted.
        """
        if not self.prefill_tokens:
            return 1.0
        return max(0.0, 1.0 - self.recomputed_tokens / self.prefill_tokens)


def window_metrics(
    trace: Trace, start_ns: int | None = None, end_ns: int | None = None
) -> WindowMetrics:
    """Summarise engine behaviour, optionally restricted to a time window."""
    w = WindowMetrics()
    first = last = 0

    for ts, kind, _req, _batch, a, b in trace.events:
        if start_ns is not None and ts < start_ns:
            continue
        if end_ns is not None and ts > end_ns:
            continue
        first = first or ts
        last = ts

        if kind == K.BATCH:
            w.iterations += 1
            w.prefill_tokens += a
            w.decode_tokens += b
            if a:
                w.prefill_iterations += 1
            if b:
                w.decode_iterations += 1
                w.batch_sizes.append(b)
        elif kind == K.BATCH_PADDING:
            w.padding_tokens += a
        elif kind == K.KV_USAGE:
            if b:
                w.kv_occupancy.append(a / b)
        elif kind == K.KV_EVICT:
            w.evictions += 1
            w.evicted_blocks += a
        elif kind == K.PREEMPTED:
            w.preemptions += 1
        elif kind == K.RESUMED:
            w.recomputed_tokens += a

    w.span_ns = last - first
    return w
