"""The four questions, as first-class queries.

Each returns a small structured result carrying a plain-language ``verdict``,
because a metric nobody can interpret under incident pressure is not
observability. Thresholds are module constants, calibrated against the
reference workloads in ``inferscope_lab.pathologies`` -- see ``docs/queries.md``
for the numbers each scenario actually produces.

    python -m inferscope.query --db traces.db
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass, field

from inferscope.events import EventKind as K
from inferscope.metrics import per_request, window_metrics
from inferscope.trace import Trace

#: Share of total request time spent prefilled-but-unscheduled, or stalled
#: mid-decode, above which the scheduler -- not the model -- is the problem.
#: Reference workloads: healthy 8.7%, prefill-starvation 40.6%.
SCHEDULING_SHARE = 0.20

#: Share of total request time spent queued before first admission, above which
#: admission is the bottleneck. Reference: healthy 0%, batch-starvation 92.8%.
QUEUE_SHARE = 0.40

#: Prefill goodput below this means meaningful work is being redone.
#: Reference: healthy 1.00, kv-thrash 0.77.
GOODPUT_FLOOR = 0.95

#: A decode step this many times the median counts as being "in the tail".
TAIL_MULTIPLE = 8.0

#: Share of total decode *time* spent in those tail steps, above which the
#: cadence problem is real. Time rather than a percentile of samples: stalls
#: are ~1% of steps in the reference workload, which puts p99 exactly on the
#: knife edge -- the same scenario scored 39x and 1.3x on separate runs purely
#: from where the boundary landed. Their share of elapsed time is ~30% and
#: stable, because each one is worth ~40 ordinary steps.
TAIL_TIME_SHARE = 0.10


def _ms(ns: float) -> float:
    return ns / 1e6


# --------------------------------------------------------------------------
# 1. Was this request slow from queueing or from compute?
# --------------------------------------------------------------------------


@dataclass
class LatencyBreakdown:
    """Where one request's wall time went. The parts sum to the whole."""

    request_id: str
    total_ns: int
    components: dict[str, int]
    dominant: str
    verdict: str

    def share(self, name: str) -> float:
        return self.components.get(name, 0) / self.total_ns if self.total_ns else 0.0

    def format(self) -> str:
        lines = [f"  {self.request_id}: {_ms(self.total_ns):.1f} ms total"]
        for name, ns in self.components.items():
            if ns:
                bar = "#" * max(1, round(self.share(name) * 40))
                lines.append(
                    f"    {name:<13}{_ms(ns):8.1f} ms  {self.share(name) * 100:5.1f}%  {bar}"
                )
        lines.append(f"    -> {self.verdict}")
        return "\n".join(lines)


def explain_latency(trace: Trace, request_id: str) -> LatencyBreakdown:
    """Decompose one request's latency into terms that sum to its wall time."""
    metrics = per_request(trace)
    if request_id not in metrics:
        raise KeyError(f"{request_id!r} is not in this trace")
    m = metrics[request_id]

    components = {
        "queued": m.queue_ns,
        "prefill": m.prefill_ns,
        "decode_wait": m.decode_wait_ns,
        "decoding": m.decode_ns,
        "stalled": m.stall_ns,
        "requeued": m.requeue_ns,
        "unattributed": max(0, m.unattributed_ns),
    }
    dominant = max(components, key=lambda k: components[k])

    scheduling = m.decode_wait_ns + m.stall_ns
    share = scheduling / m.total_ns if m.total_ns else 0.0
    if m.preemptions:
        verdict = (
            f"KV pressure: preempted {m.preemptions}x, "
            f"{m.recomputed_tokens} tokens recomputed"
        )
    elif share > SCHEDULING_SHARE:
        verdict = (
            f"waiting on the scheduler, not computing: {share * 100:.0f}% of its life "
            f"was prefilled-but-unscheduled or stalled mid-decode"
        )
    elif dominant == "queued":
        verdict = "queued before admission: the engine never got to it"
    elif dominant == "decoding":
        verdict = "genuinely computing: time went into generating tokens"
    else:
        verdict = f"dominated by {dominant}"

    return LatencyBreakdown(request_id, m.total_ns, components, dominant, verdict)


# --------------------------------------------------------------------------
# 2. What was it batched with?
# --------------------------------------------------------------------------


@dataclass
class BatchContext:
    """Who this request shared iterations with, and what else ran meanwhile."""

    request_id: str
    iterations: int
    mean_batch_size: float
    co_residents: list[tuple[str, int]]
    prefill_iterations_during: int
    prefill_tokens_during: int
    verdict: str

    def format(self) -> str:
        lines = [
            f"  {self.request_id}: {self.iterations} iterations, "
            f"mean batch {self.mean_batch_size:.1f}",
        ]
        if self.co_residents:
            top = ", ".join(f"{n} ({c})" for n, c in self.co_residents[:5])
            lines.append(f"    most often batched with: {top}")
        if self.prefill_iterations_during:
            lines.append(
                f"    {self.prefill_iterations_during} prefill iterations ran during its "
                f"lifetime, prefilling {self.prefill_tokens_during} tokens for other requests"
            )
        lines.append(f"    -> {self.verdict}")
        return "\n".join(lines)


def batch_context(trace: Trace, request_id: str) -> BatchContext:
    """Reconstruct what else was in the engine while this request ran."""
    req = trace.index_of(request_id)
    metrics = per_request(trace)[request_id]

    # Batches this request belonged to.
    my_batches = {e[3] for e in trace.by_request.get(req, ()) if e[1] == K.BATCH_MEMBER}
    members: dict[int, list[int]] = {}
    for _ts, kind, r, batch, _a, _b in trace.events:
        if kind == K.BATCH_MEMBER and batch in my_batches:
            members.setdefault(batch, []).append(r)

    counter: Counter[str] = Counter()
    for rows in members.values():
        for other in rows:
            if other != req:
                counter[trace.name(other)] += 1

    # Prefill work done for *other* requests during this one's lifetime.
    start, end = metrics.arrival_ns, metrics.finish_ns or metrics.arrival_ns
    prefill_iters = 0
    prefill_tokens = 0
    for ts, kind, _r, batch, a, _b in trace.events:
        if kind != K.BATCH or not a or not (start <= ts <= end):
            continue
        if batch in my_batches:
            continue  # its own prefill
        prefill_iters += 1
        prefill_tokens += a

    waiting = metrics.decode_wait_ns + metrics.stall_ns
    waiting_share = waiting / metrics.total_ns if metrics.total_ns else 0.0

    # Order matters: a request can both run alone *and* have prefill happening
    # around it. Running alone is the more specific finding, and claiming it
    # "shared the engine" while co_residents is empty reads as a contradiction.
    if not counter and metrics.mean_batch_size and metrics.mean_batch_size < 2:
        verdict = "ran nearly alone: batches never filled, so the engine was underused"
    elif prefill_tokens and waiting_share > SCHEDULING_SHARE:
        verdict = (
            f"shared the engine with {prefill_tokens} tokens of other requests' prefill; "
            f"that is what it was waiting behind"
        )
    elif metrics.mean_batch_size and metrics.mean_batch_size < 2:
        verdict = "ran nearly alone: batches never filled, so the engine was underused"
    else:
        verdict = "batched normally with other decoding requests"

    return BatchContext(
        request_id=request_id,
        iterations=len(my_batches),
        mean_batch_size=metrics.mean_batch_size,
        co_residents=counter.most_common(),
        prefill_iterations_during=prefill_iters,
        prefill_tokens_during=prefill_tokens,
        verdict=verdict,
    )


# --------------------------------------------------------------------------
# 3. Is KV-cache pressure causing preemption and recompute?
# --------------------------------------------------------------------------


@dataclass
class KVPressure:
    mean_occupancy: float
    peak_occupancy: float
    evictions: int
    evicted_blocks: int
    preemptions: int
    recomputed_tokens: int
    prefill_tokens: int
    goodput_ratio: float
    verdict: str

    def format(self) -> str:
        return "\n".join([
            f"  KV occupancy: mean {self.mean_occupancy * 100:.0f}%, "
            f"peak {self.peak_occupancy * 100:.0f}%",
            f"  evictions: {self.evictions} ({self.evicted_blocks} blocks), "
            f"preemptions: {self.preemptions}",
            f"  prefill goodput: {self.goodput_ratio * 100:.0f}% "
            f"({self.recomputed_tokens} of {self.prefill_tokens} prefill tokens were recompute)",
            f"    -> {self.verdict}",
        ])


def kv_pressure(
    trace: Trace, start_ns: int | None = None, end_ns: int | None = None
) -> KVPressure:
    """Assess whether the KV pool is the constraint."""
    w = window_metrics(trace, start_ns, end_ns)

    if w.preemptions and w.goodput_ratio < GOODPUT_FLOOR:
        wasted = (1 - w.goodput_ratio) * 100
        verdict = (
            f"KV pool is too small for the working set: {w.preemptions} preemptions, "
            f"{wasted:.0f}% of prefill work thrown away and redone"
        )
    elif w.preemptions:
        verdict = f"occasional preemption ({w.preemptions}), but little work lost"
    elif w.peak_kv_occupancy > 0.9:
        verdict = "pool nearly full at peak but no preemption yet -- little headroom"
    else:
        verdict = "KV pool is not the constraint"

    return KVPressure(
        mean_occupancy=w.mean_kv_occupancy,
        peak_occupancy=w.peak_kv_occupancy,
        evictions=w.evictions,
        evicted_blocks=w.evicted_blocks,
        preemptions=w.preemptions,
        recomputed_tokens=w.recomputed_tokens,
        prefill_tokens=w.prefill_tokens,
        goodput_ratio=w.goodput_ratio,
        verdict=verdict,
    )


# --------------------------------------------------------------------------
# 4. Is the bottleneck TTFT or TPOT?
# --------------------------------------------------------------------------


@dataclass
class LatencyRegime:
    requests: int
    ttft_p50_ns: float
    ttft_p99_ns: float
    tpot_p50_ns: float
    tpot_p99_ns: float
    tpot_max_ns: float
    tpot_tail_ratio: float
    tail_time_share: float
    ttft_share: float
    regime: str
    verdict: str

    def format(self) -> str:
        return "\n".join([
            f"  {self.requests} requests",
            f"  TTFT  p50 {_ms(self.ttft_p50_ns):7.1f} ms   p99 {_ms(self.ttft_p99_ns):7.1f} ms",
            f"  TPOT  p50 {_ms(self.tpot_p50_ns):7.3f} ms   p99 {_ms(self.tpot_p99_ns):7.3f} ms"
            f"   max {_ms(self.tpot_max_ns):7.1f} ms",
            f"  {self.tail_time_share * 100:.0f}% of decode time is in steps "
            f">{TAIL_MULTIPLE:.0f}x the median",
            f"    -> {self.verdict}",
        ])


def latency_regime(trace: Trace) -> LatencyRegime:
    """Say whether time is going into first-token latency or per-token cadence."""
    metrics = [m for m in per_request(trace).values() if m.completed or m.failed]
    if not metrics:
        raise ValueError("trace contains no completed requests")

    ttfts = sorted(float(m.ttft_ns) for m in metrics if m.ttft_ns)
    tpots = sorted(x for m in metrics for x in m.tpot_samples)
    if not tpots:
        raise ValueError("trace contains no decode steps")

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]

    ttft_p50, ttft_p99 = pct(ttfts, 0.5), pct(ttfts, 0.99)
    tpot_p50, tpot_p99 = statistics.median(tpots), pct(tpots, 0.99)
    ratio = tpot_p99 / tpot_p50 if tpot_p50 else 0.0

    tail_cut = tpot_p50 * TAIL_MULTIPLE
    tail_time = sum(x for x in tpots if x > tail_cut)
    tail_share = tail_time / sum(tpots) if tpots else 0.0

    total = sum(m.total_ns for m in metrics)
    ttft_share = sum(m.ttft_ns for m in metrics) / total if total else 0.0

    if tail_share > TAIL_TIME_SHARE:
        regime = "tpot-tail"
        verdict = (
            f"decode cadence is the problem: {tail_share * 100:.0f}% of decode time goes "
            f"into steps over {TAIL_MULTIPLE:.0f}x the median, worst {_ms(max(tpots)):.0f} ms. "
            f"Request-level latency will look far healthier than this"
        )
    elif ttft_share > 0.5:
        regime = "ttft-bound"
        verdict = (
            f"first-token latency dominates: {ttft_share * 100:.0f}% of request time "
            f"elapses before the first token"
        )
    else:
        regime = "tpot-bound"
        verdict = "steady decode dominates, with no meaningful tail"

    return LatencyRegime(
        requests=len(metrics),
        ttft_p50_ns=ttft_p50,
        ttft_p99_ns=ttft_p99,
        tpot_p50_ns=tpot_p50,
        tpot_p99_ns=tpot_p99,
        tpot_max_ns=max(tpots),
        tpot_tail_ratio=ratio,
        tail_time_share=tail_share,
        ttft_share=ttft_share,
        regime=regime,
        verdict=verdict,
    )


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------


@dataclass
class Diagnosis:
    """The single answer, with the evidence that produced it."""

    pathology: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    regime: LatencyRegime | None = None
    kv: KVPressure | None = None
    #: The slowest completed request -- the one to look at first.
    worst_request: str = ""

    def format(self) -> str:
        lines = [f"  diagnosis: {self.pathology}", f"  {self.summary}", ""]
        lines += [f"    - {e}" for e in self.evidence]
        return "\n".join(lines)


def diagnose(trace: Trace) -> Diagnosis:
    """Name the pathology, or say there isn't one.

    Ordering matters: KV pressure is checked first because it *also* produces
    long queue times, and would otherwise be misread as an admission problem.
    """
    metrics = [m for m in per_request(trace).values() if m.completed]
    if not metrics:
        raise ValueError("trace contains no completed requests")

    kv = kv_pressure(trace)
    regime = latency_regime(trace)
    total = sum(m.total_ns for m in metrics) or 1

    queue_share = sum(m.queue_ns for m in metrics) / total
    scheduling_share = sum(m.decode_wait_ns + m.stall_ns for m in metrics) / total
    # Slowest, not most-scheduler-bound. Picking the latter meant that on a
    # healthy run the CLI printed "diagnosis: healthy" directly above a
    # request-level verdict blaming the scheduler -- true of that request, but
    # a contradiction to read.
    worst = max(metrics, key=lambda m: m.total_ns)

    evidence = [
        f"queued before admission: {queue_share * 100:.1f}% of total request time",
        f"prefilled-but-unscheduled or stalled: {scheduling_share * 100:.1f}%",
        f"decode time in tail steps (>{TAIL_MULTIPLE:.0f}x median): "
        f"{regime.tail_time_share * 100:.1f}%",
        f"preemptions: {kv.preemptions}, prefill goodput: {kv.goodput_ratio * 100:.0f}%",
    ]

    if kv.preemptions and kv.goodput_ratio < GOODPUT_FLOOR:
        pathology = "kv-pressure"
        summary = (
            f"The KV pool cannot hold the working set. {kv.preemptions} preemptions forced "
            f"{kv.recomputed_tokens} tokens of recompute, wasting "
            f"{(1 - kv.goodput_ratio) * 100:.0f}% of prefill work."
        )
    elif scheduling_share > SCHEDULING_SHARE:
        pathology = "prefill-starvation"
        summary = (
            f"Requests are waiting on the scheduler rather than computing: "
            f"{scheduling_share * 100:.0f}% of request time is spent prefilled-but-unscheduled "
            f"or stalled mid-decode, while long prompts monopolise iterations."
        )
    elif queue_share > QUEUE_SHARE:
        pathology = "admission-starvation"
        summary = (
            f"Admission is the bottleneck: {queue_share * 100:.0f}% of request time is spent "
            f"queued before the engine will take the request, with no KV pressure to justify it."
        )
    else:
        pathology = "healthy"
        summary = (
            "No scheduling pathology detected: time is going into decode, batches are "
            "filling, and the KV pool has headroom."
        )

    return Diagnosis(
        pathology=pathology,
        summary=summary,
        evidence=evidence,
        regime=regime,
        kv=kv,
        worst_request=worst.request_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="SQLite trace written by a Tracer")
    parser.add_argument("--request", default="", help="explain one request in detail")
    args = parser.parse_args()

    trace = Trace.from_sqlite(args.db)
    print(f"\n  {trace!r}\n")

    d = diagnose(trace)
    print(d.format())
    print()
    assert d.regime is not None and d.kv is not None
    print(d.regime.format())
    print()
    print(d.kv.format())
    print()

    target = args.request or d.worst_request
    if target:
        print(explain_latency(trace, target).format())
        print()
        print(batch_context(trace, target).format())
    print()


if __name__ == "__main__":
    main()
