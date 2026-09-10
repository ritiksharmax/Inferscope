"""Everything the dashboard shows, computed without reference to HTTP.

Kept separate from the server so the panels can be tested directly, and so a
notebook can call ``build_payload`` and get the same numbers the page shows.
"""

from __future__ import annotations

from typing import Any

from inferscope.events import EventKind as K
from inferscope.metrics import per_request, window_metrics
from inferscope.query import batch_context, diagnose, explain_latency
from inferscope.trace import Trace

BUCKETS = 120


def _histogram(values: list[float], bins: int = 28) -> dict[str, Any]:
    """A log-spaced histogram, because latency distributions have long tails.

    Linear bins put 99% of an inference workload's steps in the first bucket
    and the interesting tail in a bar one pixel tall.
    """
    if not values:
        return {"edges": [], "counts": [], "log": True}
    import math

    lo = max(min(values), 1e-3)
    hi = max(max(values), lo * 1.001)
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    width = (log_hi - log_lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((math.log10(max(v, lo)) - log_lo) / width) if width else 0
        counts[min(bins - 1, max(0, idx))] += 1
    edges = [10 ** (log_lo + i * width) for i in range(bins + 1)]
    return {"edges": edges, "counts": counts, "log": True}


def _timeline(trace: Trace) -> dict[str, Any]:
    """Bucket the run into a fixed number of slices for the timeline chart."""
    if not trace.events:
        return {"t": [], "batch": [], "prefill_tokens": [], "kv": [], "stalls": []}

    start = trace.events[0][0]
    span = max(1, trace.events[-1][0] - start)
    width = span / BUCKETS

    batch_sum = [0.0] * BUCKETS
    batch_n = [0] * BUCKETS
    prefill = [0.0] * BUCKETS
    kv_sum = [0.0] * BUCKETS
    kv_n = [0] * BUCKETS
    stalls = [0] * BUCKETS

    for ts, kind, _req, _batch, a, b in trace.events:
        i = min(BUCKETS - 1, int((ts - start) / width))
        if kind == K.BATCH:
            if b:
                batch_sum[i] += b
                batch_n[i] += 1
            prefill[i] += a
        elif kind == K.KV_USAGE and b:
            kv_sum[i] += a / b
            kv_n[i] += 1
        elif kind == K.DECODE_STALL:
            stalls[i] += 1

    return {
        "t": [i * width / 1e6 for i in range(BUCKETS)],
        "batch": [batch_sum[i] / batch_n[i] if batch_n[i] else 0.0 for i in range(BUCKETS)],
        "prefill_tokens": prefill,
        "kv": [kv_sum[i] / kv_n[i] if kv_n[i] else 0.0 for i in range(BUCKETS)],
        "stalls": stalls,
    }


def build_payload(trace: Trace, top: int = 40) -> dict[str, Any]:
    """The whole dashboard, as one JSON-serialisable dict."""
    metrics = per_request(trace)
    done = [m for m in metrics.values() if m.completed or m.failed]
    window = window_metrics(trace)

    payload: dict[str, Any] = {
        "trace": {
            "events": len(trace),
            "requests": len(done),
            "span_ms": trace.span_ns / 1e6,
            "epoch_offset_ns": trace.epoch_offset_ns,
        },
        "window": {
            "iterations": window.iterations,
            "prefill_iterations": window.prefill_iterations,
            "decode_iterations": window.decode_iterations,
            "mean_batch_size": round(window.mean_batch_size, 2),
            "max_batch_size": window.max_batch_size,
            "mean_kv_occupancy": round(window.mean_kv_occupancy, 3),
            "peak_kv_occupancy": round(window.peak_kv_occupancy, 3),
            "preemptions": window.preemptions,
            "recomputed_tokens": window.recomputed_tokens,
            "goodput_ratio": round(window.goodput_ratio, 3),
            "padding_ratio": round(window.padding_ratio, 3),
        },
        "timeline": _timeline(trace),
    }

    if not done:
        payload["diagnosis"] = {
            "pathology": "no-data",
            "summary": "No completed requests in this trace yet.",
            "evidence": [],
        }
        payload["ttft"] = _histogram([])
        payload["tpot"] = _histogram([])
        payload["requests"] = []
        return payload

    d = diagnose(trace)
    assert d.regime is not None
    payload["diagnosis"] = {
        "pathology": d.pathology,
        "summary": d.summary,
        "evidence": d.evidence,
        "worst_request": d.worst_request,
        "regime": d.regime.regime,
        "regime_verdict": d.regime.verdict,
        "ttft_p50_ms": d.regime.ttft_p50_ns / 1e6,
        "ttft_p99_ms": d.regime.ttft_p99_ns / 1e6,
        "tpot_p50_ms": d.regime.tpot_p50_ns / 1e6,
        "tpot_p99_ms": d.regime.tpot_p99_ns / 1e6,
        "tpot_max_ms": d.regime.tpot_max_ns / 1e6,
        "tail_time_share": d.regime.tail_time_share,
    }

    payload["ttft"] = _histogram([m.ttft_ns / 1e6 for m in done if m.ttft_ns])
    payload["tpot"] = _histogram([x / 1e6 for m in done for x in m.tpot_samples])

    slowest = sorted(done, key=lambda m: -m.total_ns)[:top]
    payload["requests"] = [
        {
            "id": m.request_id,
            "total_ms": m.total_ns / 1e6,
            "prompt_tokens": m.prompt_tokens,
            "output_tokens": m.output_tokens,
            "ttft_ms": m.ttft_ns / 1e6,
            "tpot_mean_ms": m.tpot_mean_ns / 1e6,
            "preemptions": m.preemptions,
            "parts": {
                "queued": m.queue_ns / 1e6,
                "prefill": m.prefill_ns / 1e6,
                "decode_wait": m.decode_wait_ns / 1e6,
                "decoding": m.decode_ns / 1e6,
                "stalled": m.stall_ns / 1e6,
                "requeued": m.requeue_ns / 1e6,
            },
        }
        for m in slowest
    ]
    return payload


def request_detail(trace: Trace, request_id: str) -> dict[str, Any]:
    """The drill-down for one request."""
    breakdown = explain_latency(trace, request_id)
    context = batch_context(trace, request_id)
    return {
        "id": request_id,
        "total_ms": breakdown.total_ns / 1e6,
        "components": {k: v / 1e6 for k, v in breakdown.components.items()},
        "dominant": breakdown.dominant,
        "verdict": breakdown.verdict,
        "iterations": context.iterations,
        "mean_batch_size": context.mean_batch_size,
        "co_residents": context.co_residents[:10],
        "prefill_iterations_during": context.prefill_iterations_during,
        "prefill_tokens_during": context.prefill_tokens_during,
        "batch_verdict": context.verdict,
    }
