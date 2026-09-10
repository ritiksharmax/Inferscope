"""OpenTelemetry exporter.

Turns the event stream into spans so a trace lands in Jaeger, Tempo or whatever
else is already deployed. Requires the ``otel`` extra; import this module only
when you want it.

The impedance mismatch is per-token data. OTel's model is nested intervals, and
a span per generated token would produce 200 spans per request and swamp any
backend. So decode becomes *one* span carrying the token statistics as
attributes, with stalls, preemptions and evictions as span *events* on it --
which is what span events are for.

Spans are built only when a request terminates, because a span needs an end
time and events arrive incrementally. That means a request still in flight is
not yet exported, and one that never terminates never is: ``close()`` flushes
whatever is still pending as spans marked incomplete, so nothing is silently
dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inferscope.events import NONE_IDX, Event, EventKind
from inferscope.metrics import per_request
from inferscope.sinks.base import Sink
from inferscope.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

#: Requests buffered awaiting a terminal event, before the oldest are given up
#: on. A request that never completes must not pin memory forever.
DEFAULT_MAX_PENDING = 10_000

_TERMINAL = frozenset({EventKind.COMPLETE, EventKind.FAILED})


def _default_provider() -> TracerProvider:
    """A provider exporting over OTLP, honouring the standard env vars."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError as exc:  # pragma: no cover - depends on which extra
        raise RuntimeError(
            "OTelSink needs an exporter: pip install 'inferscope[otel]', "
            "or pass your own tracer_provider"
        ) from exc

    provider = TracerProvider(
        resource=Resource.create({"service.name": "inferscope"})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


class OTelSink(Sink):
    """Emits one span tree per request: queue, prefill(s), decode."""

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        *,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        self._provider = tracer_provider or _default_provider()
        self._tracer = self._provider.get_tracer("inferscope")
        self._pending: dict[int, list[Event]] = {}
        self._names: dict[int, str] = {}
        self._epoch_offset_ns = 0
        self._max_pending = max_pending
        self.spans_emitted = 0
        self.requests_abandoned = 0

    # -- Sink ---------------------------------------------------------------

    def describe(self, meta: dict[str, str]) -> None:
        self._epoch_offset_ns = int(meta.get("epoch_offset_ns", 0))

    def write(self, events: list[Event], names: list[tuple[int, str]]) -> None:
        self._names.update(names)
        finished: list[int] = []
        for event in events:
            req = event[2]
            if req == NONE_IDX:
                continue  # batch- and engine-level events have no span of their own
            self._pending.setdefault(req, []).append(event)
            if event[1] in _TERMINAL:
                finished.append(req)

        for req in finished:
            self._emit(req, self._pending.pop(req, []), complete=True)

        self._evict_if_needed()

    def flush(self) -> None:
        self._provider.force_flush()

    def close(self) -> None:
        for req, events in list(self._pending.items()):
            self._emit(req, events, complete=False)
        self._pending.clear()
        self.flush()
        self._provider.shutdown()

    # -- internals ----------------------------------------------------------

    def _evict_if_needed(self) -> None:
        if len(self._pending) <= self._max_pending:
            return
        # dicts iterate in insertion order, so the oldest are first
        overflow = len(self._pending) - self._max_pending
        for req in list(self._pending)[:overflow]:
            self._emit(req, self._pending.pop(req), complete=False)
            self.requests_abandoned += 1

    def _wall(self, ts_ns: int) -> int:
        return ts_ns + self._epoch_offset_ns

    def _emit(self, req: int, events: list[Event], *, complete: bool) -> None:
        if not events:
            return
        name = self._names.get(req)
        if name is None:
            return  # a BATCH_MEMBER reference to a request we never saw start

        # Reuse the reader's timeline walk rather than reimplementing it here:
        # one definition of what "prefill time" means, not two.
        sub = Trace(list(events), {req: name}, self._epoch_offset_ns)
        metrics = per_request(sub).get(name)
        if metrics is None:
            return

        start = self._wall(events[0][0])
        end = self._wall(events[-1][0])

        root = self._tracer.start_span("inference.request", start_time=start)
        root.set_attribute("inferscope.request_id", name)
        root.set_attribute("inferscope.prompt_tokens", metrics.prompt_tokens)
        root.set_attribute("inferscope.output_tokens", metrics.output_tokens)
        root.set_attribute("inferscope.complete", complete and metrics.completed)
        if metrics.ttft_ns:
            root.set_attribute("inferscope.ttft_ms", metrics.ttft_ns / 1e6)
        if metrics.tpot_samples:
            root.set_attribute("inferscope.tpot_mean_ms", metrics.tpot_mean_ns / 1e6)
            root.set_attribute("inferscope.tpot_p99_ms", metrics.tpot_p99_ns / 1e6)
        if metrics.preemptions:
            root.set_attribute("inferscope.preemptions", metrics.preemptions)
            root.set_attribute("inferscope.recomputed_tokens", metrics.recomputed_tokens)
        if metrics.failed:
            root.set_status(self._error_status("request failed"))

        self._child_spans(root, events, metrics)
        root.end(end_time=end)
        self.spans_emitted += 1

    def _error_status(self, description: str) -> Any:
        from opentelemetry.trace import Status, StatusCode

        return Status(StatusCode.ERROR, description)

    def _child_spans(self, root: Any, events: list[Event], metrics: Any) -> None:
        from opentelemetry import trace as otel_trace

        ctx = otel_trace.set_span_in_context(root)
        first_ts = events[0][0]
        prefill_open: int | None = None
        preempted_at: int | None = None
        queue_emitted = False
        decode_span: Any = None

        for ts, kind, _req, _batch, a, b in events:
            wall = self._wall(ts)

            if kind == EventKind.PREFILL_START:
                if not queue_emitted:
                    # Only the first prefill closes the initial queue. A resumed
                    # request prefills again, and keying off `prefill_open`
                    # emitted a second "queue" span covering the whole request.
                    queue_emitted = True
                    if metrics.queue_ns:
                        queued = self._tracer.start_span(
                            "queue", context=ctx, start_time=self._wall(first_ts)
                        )
                        queued.end(end_time=wall)
                elif preempted_at is not None:
                    requeued = self._tracer.start_span(
                        "requeue", context=ctx, start_time=self._wall(preempted_at)
                    )
                    requeued.end(end_time=wall)
                    preempted_at = None
                prefill_open = ts
            elif kind == EventKind.PREFILL_END:
                if prefill_open is not None:
                    span = self._tracer.start_span(
                        "prefill", context=ctx, start_time=self._wall(prefill_open)
                    )
                    span.set_attribute("inferscope.tokens", a)
                    span.end(end_time=wall)
                    prefill_open = None
            elif kind == EventKind.FIRST_TOKEN:
                decode_span = self._tracer.start_span(
                    "decode", context=ctx, start_time=wall
                )
                decode_span.set_attribute("inferscope.batch_size_at_first_token", b)
            elif kind == EventKind.PREEMPTED:
                # Handled outside any decode-span guard: a request can be
                # evicted after prefill but before its first token, and keying
                # this off the decode span lost those requeues entirely.
                preempted_at = ts
                if decode_span is not None:
                    decode_span.add_event(
                        "preempted",
                        {"inferscope.blocks_freed": a, "inferscope.tokens_discarded": b},
                        timestamp=wall,
                    )
            elif decode_span is not None:
                if kind == EventKind.DECODE_STALL:
                    decode_span.add_event(
                        "stall",
                        {"inferscope.stall_ms": a / 1e6, "inferscope.batch_size": b},
                        timestamp=wall,
                    )
                elif kind == EventKind.KV_EVICT:
                    decode_span.add_event(
                        "kv_evict", {"inferscope.blocks_freed": a}, timestamp=wall
                    )
                elif kind == EventKind.RESUMED:
                    decode_span.add_event(
                        "resumed", {"inferscope.tokens_recomputed": a}, timestamp=wall
                    )

        if decode_span is not None:
            decode_span.set_attribute("inferscope.output_tokens", metrics.output_tokens)
            if metrics.stalls:
                decode_span.set_attribute("inferscope.stalls", len(metrics.stalls))
                decode_span.set_attribute(
                    "inferscope.stall_total_ms", metrics.stall_ns / 1e6
                )
            decode_span.end(end_time=self._wall(events[-1][0]))
