# Export and the dashboard

## Wall-clock anchoring

Event timestamps are `perf_counter_ns`: monotonic, with an arbitrary origin. That is the right
clock to *measure* with and a useless one to correlate with anything else — exported as-is, every
trace would land in 1970 and line up with no incident, log line or metric.

So the collector captures `time.time_ns() - perf_counter_ns()` once at construction and hands it
to the sink before the first write. SQLite stores it in the `meta` table; `Trace.wall_ns(ts)`
converts. Everything that needs real time uses it, and nothing on the hot path pays for it.

## OpenTelemetry

```python
from inferscope import Tracer
from inferscope.sinks.otel import OTelSink

tracer = Tracer(OTelSink())          # OTLP, honouring the standard env vars
tracer = Tracer("otel")              # same thing via the sink spec
tracer = Tracer(OTelSink(provider))  # or bring your own TracerProvider
```

Each request becomes one span tree:

```
inference.request          request_id, prompt/output tokens, ttft_ms, tpot_mean_ms, tpot_p99_ms
├── queue                  arrival -> first prefill
├── prefill                one per prefill, including recompute after preemption
├── requeue                one per preemption: evicted -> re-admitted
└── decode                 first token -> terminal
      events: stall, preempted, kv_evict, resumed
```

### The impedance mismatch

OTel's model is nested intervals. A span per generated token would mean 200 spans per request and
would swamp any backend, so decode is **one** span carrying token statistics as attributes, with
stalls, preemptions and evictions as span *events* on it. That is what span events are for, and it
is the one place the event stream does not map cleanly onto OTel's shape.

### Two things worth knowing

A span needs an end time, so spans are built only when a request **terminates**. A request still
in flight has not been exported yet; `close()` flushes whatever is pending as spans marked
`inferscope.complete=false`, so nothing is silently dropped. Pending requests are capped
(`max_pending`, default 10k) and the oldest are given up on first — a request that never
terminates must not pin memory forever.

Preemption is handled outside the decode span's scope on purpose. A request can be evicted after
prefill but *before* its first token; keying that off the decode span lost those requeues
entirely, which the span-count assertions in `tests/test_sink_otel.py` now catch.

## Dashboard

```bash
python -m inferscope.dashboard --db traces.db
```

A FastAPI app and one self-contained HTML page — no build step, no CDN. It re-reads the SQLite
file per request rather than caching, so it serves a live trace an engine is still writing to
exactly like a finished one.

Panels: the diagnosis banner, engine tiles, a timeline (decode batch size, prefill tokens, KV
occupancy, stall markers), TTFT and TPOT histograms, and the slowest requests with their latency
broken down as a stacked bar. Clicking a row drills into that request.

Histograms are **log-spaced**. Linear bins put 99% of an inference workload's steps in the first
bucket and the interesting tail in a bar one pixel tall.

`dashboard/data.py` computes everything without reference to HTTP, so the panels are testable
directly and a notebook can call `build_payload` and get the numbers the page shows.

### Testing a page without a browser

`tests/test_dashboard.py` runs the page's own JavaScript under a minimal DOM shim in node, against
real payloads from all four scenarios. The shim's `setAttribute` rejects `NaN` and `undefined`,
which is the class of bug static checks miss: a chart that divides by zero on an empty series, or
writes a `NaN` coordinate into an SVG path. The test skips if node is not installed.
