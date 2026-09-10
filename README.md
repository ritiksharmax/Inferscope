# inferscope

**OpenTelemetry for LLM inference internals.**

Standard observability treats inference as a black box. Prometheus, Datadog and OTel give you
`request_duration` and `gpu_util%`, which cannot tell you *why* p99 latency spiked. inferscope
instruments the inference loop itself — request lifecycle, batch formation, KV-cache pressure —
so you can answer the questions you actually have:

1. Was this request slow from **queueing** or from **compute**?
2. What was the **batch composition** when it ran — was it stuck behind a huge prefill?
3. Is **KV-cache pressure** causing preemption and recompute?
4. Is the bottleneck **TTFT** (prefill) or **TPOT** (decode)?

## Why the request-level view is not enough

A burst of long prompts hits a server. Every already-running request stops emitting tokens while
the scheduler prefills the burst. Here is that incident, measured two ways:

| view | p99 / p50 |
|---|---|
| end-to-end request latency | **1.1x** — looks fine |
| per-token latency (TPOT) | **40x** — ~25 ms stalls |

The damage is spread thinly across every request's duration and concentrated sharply in its token
cadence. A tool that only records request duration cannot see it. That gap is the whole argument
for instrumenting the inference loop rather than the HTTP handler, and it is reproducible:

```bash
python -m inferscope_lab.pathologies --inject prefill-starvation --db traces.db
```

The fix is chunked prefill, and the trace says so — worst decode step 30.4 ms → 2.4 ms with TTFT
and batch size intact, while the *obvious* fix (drain decode first) removes the stalls and makes
TTFT p99 nearly 7x worse. The whole walkthrough:
[docs/debugging-story.md](docs/debugging-story.md).

## Install

```bash
pip install inferscope
```

## Use

```python
from inferscope import Tracer, EventKind as K

tracer = Tracer("sqlite:///traces.db")

with tracer.trace_request("req-abc", prompt_tokens=512) as span:
    span.mark(K.QUEUED)
    span.mark(K.PREFILL_START)
    span.mark(K.PREFILL_END, 512)
    while generating:
        span.decode_step(batch_size=current_batch_size)
    # COMPLETE emitted on exit -- or FAILED, if the body raises

# from your scheduler, once per iteration
tracer.record_batch("iter-1041", ["req-abc", "req-def"],
                    prefill_tokens=512, decode_tokens=2)
tracer.record_kv_usage(blocks_used=1900, blocks_total=2048)
tracer.record_kv_event(K.KV_EVICT, "req-abc", blocks=4)
```

That is the whole instrumentation surface — small on purpose, because it gets bolted onto engines
written by other people. Everything else (TTFT, TPOT, queue time, batch efficiency, KV hit rate)
is derived offline from the event stream.

Set `INFERSCOPE_DISABLED=1`, or pass `Tracer(enabled=False)`, to turn every call into a no-op
without touching the instrumented code.

## Cost

**On a real model, unmeasurable: 0.15% ± 0.46%** — Qwen2.5-0.5B on MPS, instrumented against
uninstrumented, an interval containing zero. A decode step costs ~7 ms of GPU work against ~100 ns
of instrumentation. On a synthetic engine calibrated to 500 req/s × 200 tokens, where the cost
*can* be resolved: **1.53% ± 0.32%**.

Decode is ~97% of the event stream, so `decode_step` gets three modes:

| mode | per token | keeps |
|---|---|---|
| `full` | 138 ns | every token's exact timestamp |
| `aggregate` *(default)* | 100 ns | run lengths, batch sizes, and stalls |
| `coarse` | 71 ns | run lengths and batch sizes only |

`span.mark()` costs 83 ns. The default mode detects the starvation pathology above as well as
`full` does, on a fifth of the events, because runs split on stalls and the gap is emitted as its
own event rather than averaged across the tokens around it.

Full methodology, confidence intervals and honest limits: [docs/overhead.md](docs/overhead.md).

## Diagnosing a trace

```bash
python -m inferscope.query --db traces.db
```

```
  diagnosis: prefill-starvation
  Requests are waiting on the scheduler rather than computing: 40% of request time is
  spent prefilled-but-unscheduled or stalled mid-decode.

  r23: 79.0 ms total
    queued           15.7 ms   19.9%  ########
    decode_wait      32.8 ms   41.6%  #################
    decoding         30.0 ms   37.9%  ###############
    -> waiting on the scheduler, not computing

    6 prefill iterations ran during its lifetime, prefilling 5184 tokens for others
```

The four questions are `explain_latency`, `batch_context`, `kv_pressure` and `latency_regime`,
with `diagnose` combining them. Thresholds are calibrated against reference workloads with known
answers, and every scenario is asserted to diagnose correctly in both decode modes —
[docs/queries.md](docs/queries.md).

## How it works

Each call records one fixed-arity tuple `(ts_ns, kind, req, batch, a, b)` into a plain list owned
by the calling thread. A background thread drains those buffers into a sink. Nothing is derived,
correlated or aggregated on the hot path — that is what keeps a call at tens of nanoseconds. If a
buffer fills, events are dropped and counted (`tracer.dropped_events`) rather than blocking the
engine.

See [docs/design.md](docs/design.md) for the data model and the measurements behind each decision.

## Export and dashboard

```bash
python -m inferscope.dashboard --db traces.db     # local dashboard, no build step
```

```python
from inferscope.sinks.otel import OTelSink
tracer = Tracer(OTelSink())    # one span tree per request, into Jaeger/Tempo/...
```

Each request exports as `inference.request` with `queue`, `prefill`, `requeue` and `decode`
children; stalls, preemptions and evictions become span events on `decode`. Details and the
impedance mismatch with OTel's model: [docs/export.md](docs/export.md).

## The lab

`inferscope_lab` is a continuous-batching engine built to be instrumented: iteration-level
scheduling, a fixed KV block pool, preemption with real recompute, and four reproducible
pathologies. It runs either against a cost model (deterministic, no GPU) or a real HF causal LM.
See [docs/lab.md](docs/lab.md).

## Status

| phase | scope | state |
|---|---|---|
| 1 | core tracing primitives, sinks, overhead benchmark | done |
| 2 | `inferscope_lab`, a real engine to instrument, and its pathologies | done |
| 3 | derived metrics and the four diagnostic queries | done |
| 4 | OTel exporter and built-in dashboard | done |
| 5 | end-to-end benchmark on a real model, and the debugging story | done |

## Development

```bash
uv venv && uv pip install -e ".[dev,lab]"
uv run pytest
uv run ruff check . && uv run mypy
uv run python -m benchmarks.overhead --micro
```

## License

Apache-2.0
