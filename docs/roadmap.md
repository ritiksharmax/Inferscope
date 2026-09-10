# Roadmap

## Phase 1 — core tracing primitives *(done)*

Event schema, interner, per-thread buffers, flush thread, memory/null/SQLite sinks, overhead
benchmark. Exit criterion was `span.mark()` under 100 ns: it lands at 86 ns.

## Phase 2 — `inferscope_lab`, the validation target *(done)*

A continuous-batching engine with iteration-level scheduling, a fixed KV block pool, preemption
with real recompute, two interchangeable model runners (a cost model and a real HF causal LM),
and four reproducible pathologies. See [lab.md](lab.md).

Exit criterion was that a pathology run produce traces visibly different from a healthy run. It
does, and each scenario's signature is asserted in `tests/test_lab_pathologies.py`.

Building it paid for itself immediately by breaking the library three times:

- The default decode mode averaged scheduling stalls away entirely — 2.5x TPOT p99/p50 against
  full mode's 39x. Fixed by splitting runs on stalls and emitting `DECODE_STALL` events.
- A run-length off-by-one made single-step runs report a mean step time of zero, collapsing the
  adaptive threshold and producing 374 false stalls in the `healthy` scenario.
- `SQLiteSink` pinned its connection to the flush thread, so `close()` raised on every ordinary
  use. Every sqlite test until then had used `autostart=False` and never crossed threads.

None of those were reachable from unit tests written against the tracer alone.

## Phase 3 — derived metrics and the four questions *(done)*

`trace.py` reads a recording back, `metrics.py` reconstructs each request's timeline into a
breakdown that sums to its wall time, and `query.py` answers the four questions and names the
pathology. See [queries.md](queries.md).

Exit criterion was that each query, run against a Phase 2 pathology, name the pathology actually
injected. It does, in both `aggregate` and `full` decode modes.

Three things the reference workloads corrected:

- The breakdown reached **143% of wall time** for preempted requests: a `DECODE_STALL` spanning a
  preemption was counted both as a stall and as requeue time. Stall duration is now measured from
  the reader's own anchor rather than from the tracer's reported gap.
- `PREFILL_END → FIRST_TOKEN` was attributed to nothing, leaving up to **72% of a request
  unexplained**. It is the time a request sits prefilled but unscheduled, and under
  prefill-priority it is where a starved request's latency actually goes — now its own term.
- TPOT p99/p50 was the intended tail statistic and is unusable: stalls are ~1% of steps, so the
  same scenario scored 39x and 1.3x on consecutive runs. Replaced by the share of decode *time*
  in steps over 8x the median, stable at 30%.

## Phase 4 — export and dashboard *(done)*

An OTel sink mapping the event stream onto span trees, and a FastAPI + single-page dashboard that
serves live and recorded traces alike. See [export.md](export.md).

Closed a gap this exposed: event timestamps are monotonic with an arbitrary origin, so exported
as-is every trace would land in 1970. The collector now captures the offset to wall time once and
passes it to sinks via `describe()`.

## Phase 5 — end-to-end benchmark and the debugging story *(done)*

`benchmarks/e2e.py` measures instrumented against uninstrumented serving of a real model
(Qwen2.5-0.5B-Instruct on MPS), and [debugging-story.md](debugging-story.md) walks the whole arc:
an incident invisible to request-level metrics, diagnosed from a trace, fixed, and verified.

The real-model result is that overhead is **not measurable** — 0.15% ± 0.46%, an interval
containing zero — because a decode step costs ~7 ms against ~100 ns of instrumentation. That is
the honest finding, and the reason the synthetic benchmark exists.

Writing the story required implementing **chunked prefill** in the lab, because none of the
existing knobs actually fixed the pathology and a story that ends in advice is not a story. That
in turn exposed two more bugs:

- Partially-prefilled requests hold KV blocks while appearing in neither `running` nor `waiting`,
  so they were invisible to victim selection and the pool could deadlock. `run_workload` also had
  no iteration cap, so it spun forever instead of failing loudly.
- A preemption or stall split leaves the run counter at zero, and the next batch-size change
  emitted an **empty** `DECODE_RUN` sharing a timestamp with the real one. Events sort by payload,
  so the empty run landed first, moved the reader's anchor onto the real run's own timestamp, and
  charged 9 ms of genuine decoding as instantaneous.

## Where it stands

All five phases are done. Open, and deliberately not on the critical path:

- A vLLM adapter, written against its API and validated in a single rented-GPU session. The
  scheduler hook points here map onto vLLM's fairly directly; the work is in its internals
  churning, not in the data model.
- An upstream hook-point PR to vLLM or SGLang, worth attempting now the local story is solid.
- Thresholds in `query.py` are calibrated against one engine's behaviour. They are a starting
  point for a real deployment, not a universal constant; the underlying metrics are the durable
  part.
