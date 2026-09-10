# Design

## The shape of the problem

An inference server's latency is not one number, it is a stack of them. A request waits in a
queue, gets admitted to a batch, is prefilled, then emits tokens one step at a time, sharing
each step with whatever else the scheduler put in that batch — and possibly getting preempted
and recomputed along the way when KV blocks run short.

Generic tracing sees a single `request_duration` span over all of that. Every interesting
question lives *inside* the span:

| question | what you need to have recorded |
|---|---|
| queueing or compute? | the boundary between admission and prefill |
| stuck behind a prefill? | which requests shared each step, and their prefill token counts |
| KV pressure? | allocations, evictions, preemptions, and recomputed tokens |
| TTFT or TPOT bound? | the first-token instant, separately from the per-step cadence |

So the data model is built around those four, and nothing else.

## Flat events, derived offline

Every observation is a fixed-arity tuple:

```
(ts_ns, kind, req, batch, a, b)
```

`a` and `b` are generic integer payload slots whose meaning depends on `kind`
(see `EventKind` in `src/inferscope/events.py`). Request and batch ids are interned to dense
integers on entry, so the buffer stays homogeneous and small.

Nothing is derived, correlated, or aggregated on the hot path. TTFT, TPOT, queue time, batch
composition and KV occupancy are all reconstructed after the fact by walking the event stream.
This is the single most important decision in the library: it is what lets an instrumentation
call be a timestamp and an append, and it is what makes the overhead number defensible.

The cost is that the stream is only useful once you have all of it — a half-collected trace
gives half-answers. That is the right trade for an observability tool, which is read far less
often than it is written.

## The hot path

`span.mark()` and `span.decode_step()` are closures built per span rather than methods. Binding
the buffer, its `append`, the capacity and the request index as closure default arguments makes
every access a `LOAD_FAST` instead of an attribute chain. Measured, that is the difference
between ~74 ns and ~88 ns per call — worth the mild oddity on the two hottest calls in the
library, and used nowhere else.

Everything else about the hot path follows from "never make the engine wait":

- **Never block.** Append to a plain list owned by the calling thread. No locks, no I/O, no
  serialization.
- **Never stall on backpressure.** When a buffer is at capacity the event is dropped and
  counted (`tracer.dropped_events`). An observability library that blocks the engine when its
  disk is slow has failed at its job.
- **Never swap the buffer out from under a producer.** The flush thread drains *in place* with
  `chunk = ev[:n]; del ev[:n]`. Both are atomic list operations; an append racing between them
  lands at an index `>= n` and survives the delete. Producers therefore never synchronize with
  the flush thread at all.

Two things that measurement ruled out, recorded so nobody re-litigates them:

- **Packed binary buffers.** `struct.pack_into` of five fields costs ~56 ns against ~9 ns for a
  tuple append. Packing belongs on the flush thread, not the hot path.
- **`ContextVar` for the ambient span.** ~11 ns to read, but it makes the span implicit, and an
  engine's scheduler loop touches many requests per iteration — implicit context is wrong for
  this shape of code. Spans are passed explicitly.

## Decode steps are the volume problem

A request emits ~5 lifecycle events and one event *per generated token*. At 500 req/s × 200
tokens that is 100k decode events/s against 2.5k of everything else — decode is 97% of the
stream. So decode gets its own recording modes:

| mode | per token | what it keeps |
|---|---|---|
| `full` | 144 ns | every token's exact timestamp |
| `aggregate` (default) | 107 ns | run lengths, batch sizes, **and stalls** |
| `coarse` | 74 ns | run lengths and batch sizes only |

In `aggregate` and `coarse`, contiguous steps at the same batch size collapse into a single
`DECODE_RUN` event carrying `(n_steps, batch_size)`. The sink sees O(runs), not O(tokens).

`DECODE_RUN` carries only its *end* timestamp, so a run costs one event rather than two. Its
start is the most recent preceding **`DECODE_ANCHOR`** event for the same request —
`FIRST_TOKEN`, `DECODE_STEP`, `DECODE_RUN`, `DECODE_STALL` or `PREFILL_END`.

That set is load-bearing and was got wrong once already. A request emits other events
mid-flight (`KV_ALLOC` when it needs a block, `BATCH_MEMBER` once per iteration), and those land
*between* the stall and the next token. Anchoring a run on "the previous event of any kind"
therefore measures from a mid-iteration `KV_ALLOC` and charges a 30 ms starvation stall as a
0.8 ms decode step — the pathology disappears from the data that exists to show it.
`PREFILL_END` is in the set so the first token after a preemption is measured from the recompute
finishing, not from the token before the request was evicted.

### Aggregation has to preserve stalls, or it is worthless

Collapsing runs on batch-size changes alone — the `coarse` mode — destroys the thing the library
is for. Measured against the `prefill-starvation` scenario, where a burst of long prompts stops
short requests from emitting tokens for ~25 ms:

| mode | decode events | TPOT p99/p50 |
|---|---|---|
| `full` | 1170 | **39.4x** |
| `aggregate` | 232 | **40.0x** |
| `coarse` | 228 | 2.5x |

A stall inside a run at constant batch size gets averaged across that run's tokens: 25 ms spread
over 48 tokens raises mean TPOT by 0.5 ms and vanishes. So `aggregate` also ends a run when a
step takes more than `stall_multiplier` (default 8) times the run's mean, and — critically —
emits the gap as its own `DECODE_STALL` event. Ending the run is *not* sufficient on its own:
the gap would simply land inside the *next* run and be averaged there instead.

The threshold is adaptive and re-armed from each closed run's mean step time, which costs a
division on the cold path and nothing per token. Per-token cost is one subtraction and one
comparison over `coarse`: 107 ns against 74 ns. The default buys back full-fidelity tail
detection at a fifth of full mode's event volume.

### The off-by-one that made it misfire

A run of k tokens spans exactly k intervals *from its anchor*, so mean step time is
`(last_step - anchor) / k`. Keeping that true costs one subtlety: when the anchor is the current
token's own timestamp — true for `FIRST_TOKEN` and for the token that ends a stall — that token
is already accounted for by the anchor event, so the next run must not count it again.

Getting it wrong makes a single-step run report a mean of zero, which drops the adaptive
threshold to its floor and turns every ordinary token into a false stall: the `healthy` scenario
reported 374 of them. With the accounting fixed it reports none.

A reader reconstructs the token count identically in every mode as
`1 (FIRST_TOKEN) + one per DECODE_STALL + sum(DECODE_RUN lengths) + count(DECODE_STEP)`, and
`tests/test_tracer.py` asserts that against the terminal event's total.

## What this design gives up

- **Distributed tracing across processes.** Everything here is single-process. Cross-process
  correlation comes from the OTel exporter, not from the core.
- **Exact ordering across threads.** Per-thread buffers are merged by timestamp at flush.
  `perf_counter_ns` has ~42 ns resolution on macOS, well below any real event spacing, but two
  events from different threads within the same tick have arbitrary relative order.
- **Sub-run decode resolution in the default mode.** Deliberate; use `decode_mode="full"` when
  you need it.
