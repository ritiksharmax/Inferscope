# A debugging story

The point of all of this, start to finish: a latency incident that standard observability cannot
explain, diagnosed from a trace, fixed, and the fix verified.

Everything below is reproducible:

```bash
python -m inferscope_lab.pathologies --inject prefill-starvation --db traces.db
python -m inferscope.query --db traces.db
python -m inferscope.dashboard --db traces.db
```

## 1. The symptom, and why the dashboard is no help

A serving cluster gets a burst of long prompts — a batch job, a retrieval-heavy request pattern,
someone's new prompt template. Users report the service feeling "laggy". You check the metrics.

| what you have | what it says |
|---|---|
| request duration p50 | 68 ms |
| request duration p99 | 79 ms |
| **p99 / p50** | **1.2x** |
| GPU utilisation | high, steady |
| error rate | zero |

Request latency is *flat*. The GPU is busy. Nothing is failing. By every dashboard you own, the
service is healthy — and it is not.

## 2. What the trace says

```
  diagnosis: prefill-starvation
  Requests are waiting on the scheduler rather than computing: 40% of request time is
  spent prefilled-but-unscheduled or stalled mid-decode, while long prompts monopolise
  iterations.

    - queued before admission: 12.0% of total request time
    - prefilled-but-unscheduled or stalled: 40.4%
    - decode time in tail steps (>8x median): 30.0%
    - preemptions: 0, prefill goodput: 100%
```

The same run, measured per token instead of per request:

| view | p99 / p50 |
|---|---|
| end-to-end request latency | 1.2x |
| **per-token latency (TPOT)** | worst step **30.4 ms** against a 0.65 ms median |

That is the whole argument for instrumenting the inference loop. The damage is spread thinly
across every request's *duration* and concentrated sharply in its token *cadence*. A user watching
tokens appear sees them freeze for 30 ms at a time; a dashboard averaging over a 56 ms request
sees nothing.

## 3. Which request, and what it was waiting behind

```
  r23: 79.0 ms total
    queued           15.7 ms   19.9%  ########
    prefill           0.5 ms    0.6%  #
    decode_wait      32.8 ms   41.6%  #################
    decoding         30.0 ms   37.9%  ###############
    -> waiting on the scheduler, not computing

  r23: 49 iterations, mean batch 8.8
    most often batched with: r22 (47), r21 (45), r20 (43)
    6 prefill iterations ran during its lifetime, prefilling 5184 tokens for other requests
    -> shared the engine with 5184 tokens of other requests' prefill; that is what it was
       waiting behind
```

`decode_wait` — prefilled, admitted, holding KV blocks, and *not being scheduled* — is the single
largest component of this request's life. It spent longer waiting for a decode slot than
generating its entire output. And the culprit is named: 5,184 tokens of other requests' prefill
ran during its lifetime.

Meanwhile `kv_pressure` rules out the other obvious suspect:

```
  KV occupancy: mean 15%, peak 68%
  evictions: 0 (0 blocks), preemptions: 0
  prefill goodput: 100% (0 of 10752 prefill tokens were recompute)
    -> KV pool is not the constraint
```

Not memory. Scheduling.

## 4. The obvious fix makes it worse

If prefill is starving decode, stop prioritising prefill. Switch the scheduler to drain running
requests first:

```
  diagnosis: admission-starvation
```

| | baseline | decode-priority |
|---|---|---|
| worst decode step | 30.4 ms | **0.6 ms** |
| tail time share | 29.7% | **0.0%** |
| TTFT p99 | 48.3 ms | **333.0 ms** |
| request latency p99 | 78.8 ms | **356.7 ms** |
| mean batch size | 11.1 | **1.7** |

The stalls are gone — genuinely, completely gone. And TTFT got **6.8x worse**, batch size
collapsed from 11 to under 2, and end-to-end latency more than quadrupled. New requests now wait
for the running set to drain before they are admitted at all.

The tool does not congratulate you for removing the stalls. It renames the pathology.

## 5. The real fix

The problem was never that prefill runs. It is that a 1,536-token prefill is *indivisible*: once
an iteration starts one, every decoding request waits ~30 ms for it to finish. Capping how many
requests share a prefill batch does not help, because the blocking unit is a single long prompt:

| | worst step | diagnosis |
|---|---|---|
| baseline | 30.4 ms | prefill-starvation |
| `max_prefill_seqs=1` | 31.7 ms | prefill-starvation |
| `max_prefill_tokens=512` | 31.7 ms | prefill-starvation |

The fix is **chunked prefill**: split a long prompt across iterations and run those chunks
*alongside* decode, so a long prefill costs every decoding request a little instead of costing it
everything at once. This is what vLLM and SGLang do, and `inferscope_lab` implements it:

```python
EngineConfig(chunked_prefill=True, chunk_tokens=512)
```

| | baseline | decode-priority | **chunked (512)** |
|---|---|---|---|
| diagnosis | prefill-starvation | admission-starvation | **healthy** |
| worst decode step | 30.4 ms | 0.6 ms | **2.4 ms** |
| tail time share | 29.7% | 0.0% | **0.0%** |
| TTFT p99 | 48.3 ms | 333.0 ms | **51.1 ms** |
| request latency p99 | 78.8 ms | 356.7 ms | **83.2 ms** |
| mean batch size | 11.1 | 1.7 | **10.9** |

Worst decode step down **12x**, tail eliminated, TTFT and batch size essentially unchanged. After
the change, 110 of 111 iterations carry decode work; before, decode simply stopped whenever a long
prompt arrived.

## 6. Chunk size is a real trade-off

Smaller chunks interleave more finely — and starve admission, because each iteration admits less
prefill work. The tool catches that too:

| chunk | worst step | TTFT p99 | mean batch | diagnosis |
|---|---|---|---|---|
| 512 | 2.4 ms | 51.1 ms | 10.9 | healthy |
| 256 | 1.7 ms | 59.0 ms | 10.8 | healthy |
| 128 | 1.3 ms | 81.4 ms | 8.5 | **admission-starvation** |

There is no setting that is best at everything, which is the normal condition for a scheduler. The
value of the trace is that each configuration's cost is *named* rather than inferred.

## What the incident actually required

- Per-token timing, not per-request. The pathology is invisible at request granularity.
- A latency breakdown that **sums to wall time**, so `decode_wait` shows up as 42% of a request
  rather than disappearing into an unattributed remainder.
- Batch composition, to name what the request was waiting behind rather than guessing.
- KV metrics, to rule out the other suspect quickly.
- A statistic robust enough to compare configurations — see
  [queries.md](queries.md#why-the-tail-statistic-is-a-share-of-time-not-a-percentile) for why the
  obvious one, TPOT p99/p50, was not.

Every table on this page is asserted in `tests/test_lab_chunked.py`, including that the obvious fix
trades one pathology for another and that over-chunking starves admission. If a change to the
scheduler or the metrics breaks this story, the suite fails.
