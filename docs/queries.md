# The four questions

```bash
python -m inferscope_lab.pathologies --inject prefill-starvation --db traces.db
python -m inferscope.query --db traces.db
```

```
  diagnosis: prefill-starvation
  Requests are waiting on the scheduler rather than computing: 40% of request time is spent
  prefilled-but-unscheduled or stalled mid-decode, while long prompts monopolise iterations.

    - queued before admission: 12.0% of total request time
    - prefilled-but-unscheduled or stalled: 40.4%
    - decode time in tail steps (>8x median): 30.0%
    - preemptions: 0, prefill goodput: 100%

  r23: 79.0 ms total
    queued           15.7 ms   19.9%  ########
    prefill           0.5 ms    0.6%  #
    decode_wait      32.8 ms   41.6%  #################
    decoding         30.0 ms   37.9%  ###############
    -> waiting on the scheduler, not computing

  r23: 49 iterations, mean batch 8.8
    6 prefill iterations ran during its lifetime, prefilling 5184 tokens for other requests
    -> shared the engine with 5184 tokens of other requests' prefill; that is what it was
       waiting behind
```

## `explain_latency(trace, request_id)` — queueing or compute?

Decomposes one request's wall time into terms that **sum to the whole**, with whatever is left
over reported as `unattributed` rather than hidden:

| term | meaning |
|---|---|
| `queued` | arrival → the engine first does work for it |
| `prefill` | inside prefill, including recompute after preemption |
| `decode_wait` | prefilled, waiting for a decode slot |
| `decoding` | actually producing tokens |
| `stalled` | scheduled out mid-decode |
| `requeued` | preempted → re-admitted |

`decode_wait` earns its place: in the starvation workload it is 42% of the worst request's life.
Before it existed that time was simply unattributed, at up to 72% of a request — a breakdown with
a three-quarters-empty middle is not a breakdown.

For a request that was never preempted, `ttft == queued + prefill + decode_wait` **exactly**, and
a test asserts it. After a preemption `prefill` also carries recompute that happened long after
the first token, and the identity stops holding.

## `batch_context(trace, request_id)` — what was it batched with?

Co-residency per iteration, plus the prefill work done *for other requests* during this one's
lifetime. That second number is the direct answer to "was it stuck behind a huge prefill": 5184
tokens, in the run above.

## `kv_pressure(trace)` — is the cache the constraint?

Occupancy, evictions, preemptions, and **prefill goodput** — the share of prefill work that was
not thrown away and recomputed. The kv-thrash workload runs at 74%: a quarter of everything the
engine prefilled was rebuilding context it had already built and evicted.

## `latency_regime(trace)` — TTFT or TPOT bound?

Reports TTFT and TPOT percentiles, and classifies the regime as `ttft-bound`, `tpot-bound` or
`tpot-tail`.

### Why the tail statistic is a share of time, not a percentile

The obvious statistic is TPOT p99/p50. It is a trap here. Stalls are ~1% of decode steps in the
reference workload, which puts a 99th percentile exactly on the boundary — the same scenario
scored **39x on one run and 1.3x on the next**, purely from which side of the index the stall
cluster landed.

The share of decode *time* spent in steps over 8x the median is stable at ~30%, because each
stall is worth roughly 40 ordinary steps. Measured across both decode modes:

| scenario | tail time share | p99/p50 |
|---|---|---|
| healthy | 0.0% | 1.2 – 1.5 |
| prefill-starvation | 29.6 – 30.0% | 1.3 – 1.4 |
| kv-thrash | 0.0% | 3.2 – 3.3 |
| batch-starvation | 0.0% | 1.0 – 1.9 |

The percentile is still reported, because people expect it. It is not what decides anything.

## `diagnose(trace)` — the single answer

Runs all four and names the pathology. Checks are **ordered**, and the order matters: KV pressure
is tested first because it also produces long queue times and would otherwise be misread as an
admission problem.

| check | threshold | verdict |
|---|---|---|
| preemptions > 0 and prefill goodput < 95% | `GOODPUT_FLOOR` | `kv-pressure` |
| `(decode_wait + stalled) / total` > 20% | `SCHEDULING_SHARE` | `prefill-starvation` |
| `queued / total` > 40% | `QUEUE_SHARE` | `admission-starvation` |
| otherwise | | `healthy` |

### Calibration

Thresholds come from the reference workloads, not from taste. Measured over six runs each:

| scenario | queued | scheduling share | preemptions | goodput | diagnosis |
|---|---|---|---|---|---|
| healthy | 0.0% | **9.1 – 9.6%** | 0 | 100% | `healthy` |
| prefill-starvation | 10.6% | **39.4 – 40.7%** | 0 | 100% | `prefill-starvation` |
| kv-thrash | 41.0% | 1.5% | 4 – 7 | 74 – 77% | `kv-pressure` |
| batch-starvation | 92.8% | 0.2% | 0 | 100% | `admission-starvation` |

The `SCHEDULING_SHARE` threshold of 20% sits about 2x clear of both healthy's ceiling and
starvation's floor. `tests/test_query.py` asserts every scenario is diagnosed correctly in both
`aggregate` and `full` decode modes — the numbers above must not depend on how decode was
recorded, and they do not.

These thresholds are tuned against one engine's behaviour. They are a starting point for a real
deployment, not a universal constant; the underlying metrics are the durable part.
