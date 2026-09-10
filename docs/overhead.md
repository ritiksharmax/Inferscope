# Overhead

An observability library that is expensive is a library nobody turns on in production, so this
number is the project's credibility. It is also the number easiest to quietly cheat on, so the
methodology is written out in full and the benchmark is in the repo.

**On a real model, the cost is not measurable: 0.15% ± 0.46%** — Qwen2.5-0.5B-Instruct on MPS,
serving through `inferscope_lab`, instrumented against uninstrumented over 9 paired runs. The
interval contains zero. A real decode step costs milliseconds; an instrumentation call costs
~100 ns.

**On a synthetic engine calibrated to 500 req/s × 200 output tokens, 1.53% ± 0.32%** — 205
instrumentation calls per request, ~102k calls/s. That benchmark exists precisely because the
real-model measurement cannot resolve the quantity it is trying to measure.

```bash
python -m benchmarks.overhead --micro
python -m benchmarks.overhead --qps 500 --pairs 30
python -m benchmarks.e2e --model Qwen/Qwen2.5-0.5B-Instruct --pairs 9
```

## What was measured, and on what

Apple M4, 16 GB, macOS 15.6, CPython 3.14.3 (GIL enabled). Numbers on server-class x86 will
differ; the *shape* — a hundred-odd nanoseconds per call, order 1% of an inference workload —
should not.

### Per-call cost

Minimum across 7 rounds of 200k calls, net of empty-loop overhead. Minimum rather than mean
because we want the cost of the code, not the cost of whatever else the laptop was doing.

| call | cost |
|---|---|
| `span.mark(kind)` | 83 ns |
| `span.mark(kind, a, b, batch)` | 94 ns |
| `span.decode_step()` — `aggregate` (default) | 100 ns |
| `span.decode_step()` — `coarse` | 71 ns |
| `span.decode_step()` — `full` | 138 ns |
| `span.decode_step()` — worst case, batch size changes every step | 251 ns |
| `span.mark()` with tracing disabled | 25 ns |
| a whole request span (open, 4 marks, close) | 1330 ns |
| `record_batch()` with 8 members | 986 ns |

`record_batch` is ~100 ns per event it emits, same as everything else; it is called once per
scheduler iteration rather than once per token, so it does not appear in the throughput result.

The 29 ns the default mode costs over `coarse` buys stall detection. That is not a luxury: without
it the default mode reports a TPOT p99/p50 of 2.5x on a workload where the true figure is 39x.
See [design.md](design.md#aggregation-has-to-preserve-stalls-or-it-is-worthless).

### Throughput cost

A synthetic engine stub burns a calibrated amount of CPU per step. The calibration is the part
that matters: the stub is sized so the *uninstrumented* pipeline sustains the target QPS at the
target token count — 9,950 ns per step for 500 req/s × 200 tokens. Without that step the stub
would be far cheaper than a real decode step and the ratio would flatter us by an order of
magnitude.

Sampling is **paired**: each sample runs the baseline and the instrumented pipeline back to back,
and the interval is taken over per-pair deltas. Machine drift here is several times larger than
the effect being resolved — an unpaired run of the same benchmark reported anywhere from 0.47% to
1.20% depending on when it ran.

| operating point | hook calls/req | overhead |
|---|---|---|
| 500 req/s × 200 tokens | 205 | **1.53% ± 0.32%** (30 pairs) |
| 500 req/s × 50 tokens | 55 | 0.36% ± 1.02% (25 pairs) |

The second point's interval is wider than its estimate, so it establishes only the trend, not a
precise value. It is reported rather than dropped because dropping inconvenient runs is how
benchmarks start lying.

### End to end, on a real model

`benchmarks/e2e.py` serves a fixed workload through `inferscope_lab` driving
Qwen2.5-0.5B-Instruct on MPS, instrumented against uninstrumented, paired:

| decode mode | uninstrumented | instrumented | overhead |
|---|---|---|---|
| `aggregate` (default) | 136.4 tok/s | 136.6 tok/s | **0.15% ± 0.46%** (9 pairs) |
| `full` | 135.7 tok/s | 133.7 tok/s | 1.20% ± 1.24% (6 pairs) |

Both intervals contain zero. That is the expected result and worth stating plainly rather than
dressing up: at 136 tok/s a decode step costs ~7 ms of real GPU work against ~100 ns of
instrumentation, a ratio of roughly 70,000:1. The predicted overhead is ~0.002%, which no
wall-clock measurement on a laptop will ever resolve.

So this measurement does not establish a number. It establishes that the number is small enough to
be invisible in practice, and it checks that the synthetic benchmark is not an artifact of the
stub — which is the only thing it was ever able to do.

### Sanity check against theory

Per request: one span (1440 ns for open, 4 marks and close) + 200 `decode_step` calls at 107 ns
≈ 22.8 µs, against a 1,953 µs request budget at the measured 512 req/s baseline → **1.17%**
predicted, 1.55% measured. The gap is the flush thread competing for the GIL and the sink
absorbing a larger event stream.

The same model predicts 0.30% for the 50-token point against 0.36% measured. Theory tracking
measurement across a 4x change in token count is the actual result here; a single number that
nothing predicts would not be worth much.

## Honest limits

- **This is not real GPU serving.** No CUDA device was involved — the development machine has
  none. The synthetic stub exists precisely so the model's own cost does not swamp the signal; it
  says what instrumentation costs, not what an end-to-end system does. The end-to-end number on a
  real model via `inferscope_lab`'s `HFRunner` is a separate measurement and will be reported
  separately rather than folded into this one.
- **Overhead scales with token rate, not request rate.** The per-token `decode_step` call
  dominates: 107 ns against a per-request-token step budget of ~10 µs. Engines with cheaper
  per-token steps (small models, huge batches) will see a proportionally larger percentage, and
  `coarse` mode exists for them.
- **Synthetic call rates saturate the flush thread.** A tight loop emits ~10M events/s, ~100×
  any real engine, at which point flush-thread GIL contention pushes per-call cost from ~86 ns to
  ~105 ns. That is a benchmark artifact, not a production regime, which is why the per-call table
  is measured with the flush thread idle and system-level cost is measured separately at a
  realistic rate.
- **Buffer occupancy affects per-call cost mildly** (~77 ns at 2k events buffered, ~90 ns at
  200k) through allocator pressure. Disabling the GC changed nothing, so this is not GC tracking
  of the event tuples, as was first suspected.
