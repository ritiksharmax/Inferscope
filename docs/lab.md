# inferscope_lab

A continuous-batching inference engine, built to be instrumented.

Real engines are the eventual target, but you cannot iterate on an observability data model
against someone else's scheduler — and the development machine here has no CUDA device, so vLLM
cannot run on it at all. So the validation target is a scheduler small enough to own end to end,
with every hook point deliberately placed and reproducible pathologies to aim the tooling at.

## What is real, and what is not

**Real:** iteration-level scheduling, where one iteration is *either* a prefill batch or a decode
batch. A fixed pool of fixed-size KV blocks, allocated per sequence as it grows. Preemption when
the pool runs dry, taking the newest running request first, and a genuine recompute of its whole
context on resume. With `HFRunner`, real forward passes over real KV tensors on MPS/CUDA/CPU.

**Not real:** attention is not paged. `HFRunner` left-pads a decode batch to its longest context,
exactly as a pre-paged-attention engine would, so short sequences pay for attention over padding.
That waste is *reported* (`BATCH_PADDING`) rather than hidden — it is a real inefficiency of the
design and one of the things a trace should expose. Block accounting is enforced against a real
pool but is bookkeeping alongside those tensors, not the layout of them. Prefills within one
batch run sequentially: prefill cost is dominated by total tokens either way, and concatenated
variable-length prefill is a kernel problem, not a scheduling one.

## Two runners

`FakeRunner` is a cost function — prefill scales with tokens, decode scales with batch — so the
scheduler, the KV accounting and every pathology are testable deterministically in milliseconds
without a GPU or a download. `HFRunner` drives an actual causal LM. Both satisfy the same
`ModelRunner` protocol, and `tests/test_lab_hf_runner.py` checks that the padded batched decode
is numerically equivalent to decoding each sequence alone — without which "runs a real model"
would be a claim about nothing.

## The scenarios

```bash
python -m inferscope_lab.pathologies --inject prefill-starvation --db traces.db
python -m inferscope_lab.pathologies --inject kv-thrash --decode-mode full
python -m inferscope_lab.pathologies --inject healthy --model tiny   # real weights
```

| scenario | what goes wrong | signature in the trace |
|---|---|---|
| `healthy` | nothing | no preemption, batches near max, no stalls |
| `prefill-starvation` | a burst of long prompts monopolises iterations | ~25 ms `DECODE_STALL`s on already-running requests |
| `kv-thrash` | working set exceeds the block pool | repeated `PREEMPTED`/`RESUMED`, high recomputed tokens |
| `batch-starvation` | admission too conservative to fill batches | mean decode batch far below `max_batch_size` |

`tests/test_lab_pathologies.py` asserts each of these, so a diagnostic query that "detects" a
pathology is checked against a scenario that provably has it.

### Why `prefill-starvation` is the interesting one

Its whole point is that **request-level latency looks fine**. p99/p50 over end-to-end duration
stays near 1.1x, because every request is slowed a little. The damage is concentrated in per-token
cadence: TPOT p99/p50 is ~40x, in ~25 ms stalls landing exactly on the long prefills. That gap
between "the dashboard looks normal" and "something is badly wrong" is the entire argument for
instrumenting the inference loop rather than the HTTP handler — so there is a test asserting the
end-to-end view stays unremarkable, not only that the token view spikes.

## Driving it yourself

```python
from inferscope import Tracer
from inferscope_lab import EngineConfig
from inferscope_lab.engine import Engine
from inferscope_lab.runner import FakeRunner

tracer = Tracer("sqlite:///traces.db")
engine = Engine(FakeRunner(), tracer, EngineConfig(num_blocks=512, max_batch_size=16))
engine.add_request("r0", prompt_tokens=512, max_new_tokens=128)
engine.run_until_idle()
tracer.close()
```

Workloads arrive **open-loop** (`run_workload`): arrivals land on schedule regardless of load. A
closed-loop driver that waits for a response before sending the next request would hide exactly
the queue growth being measured.
