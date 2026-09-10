"""End-to-end overhead: instrumented vs uninstrumented, on a real model.

``benchmarks/overhead.py`` measures what an instrumentation call costs against
a synthetic engine calibrated to a chosen operating point. This measures the
other thing: what happens to a real serving loop, running real forward passes
through a real model, when you turn tracing on.

The numbers are lower-rate and noisier than the synthetic benchmark -- a laptop
GPU is not a serving fleet -- so this is not a replacement for it. It exists to
check that the synthetic result is not an artifact of the stub.

    python -m benchmarks.e2e --model Qwen/Qwen2.5-0.5B-Instruct --pairs 6
    python -m benchmarks.e2e --model tiny --pairs 4      # random weights, no download
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from typing import Any

from inferscope import Tracer
from inferscope.sinks.null import NullSink
from inferscope_lab.config import EngineConfig
from inferscope_lab.engine import Engine

_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
        9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13}


def _build_runner(model: str, device: str | None) -> Any:
    from inferscope_lab.hf_runner import HFRunner, pick_device, tiny_config

    spec: Any = tiny_config() if model == "tiny" else model
    return HFRunner(spec, device=device or pick_device())


def _run_once(
    runner: Any,
    tracer: Tracer | None,
    *,
    requests: int,
    prompt_tokens: int,
    output_tokens: int,
    config: EngineConfig,
) -> tuple[float, int]:
    """Serve a fixed workload once. Returns (seconds, tokens generated)."""
    engine = Engine(runner, tracer, config)
    for i in range(requests):
        engine.add_request(f"r{i}", prompt_tokens, output_tokens)
    start = time.perf_counter()
    engine.run_until_idle()
    elapsed = time.perf_counter() - start
    return elapsed, sum(r.generated for r in engine.finished)


def benchmark(
    model: str,
    *,
    device: str | None = None,
    requests: int = 12,
    prompt_tokens: int = 256,
    output_tokens: int = 48,
    max_batch_size: int = 8,
    decode_mode: str = "aggregate",
    pairs: int = 6,
) -> dict[str, Any]:
    runner = _build_runner(model, device)
    config = EngineConfig(
        num_blocks=2048, block_size=16, max_batch_size=max_batch_size,
        max_prefill_seqs=2, max_prefill_tokens=4096,
    )
    kwargs = dict(requests=requests, prompt_tokens=prompt_tokens,
                  output_tokens=output_tokens, config=config)

    _run_once(runner, None, **kwargs)  # warm up kernels and allocator

    deltas: list[float] = []
    off_rates: list[float] = []
    on_rates: list[float] = []
    hook_calls = requests * (5 + output_tokens)

    for _ in range(pairs):
        off_s, off_tokens = _run_once(runner, None, **kwargs)
        tracer = Tracer(NullSink(), decode_mode=decode_mode, flush_interval=0.05)
        try:
            on_s, on_tokens = _run_once(runner, tracer, **kwargs)
        finally:
            tracer.close()
        assert off_tokens == on_tokens, "the two arms must do identical work"
        off_rates.append(off_tokens / off_s)
        on_rates.append(on_tokens / on_s)
        deltas.append((off_s and (on_s - off_s) / off_s * 100.0) or 0.0)

    mean = statistics.fmean(deltas)
    half = 0.0
    if len(deltas) > 1:
        sem = statistics.stdev(deltas) / math.sqrt(len(deltas))
        half = _T95.get(len(deltas) - 1, 1.96) * sem

    return {
        "model": model,
        "device": getattr(runner, "device", "?"),
        "decode_mode": decode_mode,
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "hook_calls": hook_calls,
        "tokens_per_s_off": statistics.median(off_rates),
        "tokens_per_s_on": statistics.median(on_rates),
        "overhead_pct": mean,
        "ci95": half,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--device", default="")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=48)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--decode-mode", default="aggregate",
                        choices=("aggregate", "coarse", "full"))
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()

    r = benchmark(
        args.model, device=args.device or None, requests=args.requests,
        prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
        max_batch_size=args.max_batch_size, decode_mode=args.decode_mode,
        pairs=args.pairs,
    )

    print(f"\n  {r['model']} on {r['device']}, decode_mode={r['decode_mode']}")
    print(f"  {r['requests']} requests x {r['prompt_tokens']} prompt / "
          f"{r['output_tokens']} output tokens = {r['hook_calls']} hook calls per run\n")
    print(f"    uninstrumented   {r['tokens_per_s_off']:8.1f} tok/s (median)")
    print(f"    instrumented     {r['tokens_per_s_on']:8.1f} tok/s (median)")
    print(f"    overhead         {r['overhead_pct']:8.2f} %"
          f"  +/- {r['ci95']:.2f} (95% CI, {r['pairs']} paired runs)\n")


if __name__ == "__main__":
    main()
