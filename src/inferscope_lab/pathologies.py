"""Reproducible latency pathologies.

Each scenario here is a workload plus an engine configuration chosen so that a
specific, nameable thing goes wrong. They exist so the diagnostic queries can be
tested against a *known* answer: if ``explain_latency`` cannot pin
``prefill-starvation`` on prefill, the query is wrong, not the trace.

Run one::

    python -m inferscope_lab.pathologies --inject prefill-starvation --db traces.db
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, perf_counter_ns

from inferscope import Tracer
from inferscope_lab.config import EngineConfig
from inferscope_lab.engine import Engine
from inferscope_lab.runner import FakeRunner, ModelRunner


@dataclass(frozen=True)
class Arrival:
    at_s: float
    request_id: str
    prompt_tokens: int
    max_new_tokens: int


@dataclass
class Workload:
    name: str
    description: str
    config: EngineConfig
    arrivals: list[Arrival]
    speedup: float = 20.0
    #: What the trace should show. Asserted by the tests, printed by the CLI.
    expected_signature: str = ""
    #: Swap in ``HFRunner`` to replay the same scenario against a real model.
    runner_factory: Callable[[], ModelRunner] | None = field(default=None, repr=False)

    def make_runner(self) -> ModelRunner:
        if self.runner_factory is not None:
            return self.runner_factory()
        return FakeRunner(speedup=self.speedup)


def _steady(n: int, rate_per_s: float, prompt: int, out: int,
            prefix: str = "r", start: float = 0.0) -> list[Arrival]:
    return [
        Arrival(start + i / rate_per_s, f"{prefix}{i}", prompt, out)
        for i in range(n)
    ]


def healthy(seed: int = 0) -> Workload:
    """The control. Nothing is oversubscribed; latency should be boring."""
    rng = random.Random(seed)
    arrivals = [
        Arrival(i * 0.02, f"r{i}", rng.randint(48, 96), rng.randint(16, 32))
        for i in range(24)
    ]
    return Workload(
        name="healthy",
        description="Moderate load, ample KV, short prompts.",
        config=EngineConfig(num_blocks=512, block_size=16, max_batch_size=16),
        arrivals=arrivals,
        expected_signature="no preemption, batches near max, TTFT dominated by prefill",
    )


def prefill_starvation(seed: int = 0) -> Workload:
    """A burst of very long prompts arrives mid-stream.

    Under prefill-priority scheduling every iteration goes to admitting the
    burst, so the short requests already decoding stop emitting tokens. Their
    TPOT collapses even though nothing is wrong with them -- the classic
    "why did p99 spike when the p50 looks fine" incident.
    """
    arrivals = _steady(24, rate_per_s=40, prompt=64, out=48)
    arrivals += [
        Arrival(0.30 + i * 0.005, f"long{i}", 1536, 8) for i in range(6)
    ]
    arrivals.sort(key=lambda a: a.at_s)
    return Workload(
        name="prefill-starvation",
        description="Short decoding requests starved by a burst of long prompts.",
        config=EngineConfig(
            num_blocks=1024, block_size=16, max_batch_size=16,
            max_prefill_seqs=2, max_prefill_tokens=4096,
            policy="prefill-priority",
        ),
        arrivals=arrivals,
        expected_signature="long stalls between decode runs; TPOT tail >> TPOT median",
    )


def kv_thrash(seed: int = 0) -> Workload:
    """More concurrent context than the block pool can hold.

    Requests get admitted, run for a while, then get evicted to make room for
    each other, and pay a full recompute on resume. Throughput drops while the
    engine looks busy -- goodput and throughput diverge.
    """
    arrivals = _steady(16, rate_per_s=60, prompt=192, out=96)
    return Workload(
        name="kv-thrash",
        description="Working set exceeds the KV pool; requests thrash.",
        config=EngineConfig(
            num_blocks=96, block_size=16, max_batch_size=16, max_prefill_seqs=2
        ),
        arrivals=arrivals,
        expected_signature="repeated PREEMPTED/RESUMED, high recomputed_tokens",
    )


def batch_starvation(seed: int = 0) -> Workload:
    """Admission trickles, so batches never fill.

    A conservative watermark plus one-at-a-time admission means the engine is
    never idle and never efficient: queue time dominates while batch sizes stay
    far below the configured maximum. The fix is a scheduler change, and the
    trace has to make that obvious rather than looking like "the model is slow".
    """
    arrivals = _steady(28, rate_per_s=80, prompt=96, out=32)
    return Workload(
        name="batch-starvation",
        description="Overly conservative admission keeps batches tiny.",
        config=EngineConfig(
            num_blocks=256, block_size=16, max_batch_size=16,
            max_prefill_seqs=1, admission_watermark=0.75,
            policy="decode-priority",
        ),
        arrivals=arrivals,
        expected_signature="mean batch size far below max_batch_size; queue time dominates",
    )


SCENARIOS = {
    "healthy": healthy,
    "prefill-starvation": prefill_starvation,
    "kv-thrash": kv_thrash,
    "batch-starvation": batch_starvation,
}


def run_workload(
    workload: Workload, tracer: Tracer | None = None, max_iterations: int = 200_000
) -> Engine:
    """Drive a workload open-loop: arrivals land on schedule regardless of load.

    Open-loop matters. A closed-loop driver that waits for a response before
    sending the next request hides exactly the queue growth we are trying to
    observe.
    """
    engine = Engine(workload.make_runner(), tracer, workload.config)
    pending = sorted(workload.arrivals, key=lambda a: a.at_s)
    i = 0
    t0 = perf_counter()
    scale = 1.0 / workload.speedup

    while i < len(pending) or engine.has_work():
        if engine.iteration >= max_iterations:
            # An engine that cannot make progress must say so. Without this the
            # driver spins forever on a scheduler deadlock, which is a far worse
            # failure than a loud one.
            raise RuntimeError(
                f"{workload.name} did not drain in {max_iterations} iterations: "
                f"{len(engine.waiting)} waiting, {len(engine.running)} running, "
                f"{len(engine._partial)} mid-prefill, "
                f"{engine.allocator.free_blocks}/{engine.allocator.num_blocks} blocks free"
            )
        now = (perf_counter() - t0) / scale
        while i < len(pending) and pending[i].at_s <= now:
            a = pending[i]
            engine.add_request(a.request_id, a.prompt_tokens, a.max_new_tokens)
            i += 1
        if engine.has_work():
            engine.step()
        elif i < len(pending):
            # Idle: jump the clock to the next arrival rather than spinning.
            gap = (pending[i].at_s - now) * scale
            if gap > 0:
                import time as _time
                _time.sleep(min(gap, 0.05))
    return engine


def summarize(engine: Engine, workload: Workload) -> str:
    stats = engine.stats()
    lines = [
        f"  scenario           {workload.name}",
        f"  {workload.description}",
        "",
        f"  iterations         {stats['iterations']}"
        f"  (prefill {stats['prefill_iterations']}, decode {stats['decode_iterations']})",
        f"  finished           {stats['finished']}",
        f"  preemptions        {stats['preemptions']}",
        f"  recomputed tokens  {stats['recomputed_tokens']}",
    ]
    if engine.finished:
        latencies = sorted(
            (r.finish_ns - r.arrival_ns) / 1e6 for r in engine.finished
        )
        p50 = latencies[len(latencies) // 2]
        p99 = latencies[max(0, int(len(latencies) * 0.99) - 1)]
        lines += [
            f"  latency p50        {p50:.1f} ms",
            f"  latency p99        {p99:.1f} ms",
            f"  p99/p50            {p99 / p50:.1f}x",
        ]
    lines += ["", f"  expected signature: {workload.expected_signature}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inject", choices=sorted(SCENARIOS), default="healthy")
    parser.add_argument("--db", default="", help="write the trace to this SQLite file")
    parser.add_argument(
        "--append", action="store_true",
        help="add to an existing --db instead of replacing it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decode-mode", choices=("aggregate", "coarse", "full"), default="aggregate"
    )
    parser.add_argument(
        "--model", default="",
        help="run against a real HF model instead of the cost model "
             "(e.g. Qwen/Qwen2.5-0.5B-Instruct, or 'tiny' for random weights)",
    )
    args = parser.parse_args()

    workload = SCENARIOS[args.inject](args.seed)
    if args.model:
        from inferscope_lab.hf_runner import HFRunner, tiny_config

        spec: object = tiny_config() if args.model == "tiny" else args.model
        workload.runner_factory = lambda: HFRunner(spec)
        # Real forward passes set the pace now, so stop compressing the clock.
        workload.speedup = 1.0
    if args.db and not args.append:
        # Sinks append, which is right for a library and confusing for a demo:
        # a re-run would otherwise silently blend two scenarios into one trace.
        for suffix in ("", "-wal", "-shm"):
            stale = Path(args.db + suffix)
            if stale.exists():
                stale.unlink()
    tracer = Tracer(args.db or "memory", decode_mode=args.decode_mode)
    try:
        started = perf_counter_ns()
        engine = run_workload(workload, tracer)
        wall_ms = (perf_counter_ns() - started) / 1e6
    finally:
        tracer.close()

    print()
    print(summarize(engine, workload))
    print(f"\n  wall time          {wall_ms:.0f} ms")
    if args.model:
        print(f"  model              {args.model}")
    if args.db:
        print(f"  trace              {args.db}")
    print()


if __name__ == "__main__":
    main()
