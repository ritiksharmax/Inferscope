"""Engine configuration.

Every knob here exists because turning it produces a *different* latency
pathology -- these are the dials the debugging story turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SchedulingPolicy = Literal["prefill-priority", "decode-priority"]


@dataclass
class EngineConfig:
    """Scheduler and cache sizing."""

    #: KV blocks in the pool. Shrink this to induce preemption and recompute.
    num_blocks: int = 512
    #: Tokens per block.
    block_size: int = 16
    #: Max sequences in one decode batch.
    max_batch_size: int = 16
    #: Max prompt tokens admitted in a single prefill batch.
    max_prefill_tokens: int = 2048
    #: Max sequences admitted in a single prefill batch.
    max_prefill_seqs: int = 4

    #: ``prefill-priority`` admits new work whenever it can, which is what most
    #: engines do and what starves decode when long prompts arrive in a burst.
    #: ``decode-priority`` drains running requests first: better TPOT, worse TTFT.
    policy: SchedulingPolicy = "prefill-priority"

    #: Fraction of the pool that must stay free to admit a new request. A
    #: watermark above 0 trades admission throughput for preemption avoidance.
    admission_watermark: float = 0.0

    #: Split long prefills across iterations and run them *alongside* decode,
    #: instead of letting one long prompt own an entire iteration. This is the
    #: real fix for prefill starvation, and what vLLM and SGLang do.
    chunked_prefill: bool = False
    #: Prefill tokens admitted per iteration when chunking.
    chunk_tokens: int = 256

    def __post_init__(self) -> None:
        if not 0.0 <= self.admission_watermark < 1.0:
            raise ValueError("admission_watermark must be in [0, 1)")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
