"""inferscope_lab -- a small continuous-batching inference engine.

This exists to be *instrumented*. Real engines are the eventual target, but you
cannot iterate on an observability data model against someone else's scheduler,
and the development machine here has no CUDA device to run vLLM on anyway. So:
a scheduler small enough to own end to end, with every hook point deliberately
placed, and reproducible latency pathologies to point the tooling at.

**Limitation, stated up front:** KV block *accounting* is real and enforced, so
preemption and recompute genuinely happen and genuinely cost time. But
attention itself uses the model runner's ordinary contiguous cache -- this is
not a paged-attention kernel. The pathologies are real; the memory layout is
simulated.
"""

from inferscope_lab.config import EngineConfig
from inferscope_lab.kvcache import BlockAllocator, OutOfBlocks
from inferscope_lab.request import Request, RequestState

__all__ = [
    "EngineConfig",
    "BlockAllocator",
    "OutOfBlocks",
    "Request",
    "RequestState",
]
