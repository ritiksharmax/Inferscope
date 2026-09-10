"""Request state as it moves through the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from inferscope.tracer import RequestSpan


class RequestState(Enum):
    WAITING = "waiting"      # queued, no KV blocks held
    RUNNING = "running"      # admitted, holds KV blocks
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class Request:
    """One generation request.

    ``prompt_tokens`` never changes; ``generated`` grows by one per decode step.
    A preempted request goes back to WAITING with ``generated`` intact and its
    blocks returned to the pool, so resuming it costs a prefill over
    ``prompt_tokens + generated`` -- that recompute is the whole reason KV
    pressure hurts, so it is modelled explicitly rather than waved away.
    """

    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_ns: int
    span: RequestSpan | None = None
    state: RequestState = RequestState.WAITING
    generated: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0
    prefilled: bool = False
    #: Tokens of context already in the KV cache. Equal to 0 or
    #: ``context_tokens`` unless the prefill is being chunked.
    prefilled_tokens: int = 0
    finish_ns: int = 0
    _history: list[str] = field(default_factory=list)

    @property
    def context_tokens(self) -> int:
        """Tokens whose KV must be resident for the next step."""
        return self.prompt_tokens + self.generated

    @property
    def is_done(self) -> bool:
        return self.generated >= self.max_new_tokens

    @property
    def needs_prefill(self) -> bool:
        """True for a fresh request, and for one resuming after preemption."""
        return not self.prefilled

    @property
    def prefill_remaining(self) -> int:
        return max(0, self.context_tokens - self.prefilled_tokens)
