"""Model runners: the thing that actually costs time.

The engine cares about two properties of a model: how long a prefill of N
tokens takes, and how long a decode step over a batch takes. Separating that
behind a protocol means the scheduler, the KV accounting and every pathology
can be tested deterministically in milliseconds, without a GPU, a download, or
a flaky wall-clock assertion -- while the same scheduler drives a real model
through ``HFRunner``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PrefillItem:
    seq_id: str
    #: Tokens to process in this call.
    tokens: int
    #: Tokens of this sequence already in the cache. Non-zero only when the
    #: prefill is being chunked across iterations.
    already: int = 0


@dataclass(frozen=True)
class DecodeItem:
    seq_id: str
    context_tokens: int


@runtime_checkable
class ModelRunner(Protocol):
    """Whatever turns scheduled work into elapsed time (and, sometimes, tokens)."""

    def prefill(self, items: Sequence[PrefillItem]) -> None: ...

    def decode(self, items: Sequence[DecodeItem]) -> None: ...

    def release(self, seq_id: str) -> None:
        """Drop any per-sequence state. Called on finish and on preemption."""


class FakeRunner:
    """A model-shaped cost function.

    Defaults are in the neighbourhood of a small model on a single accelerator:
    a 512-token prefill costs ~21 ms, a decode step over a batch of 16 costs
    ~9 ms. The point is not fidelity to any particular model -- it is that
    prefill scales with tokens, decode scales with batch, and the two compete
    for the same iterations, which is what makes the pathologies real.
    """

    def __init__(
        self,
        *,
        prefill_base_us: float = 1500.0,
        prefill_us_per_token: float = 38.0,
        decode_base_us: float = 6000.0,
        decode_us_per_seq: float = 200.0,
        decode_us_per_ktoken: float = 120.0,
        speedup: float = 1.0,
    ) -> None:
        self.prefill_base_us = prefill_base_us
        self.prefill_us_per_token = prefill_us_per_token
        self.decode_base_us = decode_base_us
        self.decode_us_per_seq = decode_us_per_seq
        self.decode_us_per_ktoken = decode_us_per_ktoken
        #: Divides every cost. Tests use a large value to keep runs instant.
        self.speedup = speedup
        self.prefill_calls = 0
        self.decode_calls = 0

    def prefill_cost_us(self, total_tokens: int) -> float:
        return (self.prefill_base_us + self.prefill_us_per_token * total_tokens) / self.speedup

    def decode_cost_us(self, batch_size: int, total_context: int) -> float:
        raw = (
            self.decode_base_us
            + self.decode_us_per_seq * batch_size
            + self.decode_us_per_ktoken * total_context / 1000.0
        )
        return raw / self.speedup

    def prefill(self, items: Sequence[PrefillItem]) -> None:
        if not items:
            return
        self.prefill_calls += 1
        _burn_us(self.prefill_cost_us(sum(i.tokens for i in items)))

    def decode(self, items: Sequence[DecodeItem]) -> None:
        if not items:
            return
        self.decode_calls += 1
        _burn_us(self.decode_cost_us(len(items), sum(i.context_tokens for i in items)))

    def release(self, seq_id: str) -> None:
        return None


def _burn_us(microseconds: float) -> None:
    """Consume wall time the way accelerator work does: without holding the GIL.

    ``time.sleep`` rather than a spin loop, because a spin would contend with
    the collector's flush thread for the GIL and make the lab's own timings a
    measurement of Python contention rather than of the scheduler.
    """
    if microseconds > 0:
        time.sleep(microseconds / 1e6)
