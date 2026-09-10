"""A model runner backed by a real Hugging Face causal LM.

Imports torch lazily, so the scheduler, the KV accounting and every pathology
test stay runnable without it.

What is real here: the forward passes, the KV tensors, the attention over
padded context, and therefore the cost curves the engine schedules against.

What is not: sequences in a decode batch are left-padded to the batch's longest
context, exactly as a pre-paged-attention engine would, so short sequences pay
for attention over padding. That padding waste is *reported* (``BATCH_PADDING``)
rather than hidden -- it is a real inefficiency of this design, and one of the
things the trace is supposed to make visible. Prefills within one batch run
sequentially: prefill cost is dominated by total tokens either way, and
concatenated variable-length prefill is a kernel problem, not a scheduling one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from inferscope_lab.runner import DecodeItem, PrefillItem

if TYPE_CHECKING:
    import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def tiny_config(vocab_size: int = 256) -> Any:
    """A randomly initialised 2-layer model, for tests that need no download."""
    from transformers import AutoConfig

    return AutoConfig.for_model(
        "qwen2",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=vocab_size,
        max_position_embeddings=4096,
    )


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class HFRunner:
    """Drives a real causal LM, keeping one KV cache per sequence."""

    def __init__(
        self,
        model: str | Any = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: Any = None,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self.device = device or pick_device()
        if dtype is None:
            dtype = torch.float16 if self.device != "cpu" else torch.float32

        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
            self.config = AutoConfig.from_pretrained(model)
        else:
            self.model = AutoModelForCausalLM.from_config(model).to(dtype)
            self.config = model

        # transformers' shipped stub for .to() is wrapped in a way that rejects
        # a device string; the call is correct at runtime.
        self.model = self.model.to(self.device).eval()  # type: ignore[arg-type]
        self.vocab_size = int(self.config.vocab_size)

        # seq_id -> per-layer (keys, values), each [1, kv_heads, len, head_dim]
        self._cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._length: dict[str, int] = {}
        self.padding_tokens = 0

    # -- helpers ------------------------------------------------------------

    def _next_input_ids(self, rows: int, length: int = 1) -> torch.Tensor:
        """Token ids to feed next.

        The lab generates a fixed number of tokens rather than sampling to a
        stop condition, so content is irrelevant to cost and these are random.
        It is a method so tests can substitute fixed ids and check the padded
        batched decode against an unbatched reference.
        """
        import torch

        return torch.randint(0, self.vocab_size, (rows, length), device=self.device)

    def _to_cache(self, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> Any:
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, (k, v) in enumerate(layers):
            cache.update(k, v, i)
        return cache

    # -- ModelRunner --------------------------------------------------------

    def prefill(self, items: Sequence[PrefillItem]) -> None:
        """Process ``item.tokens`` of prompt, continuing from ``item.already``.

        A chunked prefill arrives here as several calls for one sequence, each
        appending to the cache the previous chunk left behind -- which is why
        chunking costs the same total work as one long prefill, just spread
        across iterations.
        """
        import torch

        for item in items:
            ids = self._next_input_ids(1, item.tokens)
            past = None
            if item.already and item.seq_id in self._cache:
                past = self._to_cache(self._cache[item.seq_id])
            kwargs: dict[str, Any] = {"input_ids": ids, "use_cache": True}
            if past is not None:
                total = item.already + item.tokens
                kwargs["past_key_values"] = past
                kwargs["position_ids"] = torch.arange(
                    item.already, total, device=self.device
                ).unsqueeze(0)
                kwargs["attention_mask"] = torch.ones(
                    (1, total), dtype=torch.long, device=self.device
                )
            with torch.no_grad():
                out = self.model(**kwargs)
            self._cache[item.seq_id] = [
                (layer.keys, layer.values) for layer in out.past_key_values.layers
            ]
            self._length[item.seq_id] = item.already + item.tokens

    def decode(self, items: Sequence[DecodeItem]) -> None:
        import torch

        live = [i for i in items if i.seq_id in self._cache]
        if not live:
            return

        lengths = [self._length[i.seq_id] for i in live]
        max_len = max(lengths)
        self.padding_tokens += sum(max_len - n for n in lengths)
        n_layers = len(self._cache[live[0].seq_id])

        batched: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(n_layers):
            keys, values = [], []
            for item, length in zip(live, lengths, strict=True):
                k, v = self._cache[item.seq_id][layer]
                pad = max_len - length
                if pad:
                    # Left-pad: the real context stays flush against the new
                    # token, so causal attention over it remains correct.
                    k = torch.nn.functional.pad(k, (0, 0, pad, 0))
                    v = torch.nn.functional.pad(v, (0, 0, pad, 0))
                keys.append(k)
                values.append(v)
            batched.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        batch = len(live)
        input_ids = self._next_input_ids(batch)
        attention_mask = torch.zeros(
            (batch, max_len + 1), dtype=torch.long, device=self.device
        )
        for row, length in enumerate(lengths):
            attention_mask[row, max_len - length:] = 1
        position_ids = torch.tensor(
            [[n] for n in lengths], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self._to_cache(batched),
                use_cache=True,
            )

        updated = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
        for row, (item, length) in enumerate(zip(live, lengths, strict=True)):
            new_len = length + 1
            self._cache[item.seq_id] = [
                (k[row : row + 1, :, -new_len:, :], v[row : row + 1, :, -new_len:, :])
                for k, v in updated
            ]
            self._length[item.seq_id] = new_len

    def release(self, seq_id: str) -> None:
        self._cache.pop(seq_id, None)
        self._length.pop(seq_id, None)
