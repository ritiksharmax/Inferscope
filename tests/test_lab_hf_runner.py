"""The real-model runner, exercised on a randomly initialised tiny model.

No download, no GPU: these run anywhere torch does. The point is that the
padded batched decode is numerically equivalent to decoding each sequence on
its own -- without that, "runs a real model" would be a claim about nothing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from inferscope import MemorySink, Tracer  # noqa: E402
from inferscope_lab.config import EngineConfig  # noqa: E402
from inferscope_lab.engine import Engine  # noqa: E402
from inferscope_lab.hf_runner import HFRunner, tiny_config  # noqa: E402
from inferscope_lab.runner import DecodeItem, PrefillItem  # noqa: E402


class FixedIdRunner(HFRunner):
    """Feeds deterministic ids so batched and unbatched runs are comparable."""

    def _next_input_ids(self, rows: int, length: int = 1) -> torch.Tensor:
        return torch.full((rows, length), 3, dtype=torch.long, device=self.device)


@pytest.fixture(scope="module")
def model_config():
    return tiny_config()


def test_padded_batch_decode_matches_unbatched_decode(model_config) -> None:
    """Left-padding plus the attention mask must not change what a sequence sees.

    Two sequences of different length are decoded together, so the shorter one
    is padded. Its resulting KV state must match decoding it alone.
    """
    lengths = {"short": 8, "long": 24}

    solo = FixedIdRunner(model_config, device="cpu", dtype=torch.float32)
    solo.prefill([PrefillItem("short", lengths["short"])])
    solo.decode([DecodeItem("short", lengths["short"])])
    solo_keys = solo._cache["short"][0][0]

    batched = FixedIdRunner(model_config, device="cpu", dtype=torch.float32)
    batched.model.load_state_dict(solo.model.state_dict())
    batched.prefill([
        PrefillItem("short", lengths["short"]),
        PrefillItem("long", lengths["long"]),
    ])
    batched.decode([
        DecodeItem("short", lengths["short"]),
        DecodeItem("long", lengths["long"]),
    ])
    batched_keys = batched._cache["short"][0][0]

    assert solo_keys.shape == batched_keys.shape
    assert torch.allclose(solo_keys, batched_keys, atol=1e-4), (
        "padding changed the shorter sequence's KV state"
    )


def test_cache_grows_by_exactly_one_token_per_decode(model_config) -> None:
    runner = FixedIdRunner(model_config, device="cpu", dtype=torch.float32)
    runner.prefill([PrefillItem("s", 12)])
    assert runner._length["s"] == 12
    for expected in (13, 14, 15):
        runner.decode([DecodeItem("s", runner._length["s"])])
        assert runner._length["s"] == expected
        assert runner._cache["s"][0][0].shape[2] == expected


def test_padding_waste_is_counted(model_config) -> None:
    runner = FixedIdRunner(model_config, device="cpu", dtype=torch.float32)
    runner.prefill([PrefillItem("a", 4), PrefillItem("b", 20)])
    runner.decode([DecodeItem("a", 4), DecodeItem("b", 20)])
    assert runner.padding_tokens == 16, "the short sequence attended over 16 pad slots"


def test_release_frees_the_cache(model_config) -> None:
    runner = FixedIdRunner(model_config, device="cpu", dtype=torch.float32)
    runner.prefill([PrefillItem("s", 8)])
    runner.release("s")
    assert "s" not in runner._cache
    runner.decode([DecodeItem("s", 8)])  # must be a no-op, not a crash


def test_engine_drives_a_real_model_end_to_end(model_config) -> None:
    runner = HFRunner(model_config, device="cpu", dtype=torch.float32)
    tracer = Tracer(MemorySink(), autostart=False)
    try:
        engine = Engine(runner, tracer, EngineConfig(
            num_blocks=256, block_size=16, max_batch_size=4, max_prefill_seqs=2
        ))
        for i in range(5):
            engine.add_request(f"r{i}", prompt_tokens=16 + 8 * i, max_new_tokens=4)
        done = engine.run_until_idle()

        assert len(done) == 5
        assert all(r.generated == 4 for r in done)
        assert engine.allocator.used_blocks == 0
        assert runner._cache == {}, "every sequence's KV must be released"
    finally:
        tracer.close()


def test_preemption_releases_real_kv_tensors(model_config) -> None:
    runner = HFRunner(model_config, device="cpu", dtype=torch.float32)
    tracer = Tracer(MemorySink(), autostart=False)
    try:
        # 12 blocks x 16 = 192 tokens, against a working set of 4 x (48 + 16) = 256.
        engine = Engine(runner, tracer, EngineConfig(
            num_blocks=12, block_size=16, max_batch_size=8, max_prefill_seqs=2
        ))
        for i in range(4):
            engine.add_request(f"r{i}", prompt_tokens=48, max_new_tokens=16)
        done = engine.run_until_idle()
        assert len(done) == 4
        assert engine.preemptions > 0
        assert runner._cache == {}
    finally:
        tracer.close()
