"""Block allocator behaviour."""

from __future__ import annotations

import pytest

from inferscope_lab import BlockAllocator, OutOfBlocks


def test_blocks_for_rounds_up() -> None:
    a = BlockAllocator(num_blocks=8, block_size=16)
    assert a.blocks_for(0) == 0
    assert a.blocks_for(1) == 1
    assert a.blocks_for(16) == 1
    assert a.blocks_for(17) == 2


def test_allocate_grow_free_roundtrip() -> None:
    a = BlockAllocator(num_blocks=8, block_size=16)
    assert a.allocate("s1", 33) == 3
    assert a.blocks_held("s1") == 3
    assert a.capacity_tokens("s1") == 48
    assert a.grow("s1", 48) == 0, "already covered, no new blocks"
    assert a.grow("s1", 49) == 1
    assert a.used_blocks == 4
    assert a.free("s1") == 4
    assert a.used_blocks == 0
    assert "s1" not in a


def test_allocation_failure_leaves_the_pool_untouched() -> None:
    a = BlockAllocator(num_blocks=4, block_size=16)
    with pytest.raises(OutOfBlocks):
        a.allocate("s1", 200)
    assert a.free_blocks == 4
    assert "s1" not in a


def test_grow_failure_leaves_the_pool_untouched() -> None:
    a = BlockAllocator(num_blocks=4, block_size=16)
    a.allocate("s1", 16)
    with pytest.raises(OutOfBlocks):
        a.grow("s1", 200)
    assert a.blocks_held("s1") == 1
    assert a.free_blocks == 3


def test_double_allocate_is_rejected() -> None:
    a = BlockAllocator(num_blocks=8, block_size=16)
    a.allocate("s1", 16)
    with pytest.raises(ValueError):
        a.allocate("s1", 16)


def test_free_is_idempotent_and_blocks_are_reusable() -> None:
    a = BlockAllocator(num_blocks=2, block_size=16)
    a.allocate("s1", 32)
    assert a.free("s1") == 2
    assert a.free("s1") == 0
    a.allocate("s2", 32)  # must not raise: the blocks came back
    assert a.used_blocks == 2


def test_occupancy_tracks_usage() -> None:
    a = BlockAllocator(num_blocks=4, block_size=16)
    assert a.occupancy == 0.0
    a.allocate("s1", 32)
    assert a.occupancy == 0.5
