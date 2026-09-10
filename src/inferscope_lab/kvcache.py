"""Block-based KV cache accounting.

Models the thing that actually causes KV pathologies in a real server: a fixed
pool of fixed-size blocks, allocated per sequence as it grows, with nothing left
to hand out when the pool runs dry.

The allocation *bookkeeping* is real -- a sequence that cannot get a block is
genuinely preempted, and genuinely pays to recompute its cache on resume. The
tensors behind the blocks are the model runner's business; see the package
docstring for what that does and does not simulate.
"""

from __future__ import annotations


class OutOfBlocks(Exception):
    """Raised when the pool cannot satisfy an allocation."""


class BlockAllocator:
    """A fixed pool of KV blocks, handed out per sequence.

    Blocks are returned to the pool in LIFO order, which keeps recently freed
    blocks hot and makes the free list's behaviour reproducible across runs.
    """

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._tables: dict[str, list[int]] = {}

    # -- queries ------------------------------------------------------------

    def blocks_for(self, tokens: int) -> int:
        """Blocks required to hold ``tokens`` tokens."""
        if tokens <= 0:
            return 0
        return -(-tokens // self.block_size)  # ceil

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def occupancy(self) -> float:
        return self.used_blocks / self.num_blocks

    def blocks_held(self, seq_id: str) -> int:
        return len(self._tables.get(seq_id, ()))

    def capacity_tokens(self, seq_id: str) -> int:
        """How many tokens the blocks currently held by ``seq_id`` can store."""
        return self.blocks_held(seq_id) * self.block_size

    def can_allocate(self, tokens: int) -> bool:
        return self.blocks_for(tokens) <= len(self._free)

    # -- mutation -----------------------------------------------------------

    def allocate(self, seq_id: str, tokens: int) -> int:
        """Reserve enough blocks for ``seq_id`` to hold ``tokens`` tokens.

        Returns the number of blocks newly taken from the pool. Allocation is
        all-or-nothing: on failure the pool is untouched.
        """
        if seq_id in self._tables:
            raise ValueError(f"{seq_id!r} already has blocks; use grow()")
        needed = self.blocks_for(tokens)
        if needed > len(self._free):
            raise OutOfBlocks(f"need {needed} blocks, {len(self._free)} free")
        table = [self._free.pop() for _ in range(needed)]
        self._tables[seq_id] = table
        return needed

    def grow(self, seq_id: str, total_tokens: int) -> int:
        """Extend ``seq_id`` to hold ``total_tokens``. Returns blocks added."""
        table = self._tables.get(seq_id)
        if table is None:
            raise KeyError(seq_id)
        needed = self.blocks_for(total_tokens) - len(table)
        if needed <= 0:
            return 0
        if needed > len(self._free):
            raise OutOfBlocks(f"need {needed} more blocks, {len(self._free)} free")
        table.extend(self._free.pop() for _ in range(needed))
        return needed

    def free(self, seq_id: str) -> int:
        """Return ``seq_id``'s blocks to the pool. Returns blocks freed."""
        table = self._tables.pop(seq_id, None)
        if not table:
            return 0
        # LIFO: give back in reverse so the next allocation reuses the hottest.
        self._free.extend(reversed(table))
        return len(table)

    def __contains__(self, seq_id: object) -> bool:
        return seq_id in self._tables

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(used={self.used_blocks}/{self.num_blocks}, "
            f"block_size={self.block_size})"
        )
