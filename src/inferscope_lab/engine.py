"""A continuous-batching engine, instrumented at every hook point.

One ``step()`` is one scheduler iteration, and an iteration is *either* a
prefill batch or a decode batch -- never both. That is what real engines
without chunked prefill do, and it is the source of the most common latency
pathology in production: a burst of long prompts monopolises iterations and
every already-running request stops emitting tokens.

The instrumentation is the point, so it is worth reading which events go where:

- ``REQUEST_START`` / ``QUEUED`` on arrival, ``SCHEDULED`` on admission --
  the gap between them is queue time.
- ``PREFILL_START`` / ``PREFILL_END`` around prefill, so TTFT decomposes into
  queueing plus prefill rather than being one opaque number.
- ``decode_step(batch_size)`` per token, carrying the batch size at that
  instant -- this is what makes "what was it batched with?" answerable.
- ``BATCH`` / ``BATCH_MEMBER`` per iteration, giving the full composition.
- ``KV_ALLOC`` / ``KV_FREE`` / ``KV_EVICT`` / ``KV_USAGE`` for cache pressure,
  and ``PREEMPTED`` / ``RESUMED`` for what that pressure costs.
"""

from __future__ import annotations

from collections import deque
from time import perf_counter_ns

from inferscope import EventKind as K
from inferscope import Tracer
from inferscope_lab.config import EngineConfig
from inferscope_lab.kvcache import BlockAllocator, OutOfBlocks
from inferscope_lab.request import Request, RequestState
from inferscope_lab.runner import DecodeItem, ModelRunner, PrefillItem


class Engine:
    """Continuous batching over a fixed KV block pool."""

    def __init__(
        self,
        runner: ModelRunner,
        tracer: Tracer | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.runner = runner
        self.config = config or EngineConfig()
        # A disabled tracer makes every hook a no-op, which is exactly what the
        # uninstrumented arm of the end-to-end benchmark needs.
        self.tracer = tracer if tracer is not None else Tracer(enabled=False, autostart=False)
        self.allocator = BlockAllocator(self.config.num_blocks, self.config.block_size)

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        #: Admitted, holding blocks, part-way through a chunked prefill.
        self._partial: list[Request] = []
        self.finished: list[Request] = []
        self.iteration = 0
        self.prefill_iterations = 0
        self.decode_iterations = 0
        self.preemptions = 0

    # -- admission ----------------------------------------------------------

    def add_request(
        self, request_id: str, prompt_tokens: int, max_new_tokens: int
    ) -> Request:
        """Enqueue a request. It holds no KV blocks until it is scheduled."""
        span = self.tracer.trace_request(request_id, prompt_tokens=prompt_tokens)
        span.mark(K.QUEUED)
        req = Request(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            arrival_ns=perf_counter_ns(),
            span=span,
        )
        self.waiting.append(req)
        return req

    # -- the loop -----------------------------------------------------------

    def has_work(self) -> bool:
        return bool(self.waiting or self.running or self._partial)

    def step(self) -> list[Request]:
        """Run one scheduler iteration. Returns requests that finished in it."""
        if not self.has_work():
            return []

        self.iteration += 1
        batch_id = f"iter-{self.iteration}"

        if self.config.chunked_prefill:
            finished = self._run_mixed(batch_id)
        else:
            prefill_batch = self._select_prefill() if self._should_prefill() else []
            if prefill_batch:
                self._run_prefill(batch_id, prefill_batch)
                finished = []
            else:
                decode_batch = self._select_decode()
                finished = self._run_decode(batch_id, decode_batch) if decode_batch else []

        self.tracer.record_kv_usage(self.allocator.used_blocks, self.allocator.num_blocks)
        return finished

    def run_until_idle(self, max_iterations: int = 100_000) -> list[Request]:
        """Drive the loop until nothing is left. Returns everything finished."""
        done: list[Request] = []
        while self.has_work():
            if self.iteration >= max_iterations:
                raise RuntimeError(
                    f"engine did not drain in {max_iterations} iterations; "
                    f"{len(self.waiting)} waiting, {len(self.running)} running, "
                    f"{len(self._partial)} mid-prefill"
                )
            done.extend(self.step())
        return done

    # -- scheduling ---------------------------------------------------------

    def _should_prefill(self) -> bool:
        if not self.waiting:
            return False
        if self.config.policy == "prefill-priority":
            return True
        # decode-priority: only admit new work once the running set has drained,
        # or while there is still room in the decode batch.
        return len(self.running) < self.config.max_batch_size and not self.running

    def _admission_budget(self) -> int:
        """Blocks we are willing to hand out, respecting the watermark."""
        reserve = int(self.config.num_blocks * self.config.admission_watermark)
        return max(0, self.allocator.free_blocks - reserve)

    def _select_prefill(self) -> list[Request]:
        """Take as many waiting requests as tokens, seats and blocks allow."""
        batch: list[Request] = []
        tokens = 0
        budget = self._admission_budget()

        while self.waiting and len(batch) < self.config.max_prefill_seqs:
            req = self.waiting[0]
            ctx = req.context_tokens
            if batch and tokens + ctx > self.config.max_prefill_tokens:
                break
            blocks = self.allocator.blocks_for(ctx)
            if blocks > budget:
                break
            self.waiting.popleft()
            batch.append(req)
            tokens += ctx
            budget -= blocks
        return batch

    def _select_decode(self) -> list[Request]:
        return self.running[: self.config.max_batch_size]

    # -- execution ----------------------------------------------------------

    def _run_prefill(self, batch_id: str, batch: list[Request]) -> None:
        self.prefill_iterations += 1
        batch_idx = self.tracer.intern(batch_id)

        for req in batch:
            ctx = req.context_tokens
            blocks = self.allocator.allocate(req.request_id, ctx)
            assert req.span is not None
            req.span.mark(K.SCHEDULED, batch=batch_idx)
            self.tracer.record_kv_event(K.KV_ALLOC, req.request_id, blocks=blocks, tokens=ctx)
            if req.preemptions:
                # Resuming: everything already generated has to be recomputed.
                req.recomputed_tokens += ctx
                req.span.mark(K.RESUMED, ctx)
            req.span.mark(K.PREFILL_START, batch=batch_idx)

        self.runner.prefill([PrefillItem(r.request_id, r.context_tokens) for r in batch])

        for req in batch:
            assert req.span is not None
            req.span.mark(K.PREFILL_END, req.context_tokens)
            req.prefilled = True
            req.prefilled_tokens = req.context_tokens
            req.state = RequestState.RUNNING
            self.running.append(req)

        self.tracer.record_batch(
            batch_id,
            [r.request_id for r in batch],
            prefill_tokens=sum(r.context_tokens for r in batch),
            decode_tokens=0,
        )

    def _run_decode(self, batch_id: str, batch: list[Request]) -> list[Request]:
        # Decode steps do not carry a batch index: BATCH_MEMBER already ties
        # every request to the iteration it ran in, and the per-token path is
        # too hot to spend an interner lookup on.
        self.decode_iterations += 1
        batch = self._make_room(batch)
        if not batch:
            return []

        self.runner.decode(
            [DecodeItem(r.request_id, r.context_tokens) for r in batch]
        )

        batch_size = len(batch)
        finished: list[Request] = []
        for req in batch:
            req.generated += 1
            assert req.span is not None
            req.span.decode_step(batch_size)
            if req.is_done:
                finished.append(req)

        self.tracer.record_batch(
            batch_id,
            [r.request_id for r in batch],
            prefill_tokens=0,
            decode_tokens=batch_size,
        )
        for req in finished:
            self._finish(req)
        return finished

    def _run_mixed(self, batch_id: str) -> list[Request]:
        """One iteration carrying a prefill chunk *and* the decode batch.

        This is what stops a long prompt from owning an iteration: it gets
        ``chunk_tokens`` of prefill per pass while everything already running
        keeps producing tokens. Prefill and decode are issued as two runner
        calls rather than one fused forward pass -- the token counts and
        therefore the costs are the same, but a real engine would batch them
        together.
        """
        batch_idx = self.tracer.intern(batch_id)
        chunks = self._select_chunks(batch_idx)
        decode_batch = self._make_room(self._select_decode())

        if not chunks and not decode_batch:
            # Nothing can proceed: admit whatever is waiting so the loop moves.
            prefill_batch = self._select_prefill()
            if prefill_batch:
                self._admit(prefill_batch, batch_idx)
                chunks = self._select_chunks(batch_idx)
            if not chunks:
                return []

        if chunks:
            self.prefill_iterations += 1
            self.runner.prefill(
                [PrefillItem(r.request_id, n, r.prefilled_tokens) for r, n in chunks]
            )
            for req, n in chunks:
                assert req.span is not None
                req.prefilled_tokens += n
                req.span.mark(K.PREFILL_END, n)
                if req.prefill_remaining == 0:
                    req.prefilled = True
                    req.state = RequestState.RUNNING
                    if req in self._partial:
                        self._partial.remove(req)
                    self.running.append(req)

        finished: list[Request] = []
        if decode_batch:
            self.decode_iterations += 1
            self.runner.decode(
                [DecodeItem(r.request_id, r.context_tokens) for r in decode_batch]
            )
            size = len(decode_batch)
            for req in decode_batch:
                req.generated += 1
                assert req.span is not None
                req.span.decode_step(size)
                if req.is_done:
                    finished.append(req)

        self.tracer.record_batch(
            batch_id,
            [r.request_id for r, _ in chunks] + [r.request_id for r in decode_batch],
            prefill_tokens=sum(n for _, n in chunks),
            decode_tokens=len(decode_batch),
        )
        for req in finished:
            self._finish(req)
        return finished

    def _select_chunks(self, batch_idx: int) -> list[tuple[Request, int]]:
        """Take up to ``chunk_tokens`` of prefill work, admitting as needed."""
        budget = self.config.chunk_tokens
        chunks: list[tuple[Request, int]] = []

        # Finish what is already started before taking on more. Admitting
        # greedily filled the pool with half-prefilled requests and left the
        # running set unable to grow.
        for req in list(self._partial):
            if budget <= 0:
                break
            take = min(budget, req.prefill_remaining)
            if take > 0:
                chunks.append((req, take))
                budget -= take

        while budget > 0 and self.waiting and len(self._partial) < self.config.max_prefill_seqs:
            req = self.waiting[0]
            if not self._admit([req], batch_idx):
                break
            take = min(budget, req.prefill_remaining)
            if take <= 0:
                break
            chunks.append((req, take))
            budget -= take

        for req, _n in chunks:
            assert req.span is not None
            req.span.mark(K.PREFILL_START, batch=batch_idx)
        return chunks

    def _admit(self, batch: list[Request], batch_idx: int) -> bool:
        """Reserve blocks and mark admission. Returns False if the pool refuses."""
        admitted = False
        for req in batch:
            ctx = req.context_tokens
            if self.allocator.blocks_for(ctx) > self._admission_budget():
                break
            self.allocator.allocate(req.request_id, ctx)
            if req in self.waiting:
                self.waiting.remove(req)
            self._partial.append(req)
            assert req.span is not None
            req.span.mark(K.SCHEDULED, batch=batch_idx)
            self.tracer.record_kv_event(
                K.KV_ALLOC, req.request_id,
                blocks=self.allocator.blocks_held(req.request_id), tokens=ctx,
            )
            if req.preemptions:
                req.recomputed_tokens += ctx
                req.span.mark(K.RESUMED, ctx)
            admitted = True
        return admitted

    def _make_room(self, batch: list[Request]) -> list[Request]:
        """Grow each sequence by one token, preempting when the pool runs dry.

        Preemption takes the *newest* running request, so requests that are
        nearly done keep their progress -- the same choice vLLM makes, and the
        reason preemption shows up as a tail-latency problem for recent arrivals
        rather than a uniform slowdown.

        A victim may be a request earlier in this very batch, including one we
        have already grown; the final filter on ``RUNNING`` drops anything that
        got preempted mid-iteration so we never decode a sequence whose blocks
        have been handed back.
        """
        admitted: list[Request] = []
        for req in batch:
            if req.state is not RequestState.RUNNING:
                continue  # preempted earlier in this same iteration
            while True:
                try:
                    added = self.allocator.grow(req.request_id, req.context_tokens + 1)
                except OutOfBlocks:
                    victim = self._pick_victim(exclude=req)
                    if victim is None:
                        # Nothing left to reclaim: this request cannot proceed
                        # this iteration, but others in the batch still can.
                        break
                    self._preempt(victim)
                    continue
                if added:
                    self.tracer.record_kv_event(
                        K.KV_ALLOC, req.request_id, blocks=added,
                        tokens=req.context_tokens + 1,
                    )
                admitted.append(req)
                break
        return [r for r in admitted if r.state is RequestState.RUNNING]

    def _pick_victim(self, exclude: Request) -> Request | None:
        """Newest reclaimable request, mid-prefill ones included.

        A request part-way through a chunked prefill holds blocks for its whole
        context while appearing in neither ``running`` nor ``waiting``. Leaving
        those out of victim selection let the pool deadlock: nothing could grow,
        and nothing was eligible to be reclaimed.
        """
        for req in reversed(self._partial):
            if req is not exclude:
                return req
        for req in reversed(self.running):
            if req is not exclude:
                return req
        return None

    def _preempt(self, req: Request) -> None:
        """Evict a running request back to the queue, discarding its KV."""
        blocks = self.allocator.free(req.request_id)
        self.runner.release(req.request_id)
        if req in self.running:
            self.running.remove(req)
        elif req in self._partial:
            self._partial.remove(req)
        req.state = RequestState.WAITING
        req.prefilled = False
        req.prefilled_tokens = 0
        req.preemptions += 1
        self.preemptions += 1

        assert req.span is not None
        req.span.preempted(blocks, req.context_tokens)
        self.tracer.record_kv_event(
            K.KV_EVICT, req.request_id, blocks=blocks, tokens=req.context_tokens
        )
        # Front of the queue: a preempted request has already waited once, and
        # sending it to the back would starve it outright.
        self.waiting.appendleft(req)

    def _finish(self, req: Request) -> None:
        blocks = self.allocator.free(req.request_id)
        self.runner.release(req.request_id)
        self.running.remove(req)
        req.state = RequestState.FINISHED
        req.finish_ns = perf_counter_ns()
        self.tracer.record_kv_event(K.KV_FREE, req.request_id, blocks=blocks)
        assert req.span is not None
        req.span.close()
        self.finished.append(req)

    # -- introspection ------------------------------------------------------

    def stats(self) -> dict[str, float | int]:
        return {
            "iterations": self.iteration,
            "prefill_iterations": self.prefill_iterations,
            "decode_iterations": self.decode_iterations,
            "preemptions": self.preemptions,
            "finished": len(self.finished),
            "waiting": len(self.waiting),
            "running": len(self.running),
            "kv_occupancy": self.allocator.occupancy,
            "recomputed_tokens": sum(r.recomputed_tokens for r in self.finished),
        }
