"""Interner behaviour, including under concurrent first-sight of the same key."""

from __future__ import annotations

import threading

from inferscope.events import Interner


def test_intern_is_stable_and_dense() -> None:
    i = Interner()
    assert i.intern("a") == 0
    assert i.intern("b") == 1
    assert i.intern("a") == 0
    assert len(i) == 2
    assert i.name(1) == "b"
    assert i.name(99) is None


def test_drain_new_yields_each_mapping_once() -> None:
    i = Interner()
    i.intern("a")
    i.intern("b")
    assert i.drain_new() == [(0, "a"), (1, "b")]
    assert i.drain_new() == []
    i.intern("c")
    assert i.drain_new() == [(2, "c")]


def test_concurrent_intern_of_same_key_agrees() -> None:
    i = Interner()
    barrier = threading.Barrier(8)
    results: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        idx = i.intern("contended")
        with lock:
            results.append(idx)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == 1, "all threads must agree on one index"
    assert len(i) == 1
    assert len(i.drain_new()) == 1, "the mapping must be published exactly once"
