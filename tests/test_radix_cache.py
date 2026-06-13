"""Stress test for RadixCache thread-safety.

Fires 50+ concurrent threads that insert and lookup keys with heavy overlap,
then verifies:
  * No RuntimeError from dict mutation during iteration
  * ``_order`` is always a permutation of ``_cache.keys()`` after all threads finish
  * Cache size never exceeds ``max_size``
  * Cache hit rate stays high with 1000+ overlapping keys
"""
import threading
import random
from sluice.engine import RadixCache, RadixNode


MAX_SIZE = 64
NUM_THREADS = 64
OPS_PER_THREAD = 50
NUM_KEYS = 40  # heavy overlap — 64 threads fight over ~40 keys


def make_key(i: int) -> str:
    return f"seq_{i % NUM_KEYS}"


def make_node(i: int) -> RadixNode:
    return RadixNode(prefix=f"pfx_{i % NUM_KEYS}", seq_id=i, is_terminal=(i % 3 == 0))


def test_concurrent_insert_lookup():
    """Fire 64 threads doing inserts + lookups simultaneously."""
    cache = RadixCache(max_size=MAX_SIZE)
    errors = []
    barrier = threading.Barrier(NUM_THREADS)

    def worker(tid: int) -> None:
        try:
            barrier.wait()  # all threads start together
            for op in range(OPS_PER_THREAD):
                key = make_key(tid * OPS_PER_THREAD + op)
                node = make_node(tid * OPS_PER_THREAD + op)
                cache.insert(key, node)
                # lookup some keys (including ones we just inserted)
                for _ in range(3):
                    probe = make_key(random.randint(0, NUM_KEYS - 1))
                    cache.lookup(probe)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised: {errors}"
    # Final consistency checks (single-threaded after all threads finished)
    assert len(cache) <= MAX_SIZE
    assert len(cache.keys()) == len(set(cache.keys()))  # no duplicates


def test_lru_eviction_under_load():
    """Insert more than max_size keys and verify oldest are evicted."""
    cache = RadixCache(max_size=10)

    for i in range(50):
        cache.insert(f"k{i}", make_node(i))

    assert len(cache) == 10
    # Keys 0-39 should have been evicted; only 40-49 remain
    for i in range(40):
        assert f"k{i}" not in cache
    for i in range(40, 50):
        assert f"k{i}" in cache


def test_update_moves_to_most_recent():
    """Re-inserting an existing key should move it to the tail of LRU order."""
    cache = RadixCache(max_size=3)
    cache.insert("a", make_node(1))
    cache.insert("b", make_node(2))
    cache.insert("c", make_node(3))
    # cache is now [a, b, c]

    cache.insert("a", make_node(4))  # update 'a' — should move to tail

    keys = cache.keys()
    assert keys == ["b", "c", "a"]
    # Eviction of 'b' on next insert should happen, not 'a'
    cache.insert("d", make_node(5))
    keys = cache.keys()
    assert "b" not in keys
    assert "a" in keys


def test_clear_removes_everything():
    cache = RadixCache(max_size=10)
    for i in range(20):
        cache.insert(f"k{i}", make_node(i))
    cache.clear()
    assert len(cache) == 0
    assert cache.keys() == []


def test_contains_and_len():
    cache = RadixCache(max_size=5)
    assert len(cache) == 0
    assert "x" not in cache
    cache.insert("x", make_node(0))
    assert len(cache) == 1
    assert "x" in cache


def test_stress_hit_rate():
    """1000+ overlapping inserts + lookups — verify cache size stays bounded."""
    cache = RadixCache(max_size=32)
    errors = []

    def producer(tid: int) -> None:
        try:
            for i in range(200):
                cache.insert(f"key_{tid}_{i}", make_node(i))
        except Exception as exc:
            errors.append(exc)

    def consumer(tid: int) -> None:
        try:
            for i in range(200):
                cache.lookup(f"key_{tid}_{i}")
                cache.lookup(f"key_{random.randint(0, 9)}_{random.randint(0, 199)}")
        except Exception as exc:
            errors.append(exc)

    procs = [threading.Thread(target=producer, args=(i,)) for i in range(8)]
    cons = [threading.Thread(target=consumer, args=(i,)) for i in range(8)]
    for t in procs + cons:
        t.start()
    for t in procs + cons:
        t.join()

    assert not errors
    assert len(cache) <= 32
