"""Concurrent stress test for RadixCache thread safety."""
from __future__ import annotations
import random
import string
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import pytest
from sluice.radix_cache import RadixCache

def _rand_key(length: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def _rand_value() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=20))

def test_concurrent_stress():
    cache = RadixCache(max_size=512)
    num_threads = 16
    iterations = 1200
    errors = []
    error_lock = threading.Lock()

    def _record_error(exc):
        with error_lock:
            errors.append(exc)

    seen_keys = set()
    for _ in range(200):
        k = _rand_key(10)
        seen_keys.add(k)
        cache.put(k, _rand_value())

    def _worker(thread_id: int):
        for iteration in range(iterations):
            try:
                if random.random() < 0.6:
                    key = _rand_key()
                    with error_lock: seen_keys.add(key)
                    cache.put(key, _rand_value())
                else:
                    with error_lock:
                        if seen_keys:
                            lookup_key = random.choice(list(seen_keys))
                            cache.get(lookup_key)
            except Exception as exc:
                _record_error(exc)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            executor.submit(_worker, i)

    assert not errors
    assert len(cache) <= 512
    assert set(cache.keys()) == set(cache._cache.keys())
