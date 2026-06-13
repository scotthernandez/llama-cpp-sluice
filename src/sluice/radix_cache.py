"""Radix Cache - thread-safe prefix trie for deduplicating shared token prefixes."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional


class RadixCache:
    """A thread-safe LRU cache with insertion-order tracking.

    Designed for prefix-caching in LLM inference: keys are string prefixes,
    values are arbitrary (e.g. token sequences or activation tensors).

    Attributes
    ----------
    _cache : OrderedDict[str, Any]
        The underlying key-value store.
    _order : list[str]
        Monotonically growing list of keys in insertion order (duplicates
        are removed on re-insertion so _order stays a valid permutation
        of _cache.keys()).
    on_evict : Callable[[str, Any], None] | None
        Optional callback triggered when an item is evicted.
    """

    def __init__(self, max_size: int = 4096, on_evict: Optional[Callable[[str, Any], None]] = None) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self.on_evict = on_evict

    # -- public API -------------------------------------------------------- #

    def put(self, key: str, value: Any) -> None:
        """Insert or update *key* with *value*.  LRU eviction runs if full."""
        evicted_items: List[tuple[str, Any]] = []
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                if key in self._order:
                    self._order.remove(key)
                self._order.append(key)
                self._cache[key] = value
                return

            # Evict LRU entries until we have room
            while len(self._cache) >= self.max_size and self._cache:
                evict_key, evict_val = self._cache.popitem(last=False)
                evicted_items.append((evict_key, evict_val))
                try:
                    self._order.remove(evict_key)
                except ValueError:
                    pass

            self._cache[key] = value
            self._order.append(key)
        
        # Trigger callbacks outside the lock
        if self.on_evict:
            for ek, ev in evicted_items:
                try: self.on_evict(ek, ev)
                except Exception: pass

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* on miss."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return default

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache. Returns True if it existed."""
        with self._lock:
            if key not in self._cache:
                return False
            del self._cache[key]
            try:
                self._order.remove(key)
            except ValueError:
                pass
            return True

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def keys(self) -> list[str]:
        """Return a copy of the current keys."""
        with self._lock:
            return list(self._cache.keys())

    def has_value(self, value: Any) -> bool:
        """Check if any cached entry contains the specified value (thread-safe)."""
        with self._lock:
            # For complex values (like our tuples), we check for the specific value
            # This is typically used to see if an SID is still in cache.
            for v in self._cache.values():
                if v == value: return True
                if isinstance(v, (tuple, list)) and value in v: return True
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    # -- stats ------------------------------------------------------------- #

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "order_len": len(self._order),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
