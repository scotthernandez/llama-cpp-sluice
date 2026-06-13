"""Radix Cache - thread-safe prefix trie for deduplicating shared token prefixes."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional, List, Dict


class RadixNode:
    """Minimal radix tree node for prefix caching."""
    __slots__ = ("prefix", "children", "value", "is_terminal")

    def __init__(self, prefix: str = "", value: Any = None, is_terminal: bool = False) -> None:
        self.prefix = prefix
        self.children: Dict[str, "RadixNode"] = {}
        self.value = value # Typically (sid, tokens)
        self.is_terminal = is_terminal

    def __repr__(self) -> str:
        return f"RadixNode(prefix={self.prefix!r}, value={self.value}, terminal={self.is_terminal})"


class RadixCache:
    """Thread-safe LRU radix cache for KV-prefix deduplication."""

    def __init__(self, max_size: int = 256, on_evict: Optional[Any] = None) -> None:
        self.max_size = max_size
        self._cache: Dict[str, RadixNode] = {}
        self._order: List[str] = []  # LRU order (most-recent at end)
        self._lock = threading.Lock()
        self.on_evict = on_evict

    def put(self, key: str, value: Any) -> None:
        """Insert or update a key-node pair. Evicts LRU entry when over max_size."""
        evicted_items: List[tuple[str, Any]] = []
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            
            # Evict if full
            while len(self._cache) >= self.max_size and self._cache:
                evict_key = self._order.pop(0)
                evicted_items.append((evict_key, self._cache.pop(evict_key).value))

            node = RadixNode(prefix=key, value=value, is_terminal=True)
            self._cache[key] = node
            self._order.append(key)
        
        if self.on_evict:
            for ek, ev in evicted_items:
                try: self.on_evict(ek, ev)
                except Exception: pass

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for *key*, or ``None``."""
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key].value
            return None

    def has_value(self, target_value: Any) -> bool:
        """Thread-safe check if any node contains the specified value (e.g. SID)."""
        with self._lock:
            for node in self._cache.values():
                val = node.value
                if val == target_value: return True
                if isinstance(val, (tuple, list)) and target_value in val: return True
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Remove all entries."""
        evicted_items: List[tuple[str, Any]] = []
        with self._lock:
            for k, node in self._cache.items():
                evicted_items.append((k, node.value))
            self._cache.clear()
            self._order.clear()
        
        if self.on_evict:
            for ek, ev in evicted_items:
                try: self.on_evict(ek, ev)
                except Exception: pass
