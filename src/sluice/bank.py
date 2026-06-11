import asyncio
import time
import subprocess
from typing import Dict, Optional, List

class TokenBank:
    def __init__(self, pool_names: List[str], pool_capacities: Dict[str, int], reserved_for_large: int, large_threshold: int = 16384, 
                 starvation_hook: Optional[str] = None, recovery_hook: Optional[str] = None):
        self.pool_names = pool_names
        self.capacities = pool_capacities
        self.used = {name: 0 for name in pool_names}
        self.reserved_for_large = reserved_for_large
        self.large_threshold = large_threshold
        self.starvation_hook = starvation_hook
        self.recovery_hook = recovery_hook
        
        self.active_seqs: Dict[int, Dict[str, Any]] = {} # sid -> {pool, size}
        self.pinned_seqs: Dict[int, Dict[str, Any]] = {} 
        self.seq_counter = 0
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        self.waiting_large = 0
        self.is_draining = False
        self.is_expanded = False

    async def acquire(self, pool_name: str, requested_size: int, timeout: float = 60.0) -> int:
        is_large = requested_size >= self.large_threshold
        start_time = time.time()
        hook_triggered = False

        async with self.condition:
            if self.is_draining: raise RuntimeError("Server is currently draining.")
            
            # Fast-path check: If even a trimmed version won't fit the reserved floor, 
            # we should signal a backoff immediately.
            
            if is_large: self.waiting_large += 1
            
            try:
                while True:
                    available = self.capacities[pool_name] - self.used[pool_name]
                    
                    if not is_large and (self.waiting_large > 0 or (available - requested_size) < self.reserved_for_large):
                        can_fit = False
                    else:
                        can_fit = requested_size <= available

                    if can_fit:
                        if is_large: self.waiting_large -= 1
                        self.seq_counter += 1
                        sid = self.seq_counter
                        self.used[pool_name] += requested_size
                        self.active_seqs[sid] = {"pool": pool_name, "size": requested_size}
                        return sid

                    elapsed = time.time() - start_time
                    if is_large and self.starvation_hook and elapsed > 15.0 and not hook_triggered:
                        asyncio.create_task(self._run_hook(self.starvation_hook, "Scavenge"))
                        hook_triggered = True

                    if elapsed > timeout:
                        if is_large: self.waiting_large -= 1
                        # Raise a specific error that the server can map to 429
                        raise ResourceWarning(f"Bank saturated: Needed {requested_size} in {pool_name}")
                    
                    try: await asyncio.wait_for(self.condition.wait(), timeout=min(1.0, timeout - elapsed))
                    except asyncio.TimeoutError: continue
            except Exception:
                if is_large and 'can_fit' in locals() and can_fit is False: self.waiting_large -= 1
                raise

    def get_available_for_large(self, pool_name: str) -> int:
        """Returns the absolute maximum tokens available for a Large request in a pool."""
        return self.capacities[pool_name] - self.used[pool_name]

    def get_available_for_small(self, pool_name: str) -> int:
        """Returns the available tokens for Small requests, respecting the reservoir barrier."""
        available = self.capacities[pool_name] - self.used[pool_name]
        return max(0, available - self.reserved_for_large)

    async def release(self, sid: int, pin: bool = False):
        async with self.condition:
            data = self.active_seqs.pop(sid, None)
            if not data: return
            pool, size = data["pool"], data["size"]
            if pin: self.pinned_seqs[sid] = data
            else: self.used[pool] -= size
            self.condition.notify_all()

    async def evict(self, sid: int):
        async with self.condition:
            data = self.pinned_seqs.pop(sid, None)
            if not data: return
            self.used[data["pool"]] -= data["size"]
            self.condition.notify_all()

    async def drain(self):
        async with self.condition:
            self.is_draining = True
            while sum(self.used.values()) > 0: await self.condition.wait()

    async def resume(self):
        async with self.condition:
            self.is_draining = False
            self.condition.notify_all()

    async def update_capacity(self, pool_name: str, new_total: int, expanded: bool):
        async with self.condition:
            self.capacities[pool_name] = new_total
            self.is_expanded = expanded
            self.condition.notify_all()

    def get_stats(self):
        return {
            "used": self.used,
            "total": self.capacities,
            "waiting_large": self.waiting_large,
            "is_expanded": self.is_expanded,
            "is_draining": self.is_draining,
            "pinned_count": len(self.pinned_seqs)
        }

    async def _run_hook(self, hook: str, label: str):
        try:
            proc = await asyncio.create_subprocess_shell(hook, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.communicate()
        except Exception: pass
