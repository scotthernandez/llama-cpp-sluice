import asyncio
import time
import subprocess
import logging
from typing import Dict, Optional, List, Any, Callable

# Configure logger
logger = logging.getLogger("sluice.bank")

class BankSaturated(Exception):
    """Raised when a request cannot be satisfied within the timeout."""
    pass

class TokenBank:
    def __init__(self, capacity: int, reserved_for_large: int, 
                 large_threshold: int = 16384, 
                 starvation_hook: Optional[str] = None, 
                 recovery_hook: Optional[str] = None,
                 scavenge_delay: float = 15.0,
                 max_sequences: int = 16):
        self.capacity = capacity
        self.used = 0
        self.reserved_for_large = reserved_for_large
        self.large_threshold = large_threshold
        self.starvation_hook = starvation_hook
        self.recovery_hook = recovery_hook
        self.scavenge_delay = scavenge_delay
        
        self.active_seqs: Dict[int, int] = {} # sid -> size
        self.pinned_seqs: Dict[int, int] = {} 
        self.available_sids = list(range(max_sequences))
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        self.waiting_large = 0
        self.is_draining = False
        self.is_expanded = False
        self._scavenge_triggered: Dict[int, float] = {}

    async def acquire(self, requested_size: int, timeout: float = 60.0) -> int:
        is_large = requested_size >= self.large_threshold
        start_time = time.time()
        
        # Track if we need to decrement waiting_large in the finally block
        decrement_needed = is_large
        
        if is_large:
            async with self.condition:
                self.waiting_large += 1
        
        try:
            while True:
                trigger_scavenge = False
                async with self.condition:
                    if self.is_draining: raise RuntimeError("Server is currently draining.")
                    
                    available = self.capacity - self.used
                    
                    if not is_large and (self.waiting_large > 0 or (available - requested_size) < self.reserved_for_large):
                        can_fit = False
                    else:
                        can_fit = requested_size <= available and len(self.available_sids) > 0

                    if can_fit:
                        if is_large: 
                            self.waiting_large -= 1
                            decrement_needed = False
                        
                        sid = self.available_sids.pop(0)
                        self.used += requested_size
                        self.active_seqs[sid] = requested_size
                        return sid

                    elapsed = time.time() - start_time
                    # Use a dummy key for hook deduplication when sid isn't known yet
                    hook_key = -1 
                    if is_large and self.starvation_hook and elapsed > self.scavenge_delay and hook_key not in self._scavenge_triggered:
                        self._scavenge_triggered[hook_key] = time.time()
                        trigger_scavenge = True

                    if elapsed > timeout:
                        reason = "Bank saturated" if requested_size > available else "No sequence IDs available"
                        raise BankSaturated(f"{reason}: Needed {requested_size}")
                    
                    try: await asyncio.wait_for(self.condition.wait(), timeout=min(1.0, timeout - elapsed))
                    except asyncio.TimeoutError: pass
                
                # Execute hook outside of condition lock
                if trigger_scavenge:
                    asyncio.create_task(self._run_hook(self.starvation_hook, "Scavenge"))
        finally:
            if decrement_needed:
                async with self.condition:
                    self.waiting_large -= 1
                    self.condition.notify_all()

    def get_available_for_large(self) -> int:
        return self.capacity - self.used

    def get_available_for_small(self) -> int:
        return max(0, self.capacity - self.used - self.reserved_for_large)

    async def release(self, sid: int, pin: bool = False):
        async with self.condition:
            size = self.active_seqs.pop(sid, None)
            if size is None: return
            if pin: self.pinned_seqs[sid] = size
            else: 
                self.used -= size
                self.available_sids.append(sid)
            self.condition.notify_all()

    async def evict(self, sid: int):
        async with self.condition:
            size = self.pinned_seqs.pop(sid, None)
            if size is None: return
            self.used -= size
            self.available_sids.append(sid)
            self.condition.notify_all()

    async def drain(self):
        async with self.condition:
            self.is_draining = True
            while self.used > 0: await self.condition.wait()

    async def resume(self):
        async with self.condition:
            self.is_draining = False
            self.condition.notify_all()

    async def update_capacity(self, new_total: int, expanded: bool):
        async with self.condition:
            self.capacity = new_total
            self.is_expanded = expanded
            self.condition.notify_all()

    def get_stats(self):
        return {
            "used": self.used,
            "total": self.capacity,
            "waiting_large": self.waiting_large,
            "is_expanded": self.is_expanded,
            "is_draining": self.is_draining,
            "pinned_count": len(self.pinned_seqs)
        }

    async def _run_hook(self, hook: str, label: str):
        try:
            logger.info("Executing %s hook: %s", label, hook)
            proc = await asyncio.create_subprocess_shell(hook, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("%s hook failed with code %d: %s", label, proc.returncode, stderr.decode().strip())
        except Exception as e:
            logger.exception("Failed to trigger %s hook: %s", label, e)
