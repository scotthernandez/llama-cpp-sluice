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
    """Token-budget admission controller for LLM sequence slots.

    Manages a fixed-capacity pool of token slots across a limited number of
    sequence IDs.  Requests are classified as *large* (>= ``large_threshold``)
    or *small*.  Small requests are blocked while large requests are waiting,
    and a configurable *reserved_for_large* headroom is maintained to prevent
    small-request saturation.

    Reference Counting
    ------------------
    Each sequence ID (SID) has a refcount.  When acquired, refcount=1.  Adding to
    cache calls inc_ref().  Releasing a request or evicting from cache calls 
    dec_ref().  SID and tokens are only reclaimed when refcount reaches 0.
    """

    def __init__(self, capacity: int, reserved_for_large: int, 
                 large_threshold: int = 16384, 
                 starvation_hook: Optional[str] = None, 
                 scavenge_delay: float = 15.0,
                 max_sequences: int = 128,
                 shutdown_event: Optional[threading.Event] = None):
        self.capacity = capacity
        self.used = 0
        self.reserved_for_large = reserved_for_large
        self.large_threshold = large_threshold
        self.starvation_hook = starvation_hook
        self.scavenge_delay = scavenge_delay
        self.shutdown_event = shutdown_event
        
        self.active_sizes: Dict[int, int] = {} # sid -> size
        self.refcounts: Dict[int, int] = {}    # sid -> count
        self.available_sids = list(range(max_sequences))
        
        # Lazy sync primitives
        self._lock: Optional[asyncio.Lock] = None
        self._condition: Optional[asyncio.Condition] = None
        
        self.waiting_large = 0
        self.is_draining = False
        self.is_expanded = False
        self._scavenge_triggered: Dict[int, float] = {}

    def _ensure_sync(self):
        if self._condition is None:
            self._lock = asyncio.Lock()
            self._condition = asyncio.Condition(self._lock)
        return self._condition

    async def acquire(self, requested_size: int, timeout: float = 60.0) -> int:
        cond = self._ensure_sync()
        is_large = requested_size >= self.large_threshold
        start_time = time.time()
        
        decrement_needed = is_large
        if is_large:
            async with cond:
                self.waiting_large += 1
        
        try:
            while True:
                if self.shutdown_event and self.shutdown_event.is_set():
                    raise RuntimeError("Server is shutting down.")

                trigger_scavenge = False
                async with cond:
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
                        self.active_sizes[sid] = requested_size
                        self.refcounts[sid] = 1
                        return sid

                    elapsed = time.time() - start_time
                    if is_large and self.starvation_hook and elapsed > self.scavenge_delay:
                        hook_key = -1 
                        if hook_key not in self._scavenge_triggered:
                            self._scavenge_triggered[hook_key] = time.time()
                            trigger_scavenge = True

                    if elapsed > timeout:
                        reason = "Bank saturated" if requested_size > available else "No sequence IDs available"
                        raise BankSaturated(f"{reason}: Needed {requested_size}")
                    
                    try: await asyncio.wait_for(cond.wait(), timeout=min(1.0, timeout - elapsed))
                    except asyncio.TimeoutError: pass
                
                if trigger_scavenge:
                    asyncio.create_task(self._run_hook(self.starvation_hook, "Scavenge"))
        finally:
            if decrement_needed:
                async with cond:
                    self.waiting_large -= 1
                    cond.notify_all()

    async def inc_ref(self, sid: int):
        cond = self._ensure_sync()
        async with cond:
            if sid in self.refcounts:
                self.refcounts[sid] += 1

    async def dec_ref(self, sid: int, on_free: Optional[Callable[[int], Any]] = None):
        cond = self._ensure_sync()
        async with cond:
            if sid not in self.refcounts: return
            self.refcounts[sid] -= 1
            if self.refcounts[sid] <= 0:
                if on_free:
                    try:
                        res = on_free(sid)
                        if asyncio.iscoroutine(res): await res
                    except Exception as e:
                        logger.error(f"Error during SID {sid} cleanup: {e}")
                
                size = self.active_sizes.pop(sid, 0)
                self.used -= size
                del self.refcounts[sid]
                self.available_sids.append(sid)
                self._scavenge_triggered.pop(sid, None)
                cond.notify_all()

    def get_available_for_large(self) -> int:
        return self.capacity - self.used

    def get_available_for_small(self) -> int:
        return max(0, self.capacity - self.used - self.reserved_for_large)

    def get_stats(self):
        return {
            "used": self.used,
            "total": self.capacity,
            "waiting_large": self.waiting_large,
            "active_count": len(self.refcounts),
            "available_sids": len(self.available_sids),
            "is_expanded": self.is_expanded,
            "is_draining": self.is_draining
        }

    async def drain(self):
        cond = self._ensure_sync()
        async with cond:
            self.is_draining = True
            while self.used > 0: await cond.wait()

    async def resume(self):
        cond = self._ensure_sync()
        async with cond:
            self.is_draining = False
            cond.notify_all()

    async def update_capacity(self, new_total: int, expanded: bool):
        cond = self._ensure_sync()
        async with cond:
            self.capacity = new_total
            self.is_expanded = expanded
            cond.notify_all()

    async def _run_hook(self, hook: str, label: str):
        try:
            logger.info("Executing %s hook: %s", label, hook)
            proc = await asyncio.create_subprocess_shell(hook, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("%s hook failed with code %d: %s", label, proc.returncode, stderr.decode().strip())
        except Exception as e:
            logger.exception("Failed to trigger %s hook: %s", label, e)
