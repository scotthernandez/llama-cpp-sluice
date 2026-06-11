import asyncio
import time
import subprocess
from typing import Dict, Optional

class TokenBank:
    def __init__(self, total_tokens: int, reserved_for_large: int, large_threshold: int = 16384, 
                 starvation_hook: Optional[str] = None, recovery_hook: Optional[str] = None):
        self.base_total = total_tokens
        self.total = total_tokens
        self.reserved_for_large = reserved_for_large
        self.large_threshold = large_threshold
        self.starvation_hook = starvation_hook
        self.recovery_hook = recovery_hook
        
        self.used = 0
        self.active_seqs: Dict[int, int] = {}
        self.pinned_seqs: Dict[int, int] = {} # sid -> size
        self.seq_counter = 0
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        self.waiting_large = 0
        self.is_draining = False
        self.is_expanded = False

    async def acquire(self, requested_size: int, timeout: float = 60.0) -> int:
        """
        Acquires tokens from the bank. Blocks until space is available or timeout.
        """
        is_large = requested_size >= self.large_threshold
        start_time = time.time()
        hook_triggered = False

        async with self.condition:
            if self.is_draining:
                raise RuntimeError("Server is currently draining.")

            if is_large: self.waiting_large += 1
            
            try:
                while True:
                    available = self.total - self.used
                    
                    if not is_large and (self.waiting_large > 0 or (available - requested_size) < self.reserved_for_large):
                        can_fit = False
                    else:
                        can_fit = requested_size <= available

                    if can_fit:
                        if is_large: self.waiting_large -= 1
                        self.seq_counter += 1
                        sid = self.seq_counter
                        self.used += requested_size
                        self.active_seqs[sid] = requested_size
                        return sid

                    elapsed = time.time() - start_time
                    if is_large and self.starvation_hook and elapsed > 15.0 and not hook_triggered:
                        print(f"[BANK] STARVATION DETECTED. Running hook...")
                        asyncio.create_task(self._run_hook(self.starvation_hook, "Scavenge"))
                        hook_triggered = True

                    if elapsed > timeout:
                        if is_large: self.waiting_large -= 1
                        raise TimeoutError(f"Bank timeout: Needed {requested_size}")
                    
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=min(1.0, timeout - elapsed))
                    except asyncio.TimeoutError:
                        continue
            except Exception:
                if is_large and 'can_fit' in locals() and can_fit is False:
                    self.waiting_large -= 1
                raise

    async def release(self, sid: int, pin: bool = False):
        async with self.condition:
            size = self.active_seqs.pop(sid, 0)
            if pin:
                self.pinned_seqs[sid] = size
                print(f"[BANK] PINNED seq {sid} ({size} tokens). Used: {self.used}/{self.total}")
            else:
                self.used -= size
            self.condition.notify_all()

    async def evict(self, sid: int):
        """Evicts a pinned sequence."""
        async with self.condition:
            size = self.pinned_seqs.pop(sid, 0)
            self.used -= size
            print(f"[BANK] EVICTED seq {sid} ({size} tokens). Free: {self.total - self.used}")
            self.condition.notify_all()

    async def drain(self):
        async with self.condition:
            self.is_draining = True
            while self.used > 0: await self.condition.wait()

    async def resume(self):
        async with self.condition:
            self.is_draining = False
            self.condition.notify_all()

    async def update_total(self, new_total: int, expanded: bool):
        async with self.condition:
            self.total = new_total
            self.is_expanded = expanded
            self.condition.notify_all()

    def get_stats(self):
        return {
            "used": self.used,
            "total": self.total,
            "waiting_large": self.waiting_large,
            "is_expanded": self.is_expanded,
            "is_draining": self.is_draining,
            "pinned_count": len(self.pinned_seqs)
        }

    async def _run_hook(self, hook: str, label: str):
        try:
            proc = await asyncio.create_subprocess_shell(hook, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.communicate()
        except Exception as e: print(f"[BANK] Hook error: {e}")
