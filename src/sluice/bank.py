import asyncio
import time
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
        self.seq_counter = 0
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        self.waiting_large = 0
        self.is_draining = False
        self.is_expanded = False

    async def acquire(self, requested_size: int, timeout: float = 60.0) -> int:
        """
        Acquires tokens from the bank. Blocks until space is available or timeout.
        - Blocks if draining.
        - Starvation Hook: Executes if a Large request waits for > 15s.
        """
        is_large = requested_size >= self.large_threshold
        start_time = time.time()
        hook_triggered = False

        async with self.condition:
            if self.is_draining:
                raise RuntimeError("Server is currently draining and not accepting new requests.")

            if is_large:
                self.waiting_large += 1
            
            try:
                while True:
                    available = self.total - self.used
                    
                    if not is_large and (self.waiting_large > 0 or (available - requested_size) < self.reserved_for_large):
                        can_fit = False
                    else:
                        can_fit = requested_size <= available

                    if can_fit:
                        if is_large:
                            self.waiting_large -= 1
                        self.seq_counter += 1
                        sid = self.seq_counter
                        self.used += requested_size
                        self.active_seqs[sid] = requested_size
                        return sid

                    # Level 3: Starvation Escalation
                    elapsed = time.time() - start_time
                    if is_large and self.starvation_hook and elapsed > 15.0 and not hook_triggered:
                        print(f"[BANK] LARGE STARVATION DETECTED ({elapsed:.1f}s). Triggering scavenge hook...")
                        asyncio.create_task(self._run_hook(self.starvation_hook, "Scavenge"))
                        hook_triggered = True

                    if elapsed > timeout:
                        if is_large:
                            self.waiting_large -= 1
                        raise TimeoutError(f"Token bank timeout after {elapsed:.1f}s: Needed {requested_size}")
                    
                    # Manual timeout loop since asyncio.Condition.wait() has no timeout
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
            except Exception:
                if is_large and 'can_fit' in locals() and can_fit is False:
                    self.waiting_large -= 1
                raise

    async def release(self, sid: int):
        async with self.condition:
            size = self.active_seqs.pop(sid, 0)
            self.used -= size
            
            # Check for Recovery (Level 4): If pool is idle and expanded, shrink back.
            # (Logic moved to server.py elasticity_loop, but bank provides the state)
            self.condition.notify_all()

    async def drain(self):
        """Enable draining mode: reject new, wait for old to finish."""
        async with self.condition:
            self.is_draining = True
            print("[BANK] Draining mode enabled. Waiting for active sequences to complete...")
            while self.used > 0:
                await self.condition.wait()
            print("[BANK] Drain complete.")

    async def resume(self):
        """Disable draining mode."""
        async with self.condition:
            self.is_draining = False
            print("[BANK] Resumed from drain.")
            self.condition.notify_all()

    async def update_total(self, new_total: int, expanded: bool):
        """Dynamically update pool size (e.g. after resizing context)."""
        async with self.condition:
            self.total = new_total
            self.is_expanded = expanded
            self.condition.notify_all()

    async def _run_hook(self, hook: str, label: str):
        try:
            print(f"[BANK] Executing {label} hook: {hook}")
            proc = await asyncio.create_subprocess_shell(
                hook,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                print(f"[BANK] {label} hook success.")
            else:
                print(f"[BANK] {label} hook failed: {stderr.decode()}")
        except Exception as e:
            print(f"[BANK] Error running {label} hook: {e}")
