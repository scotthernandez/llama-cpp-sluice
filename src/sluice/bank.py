import asyncio
import time
from typing import Dict, Optional

class TokenBank:
    def __init__(self, total_tokens: int, reserved_for_large: int, large_threshold: int = 16384):
        self.total = total_tokens
        self.reserved_for_large = reserved_for_large
        self.large_threshold = large_threshold
        self.used = 0
        self.active_seqs: Dict[int, int] = {}
        self.seq_counter = 0
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        self.waiting_large = 0

    async def acquire(self, requested_size: int, timeout: float = 60.0) -> int:
        """
        Acquires tokens from the bank. Blocks until space is available or timeout.
        - Large requests (>= large_threshold) can use the ENTIRE pool.
        - Small requests can only use (Total - Reserved).
        - Small requests are blocked if a Large request is waiting (Anti-Starvation).
        """
        is_large = requested_size >= self.large_threshold
        start_time = time.time()
        
        async with self.condition:
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

                    if (time.time() - start_time) > timeout:
                        if is_large:
                            self.waiting_large -= 1
                        raise TimeoutError(f"Token bank timeout: Needed {requested_size}")
                    
                    await self.condition.wait()
            except Exception:
                # Ensure we decrement waiting counter on unexpected errors/cancellation
                if is_large and 'can_fit' in locals() and can_fit is False:
                    self.waiting_large -= 1
                raise

    async def release(self, sid: int):
        async with self.condition:
            size = self.active_seqs.pop(sid, 0)
            self.used -= size
            self.condition.notify_all()
