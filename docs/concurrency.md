# Concurrency & Locking Architecture

`llama-cpp-sluice` uses a multi-layered locking strategy to ensure that VRAM allocation, model inference, and administrative tasks (like resizing) never conflict.

## 1. The Token Bank Barrier (`asyncio.Condition`)
The `TokenBank` class manages the VRAM pool using an `asyncio.Condition` variable.
- **Acquisition:** When a request asks for tokens, it checks the pool availability under a lock. If it cannot fit (due to the barrier or starvation policy), it enters a `wait()` state.
- **Notification:** When any request finishes, it calls `notify_all()`, which wakes up all waiting requests to re-evaluate the pool state.
- **Safety:** This ensures that multiple simultaneous API calls never "over-subscribe" the GPU memory.

## 2. Inference Isolation (`ThreadPoolExecutor`)
Because the underlying `llama.cpp` C calls are blocking and not natively async, the `server.py` uses:
```python
loop.run_in_executor(None, low_level_generate, ...)
```
- **Thread Safety:** While `llama.cpp` model weights are read-only and shareable, the **KV Cache (Context)** is stateful. Sluice uses **Sequence IDs** (`sid`) to isolate different requests within the same memory block.
- **Atomic Operations:** Internal `llama_decode` calls are handled by the `llama.cpp` thread-pool, which is managed at the C level.

## 3. Administrative Locking (The "Drain" Flow)
Administrative functions like `resize` or `flush` require a completely idle system.
- **The Drain Flag:** When `/v1/admin/drain` is called, a global `is_draining` flag is set.
- **API Blocking:** New requests hit the `TokenBank` and are immediately rejected with a 503 error if the drain flag is active.
- **Await Idle:** The admin function then `awaits` until `BANK.used == 0`.
- **Exclusive Access:** Once idle, the admin task (e.g., `hot_swap_context`) has exclusive access to the memory pointers to perform a safe recreation.

## 4. Starvation & Escalation Coordination
The `SCAVENGE_HOOK` is triggered in a non-blocking background task:
- This ensures that while the system is waiting for VRAM to clear (stopping SST/TTS), the API server remains responsive to other small requests or status queries.
- The `is_expanded` state tracks whether the system is in its "emergency" mode to prevent redundant hook executions.
