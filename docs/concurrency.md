# Concurrency & Locking Architecture

`llama-cpp-sluice` uses a multi-layered locking strategy to ensure that VRAM allocation, model inference, and administrative tasks (like resizing) never conflict.

## 1. The Token Bank Barrier (`asyncio.Condition`)
The `TokenBank` class manages the VRAM pool using an `asyncio.Condition` variable.
- **Acquisition:** When a request asks for tokens, it checks the pool availability under a lock. If it cannot fit (due to the barrier or starvation policy), it enters a `wait()` state.
- **Notification:** When any request finishes, it calls `notify_all()`, which wakes up all waiting requests to re-evaluate the pool state.
- **Safety:** This ensures that multiple simultaneous API calls never "over-subscribe" the GPU memory.
## 2. Inference Isolation (`ThreadPoolExecutor` + `llm_lock`)

Because the underlying `llama.cpp` C calls are blocking and not natively async, the `server.py` uses:
```python
loop.run_in_executor(None, low_level_generate, ...)
```

However, **inference is further serialized by a `threading.Lock` (`llm_lock`)**. Every inference
path — non-streaming, streaming, embeddings, and tool calls — acquires this lock before calling
`llama_decode` or `llama_sampler_sample`.  The result:

- **At most one sequence generates at any moment.**  A burst of N concurrent requests will
  be processed sequentially, one after another, in the order they acquire the lock.
- **Prefill + decode are atomic.**  A request's entire prompt prefill plus its full decode
  loop runs under a single lock hold; there is no interleaving between two sequences'
  decode steps.
- **`n_seq_max` is not a concurrency guarantee.**  The context is configured with
  `n_seq_max=16` (engine.py, line 125), which tells llama.cpp how many distinct sequence
  IDs it can track internally.  But the `llm_lock` limits *live inference* to 1 sequence.
  Users should not interpret 16 as "16 parallel generations."

### Why serialization is intentional (not a bug)

* **Prevents starvation.**  The TokenBank's anti-starvation bank reserves VRAM for large
  requests.  Without serialization a flood of tiny requests could starve large ones because
  llama.cpp's internal scheduler does not honour external priority hints.
* **Eliminates latency spikes from context thrashing.**  llama.cpp's KV-cache is stateful
  per sequence on a single context pointer.  Concurrent multi-sequence decoding on the same
  context causes cache-line contention and non-deterministic decode ordering, which can spike
  latency unpredictably.  Serialisation gives stable, predictable per-request latency.
* **Avoids C-level crashes.**  The underlying `llama.cpp` C API does not support true
  concurrent multi-sequence decoding on the same context pointer.  Calling `llama_decode`
  from multiple OS threads simultaneously can trigger SIGSEGV or silent KV-cache corruption.

See `src/sluice/server.py` (lines 69–127) for the full serialization comment and
`docs/architecture.md` for the SWAMC pattern overview.

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
