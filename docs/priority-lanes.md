# Priority Lanes & Anti-Starvation

The `TokenBank` class implement a "Barrier Logic" system to manage asymmetric request sizes.

## The Barrier Logic
The bank is initialized with a `total_pool` and a `reserved_for_large` amount.

### 1. Small Request Barrier
Requests that declare a `required_ctx < 16,384` are considered **Small**. 
- They are forbidden from entering the pool if doing so would leave fewer than `reserved_for_large` tokens available.
- This ensures that a flood of tiny requests (e.g., 50 people saying "Hi") can never exhaust the space needed for a single large coding request.

### 2. Anti-Starvation Lock
If a **Large** request arrives and the pool is full, it enters a `waiting_large` state.
- While `waiting_large > 0`, **all new Small requests are blocked**, even if they would technically fit in the remaining space.
- This forces the system to "drain" active small tasks until the Large task has enough contiguous VRAM to execute.

## Wait Queues
Sluice uses an `asyncio.Condition` variable to implement high-efficiency waiting. Instead of polling or failing with a 503 error, requests will pause their execution and wait for the bank to `notify_all()` when tokens are released.
