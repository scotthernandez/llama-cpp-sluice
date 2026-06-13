# Manual Test Plan: Graceful Shutdown Within 1 Second

## Background

The server in `src/sluice/server.py` registers signal handlers for `SIGINT` and `SIGTERM` that set a global `SHUTTING_DOWN` flag (line 81-87). The generation loop (`low_level_generate` at line 265) checks this flag at the top of each decoding iteration (line 291) and also the stream step function (line 343-344). If `SHUTTING_DOWN` is set, the loop sets `finish_reason = "shutdown"` and breaks out of the generation loop.

**Important caveat:** The `SHUTTING_DOWN` check happens only at the top of the inner generation loop (one check per token). Between the time `kill -INT/TERM` is sent and the loop checks the flag, the server is still generating. The actual wall-clock time depends on how quickly the OS delivers the signal, how many tokens are buffered, and how fast each `llama_decode` call completes. The goal is to confirm that **after** the signal is received, the loop exits within 1 second.

## Test Plan

### 1. Start a long generation

1.1. Start the sluice server in the background with a model and a generation that will take more than 1 second:

```bash
# Start server (assuming a model is available at /models/gguf/model.gguf)
SLUICE_MODEL_PATH=/models/gguf/model.gguf python -m sluice \
  -c 4096 --max-tokens 2048 &
SERVER_PID=$!

# Wait for the server to be ready
sleep 5
curl -s http://localhost:8001/v1/models > /dev/null || exit 1
```

1.2. Trigger a long generation via a streaming request. Use a prompt that will generate many tokens:

```bash
# Fire-and-forget a streaming request in the background;
# capture the PID of the curl process for later reference.
curl -s -N http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sluice-model",
    "messages": [{"role": "user", "content": "Write a very long essay about nothing, continuing for at least 2000 words. Repeat the phrase \"this is a test of the emergency broadcast system\" over and over without stopping. Do not end your response. Continue generating tokens until you are forced to stop."}],
    "max_tokens": 2048,
    "stream": true
  }' > /tmp/stream_output.txt 2>&1 &
CURL_PID=$!

# Wait until curl has connected and streaming has started.
# Check that at least some tokens have been received.
sleep 2

# Verify streaming is in progress by checking output size has grown.
BEFORE_SIZE=$(stat -c%s /tmp/stream_output.txt 2>/dev/null || echo 0)
echo "Streaming started. Output so far: ${BEFORE_SIZE} bytes"
```

### 2. Send SIGINT or SIGTERM to the server process

2.1. Send the signal directly to the server's Python process:

```bash
# Option A - SIGINT (Ctrl+C equivalent)
kill -INT $SERVER_PID

# Option B - SIGTERM (clean termination)
# kill -TERM $SERVER_PID
```

2.2. Alternatively, send the signal to the uvicorn child process (which may differ from the shell's PID). Find the correct PID:

```bash
# Find the actual Python process running sluice
pgrep -f "python.*sluice"
# Then send to that PID:
kill -INT <pid>
```

### 3. Confirm the generation loop has exited

3.1. The server should print a shutdown message to stdout/stderr:

```bash
# Check server logs for the shutdown confirmation
# Expected log output (from server.py line 83):
# [SLUICE] Signal 2 received. Starting graceful shutdown...
# [SLUICE] Shutting down gateway...
# [SLUICE] Gateway offline.

# Or check the stream output for a finish_reason of "shutdown"
grep '"shutdown"' /tmp/stream_output.txt
```

3.2. The curl process should detect the closed connection and exit:

```bash
# After shutdown, the server closes the SSE connection.
# curl should exit with a non-zero code (connection reset).
wait $CURL_PID 2>/dev/null
CURL_EXIT=$?
echo "curl exit code: $CURL_EXIT"
# Expected: non-zero (e.g., 18 or 141) due to broken pipe / connection reset.
```

3.3. Verify the server process has exited:

```bash
# Check that the server process no longer exists
if kill -0 $SERVER_PID 2>/dev/null; then
  echo "FAIL: Server process $SERVER_PID is still running"
else
  echo "PASS: Server process $SERVER_PID has exited"
fi
```

### 4. Measure the shutdown time and confirm it is under 1 second

4.1. Timestamp-based measurement using `time` and `date`:

```bash
# Record the time just before sending the signal
SIG_TIME=$(date +%s%N)   # nanoseconds since epoch
kill -INT $SERVER_PID
SIG_TIME_NS=$SIG_TIME

# Poll until the server process exits or 5 seconds elapse
START_NS=$(date +%s%N)
while kill -0 $SERVER_PID 2>/dev/null; do
  sleep 0.1
  if (( $(date +%s%N) - START_NS > 5000000000 )); then
    echo "FAIL: Server still running after 5 seconds"
    break
  fi
done

# Measure wall-clock time from signal to exit
END_NS=$(date +%s%N)
DURATION_MS=$(( (END_NS - SIG_TIME_NS) / 1000000 ))
echo "Shutdown duration: ${DURATION_MS} ms"

# Check if under 1 second
if [ "$DURATION_MS" -lt 1000 ]; then
  echo "PASS: Shutdown completed in ${DURATION_MS} ms (< 1000 ms)"
else
  echo "FAIL: Shutdown took ${DURATION_MS} ms (>= 1000 ms)"
fi
```

4.2. More precise measurement by wrapping with Python's `time` module:

```bash
# Create a small helper script for nanosecond precision
python3 -c "
import subprocess, signal, time, os

# Start server in background
proc = subprocess.Popen(
    ['python3', '-m', 'sluice', '-m', '/models/gguf/model.gguf'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# Start a long generation
gen_proc = subprocess.Popen(
    ['curl', '-s', '-N', 'http://localhost:8001/v1/chat/completions',
     '-H', 'Content-Type: application/json',
     '-d', '{\"model\":\"sluice-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a very long essay.\"}],\"max_tokens\":2048,\"stream\":true}'],
    stdout=open('/tmp/stream_output.txt','w')
)

# Wait for streaming to start
time.sleep(2)

# Record time, send signal, poll for exit
t0 = time.monotonic_ns()
os.kill(proc.pid, signal.SIGINT)
proc.wait()          # blocks until process exits
t1 = time.monotonic_ns()

duration_ms = (t1 - t0) / 1_000_000
print(f'Shutdown duration: {duration_ms:.2f} ms')
print(f'PASS' if duration_ms < 1000 else 'FAIL')
"
```

### 5. Verify expected behavior after shutdown

5.1. No orphan processes:

```bash
# After shutdown, confirm no leftover Python processes for this instance
ps aux | grep "[p]ython.*sluice"
# Expected: no output (no orphan Python processes)

# Also check for any stray uvicorn children
ps aux | grep "[u]vicorn"
# Expected: no output
```

5.2. Clean exit code:

```bash
# Check the exit code from the server process
wait $SERVER_PID 2>/dev/null
EXIT_CODE=$?
echo "Server exit code: $EXIT_CODE"
# Expected: 0 (clean exit) or a non-zero code indicating intentional shutdown.
# Note: uvicorn may return 0; the important thing is no segfault/core dump.
```

5.3. No core dump or crash artifacts:

```bash
# Check for core dumps (indicates a crash rather than graceful exit)
find /tmp -name "core*" -mmin -1
find . -name "core*" -maxdepth 1
# Expected: no output (no core dumps)
```

5.4. Server cannot accept new connections after shutdown:

```bash
# Attempt a request after shutdown completes
sleep 1   # wait for shutdown to fully complete
if curl -s --max-time 2 http://localhost:8001/v1/models; then
  echo "FAIL: Server still accepting connections after shutdown"
else
  echo "PASS: Server correctly refuses new connections after shutdown"
fi
```

### 6. Additional edge-case checks

6.1. **Signal during prefill (prompt decoding):**
- The prefill phase iterates over token chunks. The `SHUTTING_DOWN` check is only at the top of the **generation** loop (line 291), not inside the prefill loop (lines 272-284).
- Test: send SIGINT immediately after submitting the request (before generation starts).
- Expected: the prefill may complete, but generation should stop on the next iteration check.

6.2. **Multiple rapid signals:**

```bash
# Send multiple signals in quick succession
for i in 1 2 3; do kill -INT $SERVER_PID; sleep 0.01; done
wait $SERVER_PID
echo "Server exited cleanly after multiple signals"
```

6.3. **SIGTERM vs SIGINT:** Repeat all of the above with `kill -TERM` instead of `kill -INT` to verify both paths behave identically.

6.4. **Non-streaming (blocking) generation:**
- Repeat the test with `"stream": false` in the request.
- The `low_level_generate` function (line 265) returns the full result before the HTTP response is sent, so the shutdown behavior should be the same but the timing profile will differ (no SSE chunking to poll).

## Summary Checklist

| Step | What to Verify | Pass Criteria |
|------|---------------|---------------|
| 1 | Server starts, generation begins | Server responds to `/v1/models`, curl receives streamed tokens |
| 2 | Signal is delivered | `kill -INT` / `kill -TERM` succeeds, no "no such process" error |
| 3 | Loop exits | Server logs `[SLUICE] Signal ... received`, curl sees `finish_reason: shutdown` |
| 4 | Under 1 second | Wall-clock time from signal to process exit < 1000 ms |
| 5 | Clean shutdown | No orphan processes, no core dumps, exit code is clean, server stops accepting connections |
| 6 | Edge cases | Works for SIGTERM, non-streaming, rapid signals, and signal during prefill |
