# Sluice API Server Documentation

The `llama-cpp-sluice` API is a high-performance, asynchronous gateway built on **FastAPI**. It is designed to behave like a standard OpenAI-compatible server while providing advanced VRAM management features.

## 🚀 Core API Features

### 1. OpenAI Compatibility
The server implements the standard `/v1/chat/completions` endpoint. This allows you to use standard libraries (Python `openai`, `LiteLLM`, `LangChain`) without modifying your client-side code.
- **Streaming:** Full support for `stream: true` using Server-Sent Events (SSE).
- **Tool Calling:** Supports `tools` and `tool_choice` for agentic workflows.

### 2. Embeddings Support
The `/v1/embeddings` endpoint provides OpenAI-compatible vector generation using the shared model weights.
- Optimized for batch processing.
- Dynamically gates VRAM usage via the Token Bank.

### 3. Virtual Context URLs (Context-Aware Routing)
Sluice supports encoding the required context size directly in the URL:
- `POST /v1/ctx/2048/chat/completions` (Chat optimized)
- `POST /v1/ctx/32768/chat/completions` (Coding optimized)

### 4. Monitoring & Metrics
The server exports Prometheus-compatible metrics at the `/metrics` endpoint:
- `sluice_vram_used`: Currently used tokens in the pool.
- `sluice_prefix_cache_hits`: Real-time tracking of deduplication efficiency.
- `sluice_latency_seconds`: Histogram of generation speeds.

### 5. Dynamic Barrier Gating
The server uses an internal **Token Bank** to gate requests. 
- **Wait Queues:** If the VRAM is full, requests "park" in an asynchronous wait queue.
- **Anti-Starvation:** Large requests (coding) are prioritized over small requests (chat) when resources are tight.

## 🛠️ Administrative Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/v1/admin/drain` | `POST` | Gracefully stops accepting new requests. |
| `/v1/admin/resume` | `POST` | Resumes accepting traffic. |
| `/v1/admin/resize` | `POST` | Triggers a **Hot-Swap** to change the pool size at runtime. |
| `/v1/admin/self-test` | `GET` | Runs a 'Golden Prompt' suite to verify model sanity and accuracy. |

## 📐 Technical Support

- **AsyncIO Native:** Handles networking and state management asynchronously.
- **Threaded Inference:** Offloads C++ inference to a `ThreadPoolExecutor`.
- **Automatic Tokenizer Detection:** Reads BPE/SentencePiece settings directly from GGUF.
- **Optional Bearer Auth:** Secure the API via `SLUICE_API_KEY`.
