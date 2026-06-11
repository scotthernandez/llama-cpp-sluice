# Sluice API Server Documentation

The `llama-cpp-sluice` API is a high-performance, asynchronous gateway built on **FastAPI**. It is designed to behave like a standard OpenAI-compatible server while providing advanced VRAM management features.

## 🚀 Core API Features

### 1. OpenAI Compatibility
The server implements the standard `/v1/chat/completions` endpoint. This allows you to use standard libraries (Python `openai`, `LiteLLM`, `LangChain`) without modifying your client-side code.

### 2. Virtual Context URLs (Context-Aware Routing)
Sluice supports encoding the required context size directly in the URL:
- `POST /v1/ctx/2048/chat/completions` (Chat optimized)
- `POST /v1/ctx/32768/chat/completions` (Coding optimized)

This is the primary way to integrate with gateways like **new-api**, allowing the load-balancer to signal VRAM needs based on the requested model alias.

### 3. Dynamic Barrier Gating
The server uses an internal **Token Bank** to gate requests. 
- **Wait Queues:** If the VRAM is full, requests are not immediately rejected. They "park" in an asynchronous wait queue for up to 60 seconds, providing a smoother user experience than hard errors.
- **Anti-Starvation:** Large requests (coding) are prioritized over small requests (chat) when resources are tight.

## 🛠️ Administrative Endpoints

The server provides a suite of `/v1/admin` tools for runtime infrastructure management:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/v1/admin/drain` | `POST` | Gracefully stops accepting new requests while allowing active ones to finish. |
| `/v1/admin/resume` | `POST` | Resumes accepting traffic after a drain or maintenance. |
| `/v1/admin/defrag` | `POST` | Triggers a `llama_kv_cache_defrag` to compact used tokens and eliminate "holes" in VRAM. |
| `/v1/admin/resize` | `POST` | Triggers a **Hot-Swap**: Drains the system and recreates the context with a new size (e.g., to reclaim VRAM for other models). |

## 📐 Technical Support

- **AsyncIO Native:** The server handles all networking and state management asynchronously, ensuring high throughput even under heavy load.
- **Threaded Inference:** CPU-bound `llama.cpp` calls are offloaded to a `ThreadPoolExecutor` to prevent blocking the main event loop.
- **Pydantic Validation:** All request and response payloads are strictly validated using Pydantic v2, ensuring data integrity.
- **Automatic Tokenizer Support:** The server automatically detects the tokenizer (BPE/SentencePiece) from the loaded `.gguf` file.

## 🚦 Headers & Meta-Data
In addition to JSON payloads, you can control the Sluice behavior via headers:
- `X-Sluice-Required-Ctx`: Override the context size for a specific request.
- Standard OpenAI headers (`Authorization`, `Content-Type`) are fully supported.
