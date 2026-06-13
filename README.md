# Llama-CPP Sluice 🌊

**Dynamic Asymmetric Inference Server with Anti-Starvation VRAM Management.**

`llama-cpp-sluice` is a high-performance wrapper for `llama-cpp-python` designed for multi-GPU home-lab environments with asymmetric PCIe bottlenecks. It implements the **Shared-Weight Asymmetric Multi-Context (SWAMC)** architecture.

### 💻 Hardware Agnostic
While optimized for multi-GPU setups (PCIe x16/x4), Sluice is **fully compatible with single-GPU and CPU-only (OpenBLAS/AVX) systems**. Any environment supported by `llama.cpp` will benefit from its dynamic VRAM management and anti-starvation logic.

## 🤖 Disclosure
**This project is 100% AI-generated under the light supervision of Scott Hernandez.** It is a personal-use tool created to solve specific hardware constraints in multi-GPU PCIe bottleneck environments.

## 🚀 Key Features
- **Shared Weight Pooling:** Load model weights into VRAM exactly once; share them across an infinite number of virtual context windows.
- **Asymmetric Contexts:** Serve 2k, 32k, and 96k requests simultaneously from the same pool without VRAM duplication.
- **Adaptive Context Negotiation:** Decoupled middleware that transparently trims conversation history (middle-out) to fit VRAM budget, protecting agent stability.
- **Radix Prefix Caching:** Automatically deduplicate VRAM for shared system prompts or large context blocks using zero-copy cloning.
- **Native Chat Templates:** Automatically detects and uses the model's native Jinja2 chat template from GGUF metadata for perfect prompt fidelity.
- **Embeddings Support:** High-performance `/v1/embeddings` endpoint sharing the same model weights.
- **Anti-Starvation Bank:** A custom `TokenBank` with barrier logic ensures that large coding tasks are never starved by a flood of smaller chat requests.
- **Proxmox & Docker Native:** Optimized for unprivileged LXC containers and CUDA-accelerated Docker environments.
- **Dynamic Auto-Elasticity:** Automatically grows the VRAM pool by scavenging resources (stopping SST/TTS) when a large request arrives, and shrinks back once idle.
- **Tool Calling Support:** Full support for OpenAI-standard `tools` and `tool_choice` parameters.
**- Optional Authentication:** Secure your server with a simple `SLUICE_API_KEY` Bearer Token.

### ⚠️ Known Limitations

**Single-Sequence Inference.** All inference is serialized to a single worker via a
hardware execution lock (`llm_lock`).  While `n_seq_max` is configured to 16, the lock
ensures that at most **one sequence generates at any given moment** — prefill plus the
full decode loop run atomically under one lock hold.  This is by design:

* **Prevents starvation** — the TokenBank's anti-starvation bank reserves VRAM for large
  requests; without serialization a flood of tiny requests could starve them.
* **Eliminates latency spikes** — concurrent multi-sequence decoding on the same context
  causes KV-cache contention and non-deterministic decode ordering, which spikes latency
  unpredictably.  Serialisation gives stable per-request latency.
* **Avoids C-level crashes** — the underlying `llama.cpp` C API does not support true
  concurrent multi-sequence decoding on the same context pointer.  Calling `llama_decode`
  from multiple OS threads can trigger SIGSEGV or silent KV-cache corruption.

Concurrent requests are multiplexed at the token-bank level but processed sequentially.
For genuine parallel generation a multi-engine / multi-context deployment would be needed
(not currently implemented).  See `src/sluice/server.py` for the full serialization
comment and `docs/concurrency.md` for architecture details.

## 🛠️ Quick Start

### 1. Installation
```bash
pip install .
```

### 2. Environment Setup
Basic tuning:
```bash
export SLUICE_MODEL_PATH="/path/to/model.gguf"
export SLUICE_BASE_POOL=98304
export SLUICE_RESERVED_POOL=32768
export SLUICE_API_KEY="your-secret-key"
```

Advanced Elasticity (VRAM Scavenging):
```bash
export SLUICE_AUTO_ELASTICITY=true
export SLUICE_SCAVENGE_HOOK="/path/to/scavenge.sh"
export SLUICE_RECOVERY_HOOK="/path/to/recovery.sh"
```

### 3. Run the Server
```bash
python -m sluice.server
```

## 📐 Architecture
See the [Documentation](./docs/architecture.md) for a deep dive into the SWAMC pattern, the [Priority Lanes](./docs/priority-lanes.md) guide for details on the Anti-Starvation logic, the [Concurrency Guide](./docs/concurrency.md) for technical details on locking, and the [API Reference](./docs/api.md) for server capabilities.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
