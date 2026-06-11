# SWAMC Architecture

The **Shared-Weight Asymmetric Multi-Context (SWAMC)** pattern addresses the "Context vs. Weights" memory trade-off in LLM inference.

## The Problem
In standard `llama.cpp` deployments, each parallel slot (`-np`) requires a static context allocation. If you want two 32k slots, you must reserve 64k of VRAM upfront. If those slots are idle, the VRAM is wasted. If you try to share weights across separate processes, you duplicate the massive model weights in VRAM.

## The Sluice Solution
Sluice uses the low-level `llama_cpp._internals` to load a single `LlamaModel` and map a single large `LlamaContext` over it.

### Virtual Sequences
Instead of separate contexts, Sluice uses **Sequence IDs** (`seq_id`) within the same master context. 
1. The model weights stay resident in VRAM.
2. The KV cache is treated as a single "Token Reservoir."
3. Each request is dynamically assigned a `seq_id` and a slice of the reservoir.

### Radix Prefix Caching
Sluice implements a high-performance **Radix Cache** that deduplicates VRAM for shared prefixes (like system prompts or large context files).
- **Zero-Copy Cloning:** Uses `llama_memory_seq_cp` to instantly clone pre-filled tokens from a cached sequence to a new request.
- **Throughput:** Near-zero "Time to First Token" for cached prompts.
- **LRU Eviction:** Automatically manages cache size to stay within VRAM limits.

### PCIe Bottleneck Mitigation
By using `split_mode: layer`, inter-GPU communication is minimized to token activation transfers. This bypasses the latency of Tensor Parallelism over narrow PCIe x4 channels typical in home-lab motherboard chipsets.

### Native Template Fidelity
Sluice extracts the `tokenizer.chat_template` directly from GGUF metadata. This ensures that the model (Qwen, Llama, Gemma, etc.) always sees the exact prompt format it was optimized for, reducing hallucinations and instruction-following errors.
