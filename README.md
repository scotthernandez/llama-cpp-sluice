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
- **Anti-Starvation Bank:** A custom `TokenBank` with barrier logic ensures that large coding tasks are never starved by a flood of smaller chat requests.
- **Proxmox & Docker Native:** Includes first-class support for LXC GPU passthrough and CUDA-accelerated containers.
- **Dynamic Auto-Elasticity:** Automatically grows the VRAM pool by scavenging resources (stopping SST/TTS) when a large request arrives, and shrinks back once idle.

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
See the [Documentation](./docs/architecture.md) for a deep dive into the SWAMC pattern, the [Priority Lanes](./docs/priority-lanes.md) guide for details on the Anti-Starvation logic, and the [Concurrency Guide](./docs/concurrency.md) for technical details on locking.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
