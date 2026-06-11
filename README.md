# Llama-CPP Sluice 🌊

**Dynamic Asymmetric Inference Server with Anti-Starvation VRAM Management.**

`llama-cpp-sluice` is a high-performance wrapper for `llama-cpp-python` designed for multi-GPU home-lab environments with asymmetric PCIe bottlenecks. It implements the **Shared-Weight Asymmetric Multi-Context (SWAMC)** architecture.

## 🤖 Disclosure
**This project is 100% AI-generated under the light supervision of Scott Hernandez.** It is a personal-use tool created to solve specific hardware constraints in multi-GPU PCIe bottleneck environments.

## 🚀 Key Features
- **Shared Weight Pooling:** Load model weights into VRAM exactly once; share them across an infinite number of virtual context windows.
- **Asymmetric Contexts:** Serve 2k, 32k, and 96k requests simultaneously from the same pool without VRAM duplication.
- **Anti-Starvation Bank:** A custom `TokenBank` with barrier logic ensures that large coding tasks are never starved by a flood of smaller chat requests.
- **Proxmox & Docker Native:** Includes first-class support for LXC GPU passthrough and CUDA-accelerated containers.

## 🛠️ Quick Start

### 1. Installation
```bash
pip install .
```

### 2. Environment Setup
Set your model path and pool sizes:
```bash
export SLUICE_MODEL_PATH="/path/to/model.gguf"
export SLUICE_TOTAL_POOL=98304
export SLUICE_RESERVED_POOL=32768
```

### 3. Run the Server
```bash
python -m sluice.server
```

## 📐 Architecture
See the [Documentation](./docs/architecture.md) for a deep dive into the SWAMC pattern and the [Priority Lanes](./docs/priority-lanes.md) guide for details on the Anti-Starvation logic.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
