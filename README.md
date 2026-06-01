# MiniTensor

A minimal C++ tensor library built from scratch — a learning project for LLM inference/deployment engineering.

## Goal

Understand the foundations of inference engines (vLLM, llama.cpp) by building a miniature tensor computation library piece by piece. Each development phase maps directly to real components in production inference systems.

## Build

```bash
mkdir -p build && cd build
cmake ..
make
./minitensor
```

## Learning Roadmap

| Phase | Days | Topics | Milestone |
|---|---|---|---|
| **v0.1** | 1-5 | Memory management, Rule of Five, Move semantics, Templates, RAII | ✅ Core Tensor class with safe memory handling |
| **v0.2** | 6-10 | Multi-dimensional tensors (shape, stride), Memory pool, Operators (matmul, softmax, layernorm) | ✅ Simple inference demo |
| **v0.3** | 11-15 | Computation graph, Operator dispatch, Batching | Planned |
| **v0.4** | 16+ | CUDA backend, Quantization (FP16/INT8/INT4) | Planned |

## Current Status: v0.2 — Day 10

**Implemented:**

### v0.1 — Core Tensor (Day 1-5)
- [x] Constructor / Destructor
- [x] Rule of Five (copy ctor, copy assign, move ctor, move assign, destructor)
- [x] Templates — generic `Tensor<T>` with header-only implementation
- [x] RAII with `std::unique_ptr<T[]>`
- [x] Element-wise operators (`operator+`, `operator+=`, `operator-`, `operator*`)
- [x] Const correctness

### v0.2 — Multi-dimensional & Operators (Day 6-10)
- [x] N-dimensional tensor with shape & stride
- [x] Memory pool — pre-allocated buffer reuse
- [x] Matrix multiplication (matmul)
- [x] Softmax activation
- [x] Layer normalization
- [x] Simple inference demo (classifier + mini attention block)

**In Progress:**
- [ ] v0.3 — Computation graph construction & operator dispatch

## Architecture

```
minitensor/
├── tensor.h          # Tensor<T> class — header-only, generic dtype
├── memory_pool.h     # MemoryPool — pre-allocated buffer management
├── main.cpp          # Test driver & inference demos
└── CMakeLists.txt
```

## Author

[Xiaoda](https://github.com/Xiaoda11) — Automation undergrad, targeting LLM Inference/Deployment
