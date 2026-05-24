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
| **v0.1** | 1-5 | Memory management, Rule of Three, Move semantics, Templates, RAII | Core Tensor class with safe memory handling |
| **v0.2** | 6-10 | Multi-dimensional tensors (shape, stride), Memory pool | N-dimensional tensor with contiguous buffer |
| **v0.3** | 11-15 | Operators (matmul, softmax, layernorm), Computation graph | Run a 2-layer transformer forward pass |
| **v0.4** | 16+ | CUDA backend, Quantization (FP16/INT8/INT4) | GPU-accelerated inference |

## Current Status: v0.1 — Day 2

**Implemented:**
- [x] Constructor / Destructor (new/delete)
- [x] Copy Constructor (deep copy)
- [x] Copy Assignment Operator (`operator=`)
- [x] Addition Operator (`operator+`)
- [x] In-place Addition (`operator+=`)
- [x] Const correctness (const-overloaded methods)

**In Progress:**
- [ ] Move Semantics (`std::move`)
- [ ] Templates (generic Tensor<T>)
- [ ] Smart Pointers / RAII

## Architecture

```
minitensor/
├── tensor.h      # Tensor class declaration
├── tensor.cpp    # Tensor implementation
├── main.cpp      # Test driver
└── CMakeLists.txt
```

## Author

[Xiaoda](https://github.com/Xiaoda11) — Automation undergrad, targeting LLM Inference/Deployment
