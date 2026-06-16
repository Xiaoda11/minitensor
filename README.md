# MiniTensor

从零手写的 C++ 张量计算库，覆盖 CPU 和 CUDA 双后端。学习项目，目标 LLM Inference/Deployment 岗位。

## 目录结构

```
minitensor/
├── cpu/                    # CPU 后端 (header-only)
│   ├── tensor.h            # N 维张量模板 (Rule of 5, broadcasting, stride)
│   ├── compute_graph.h     # 计算图引擎 (DAG 构建 + 拓扑排序 + 前向执行)
│   ├── memory_pool.h       # 内存池
│   ├── main.cpp            # 47 个测试 (Day 1–15)
│   └── CMakeLists.txt
├── cuda/                   # CUDA 后端 (Phase 1+2)
│   ├── vector_add.cu       # Week 1: Grid/Block/Thread 入门
│   ├── matmul_naive.cu     # Week 1 (续): 朴素矩阵乘法
│   ├── matmul_tiled.cu     # Week 2: Tiling + Shared Memory
│   ├── softmax.cu          # Week 3: Softmax (Warp Reduce)
│   ├── layernorm.cu        # Week 3: LayerNorm
│   ├── attention.cu        # Week 4: Fused Attention
│   ├── kv_cache.h          # Phase 2: KV Cache 数据结构
│   ├── phase2_main.cu      # Phase 2: KV Cache 测试
│   ├── prefill.cu          # Phase 2: Prefill 阶段
│   ├── decode.cu           # Phase 2: Decode 阶段
│   ├── generate.cu         # Phase 2: generate() 循环
│   ├── CMakeLists.txt
│   └── README.md           # 学习路线图
└── README.md               # 本文件
```

## 快速开始

### CPU 版本
```bash
cd cpu
g++ -std=c++17 -O2 main.cpp -o main && ./main
```

### CUDA 版本 (需要 NVIDIA GPU + CUDA Toolkit)
```bash
cd cuda
nvcc -O2 vector_add.cu -o vector_add && ./vector_add
```

## 版本历史

| 版本 | 内容 | 状态 |
|------|------|------|
| v0.1 | Tensor 基础 (Rule of 5, 模板) | ✓ |
| v0.2 | Shape/Stride, 基础算子, 推理 Demo | ✓ |
| v0.3 | 计算图引擎 + 2 层 MLP 推理 | ✓ |
| v0.4 | CUDA Kernel 重写 | ✓ |
| v0.5 | KV Cache + Prefill/Decode/Generate (Phase 2 完成) | ✓ |

## 技术栈

- C++17, CUDA C++
- 计算图: 拓扑排序 (Kahn)、DAG 前向执行
- 算子: matmul, softmax, layernorm, attention, ReLU, broadcast add/mul, transpose
- CUDA: Grid/Block/Thread, Shared Memory, Warp Reduce