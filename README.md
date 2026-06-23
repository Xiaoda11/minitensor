## Performance Analysis

> Profiling 环境: RTX 2060 (CC 7.5, 28 SMs), Nsight Compute 三刀流方法。
> 详细数据: [docs/benchmarks.md](docs/benchmarks.md) · [cuda/README.md](cuda/README.md)

### Nsight Compute 核心指标

| Kernel | SM% | Warp% | Top 1 Stall | 真实瓶颈 |
|--------|-----|-------|-------------|----------|
| matmul_naive | 97.46 | 97.88 | Long Scoreboard 10.92 | **memory-bound** |
| matmul_tiled | 96.41 | 97.84 | Long Scoreboard 8.12 + Barrier 5.47 | **memory + sync 混合** |
| softmax_warp_reduce | 41.78 | 58.57 | Barrier 6.72 | **sync-bound** |
| layernorm_welford | 48.06 | 63.85 | Barrier 5.35 + Wait | **依赖链** |
| attention_fused_v2 | 59.70 | 70.78 | Wait 3.38 | **并行度不足** |

### Key Insights

#### matmul_naive: Memory-Bound, Not Compute-Bound

SM% = 97.46% 极具欺骗性——kernel "看起来"很忙，但 Warp State Statistics 揭示真相: Top 1 Stall 是 Long Scoreboard (10.92 inst/issue)。这意味着 warp 绝大部分时间在等 global memory load (`LDG`) 返回数据，SM 只是因为 load 指令占满了流水线才显得 97% 活跃。→ **瓶颈是 HBM 带宽，不是 ALU。**

#### matmul_tiled: Stall Category Trade, Not Better Locality

Long Scoreboard 从 10.92 降到 8.12（等内存变少了），但 Barrier (5.47) 作为新瓶颈顶上来——`__syncthreads()` 等待变成了代价。本质是 **"memory wait → sync wait"** 的交换，不是消除了瓶颈。Tile 确实减少了 global memory 访问，但把等待转移到了 block 内的同步点。

#### softmax: Sync-Bound

Barrier (6.72) 排第一——warp reduce 的串行尾巴（每轮 `__shfl_down_sync` 后必须等所有线程）成为瓶颈。优化方向: 减少 reduce 步数或提高 occupancy 让更多 warp 切换隐藏 barrier。

#### layernorm: Dependency Chain Bottleneck

Barrier (5.35) + Wait 揭示了 Welford 算法的本质问题——必须先算完 mean 才能算 variance，两次 warp reduce 是串行的，中间有 hard dependency。优化方向: 两趟 reduce 合并或改用 parallel reduction。

#### attention: Insufficient Parallelism

Wait (3.38) 排第一，但根本原因不是延迟——是 grid 太小。128 blocks 分给 28 SMs = 每个 SM 只能分到约 4.6 个 block（0.91 波），SM 无法靠多 block 切换来隐藏 latency。真实推理中 S≥2048 时 blocks 数量会增长，并行度自然回升。

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
├── cuda/                   # CUDA 后端 (Phase 1+2+3)
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
│   ├── fragmentation.cu    # Phase 3: KV Cache 碎片分析
│   ├── paged_attention.cu  # Phase 3: PagedAttention 实现
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
| v0.6 | PagedAttention + Continuous Batching (Phase 3 完成) | ✓ |
| v0.7 | Benchmark 数据采集 (Phase 4 Week 1) | ~ |

## 构建

### CPU (CMake)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# 47 个单元测试
./cpu/minitensor

# 基础算子 benchmark
./cpu/minitensor_cpu_benchmark

# Prefill vs Decode 延迟对比
./cpu/minitensor_cpu_inference_benchmark
```

### CUDA (仅 WSL2, 需要 nvcc + GPU)

```bash
cd tests/cuda && mkdir -p build && cd build
cmake .. && make -j$(nproc) cuda_kernel_benchmark
./benchmark/cuda_kernel_benchmark
```

Benchmark 结果详见 [docs/benchmarks.md](docs/benchmarks.md)

## 技术栈

- C++17, CUDA C++
- 计算图: 拓扑排序 (Kahn)、DAG 前向执行
- 算子: matmul, softmax, layernorm, attention, ReLU, broadcast add/mul, transpose
- CUDA: Grid/Block/Thread, Shared Memory, Warp Reduce
