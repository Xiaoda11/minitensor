# MiniTensor

> 一个从零实现的 C++ / CUDA 张量库与 LLM 推理系统学习项目，重点复现现代大模型推理引擎中的 **CUDA Kernel、KV Cache、PagedAttention、Prefill / Decode、Continuous Batching** 等核心机制。

MiniTensor 不是一个完整的深度学习框架，而是一个面向 **AI Infra / LLM Inference / CUDA Runtime** 方向的底层学习项目。

项目目标是：
从张量计算、CUDA 算子、KV Cache 到 PagedAttention 和调度机制，逐步理解现代 LLM 推理系统的性能瓶颈与工程设计。

---

## 项目动机

现代大模型推理性能不仅取决于模型结构，也高度依赖底层系统能力，例如：

* CUDA kernel 的实现效率
* 显存访问模式
* KV Cache 的组织方式
* Prefill / Decode 阶段差异
* PagedAttention 式显存管理
* Continuous Batching 调度策略
* Nsight Compute 性能分析能力

MiniTensor 是我从零复现这些机制的实践项目。
它更像是一个小型的 LLM inference runtime playground，用来理解 vLLM 等推理框架背后的核心思想。

---

## 项目亮点

| 模块          | 实现内容                                                                               |
| ----------- | ---------------------------------------------------------------------------------- |
| Tensor 基础库  | N 维 Tensor、shape / stride、broadcast、基础算子                                           |
| 计算图         | DAG 构建、拓扑排序、前向执行                                                                   |
| CPU 算子      | add、mul、matmul、blocked matmul、softmax、layernorm、transpose                          |
| CUDA Kernel | vector add、naive matmul、tiled matmul、softmax warp reduce、layernorm、fused attention |
| 推理流程        | KV Cache、Prefill、Decode、Generate loop                                              |
| 显存管理        | KV Cache 碎片分析、PagedAttention-style block table                                     |
| 调度机制        | Continuous Batching 模拟                                                             |
| 性能分析        | CUDA benchmark 框架、Nsight Compute profiling                                         |

---

## 项目结构

```text
minitensor/
├── cpu/
│   ├── tensor.h              # N-D Tensor 模板、shape / stride、broadcast
│   ├── compute_graph.h       # DAG 计算图
│   ├── memory_pool.h         # 简单内存池
│   └── main.cpp              # CPU 测试与 demo
│
├── cuda/
│   ├── vector_add.cu         # CUDA 入门 kernel
│   ├── matmul_naive.cu       # naive global-memory matmul
│   ├── matmul_tiled.cu       # shared-memory tiled matmul
│   ├── softmax.cu            # warp-level softmax reduce
│   ├── layernorm.cu          # Welford-based layernorm
│   ├── attention.cu          # fused attention kernel
│   ├── kv_cache.h            # KV Cache 数据结构
│   ├── prefill.cu            # Prefill 阶段
│   ├── decode.cu             # Decode 阶段
│   ├── generate.cu           # 自回归生成流程
│   ├── fragmentation.cu      # KV Cache 碎片分析
│   └── paged_attention.cu    # PagedAttention-style block mapping
│
├── tests/cuda/               # CUDA benchmark 与 profiling 入口
├── docs/benchmarks.md        # benchmark 结果与 Nsight 分析
├── scripts/                  # 辅助脚本
└── CMakeLists.txt
```

---

## 性能测试环境

```text
GPU: NVIDIA RTX 3060
Compute Capability: 8.6
VRAM: 12GB
SM 数量: 28
Compiler: nvcc -O3
Profiler: NVIDIA Nsight Compute
CPU: Intel Xeon / C++17
```

---

## CUDA Kernel Benchmark

| Kernel       |             输入规模 |      延迟 | 主要观察                              |
| ------------ | ---------------: | ------: | --------------------------------- |
| vector_add   |     16M elements |  609 us | memory bandwidth bound            |
| matmul_naive |            1024³ | 2858 us | global memory 访问成为主要瓶颈            |
| matmul_tiled |   1024³, tile=16 | 2116 us | shared memory 提升数据复用，但仍受访存影响      |
| softmax      |        1024×1024 |   59 us | row-wise reduction，偏 memory-bound |
| layernorm    |        1024×1024 |  208 us | Welford + shuffle reduce 有额外开销    |
| attention    | B=1, S=128, D=64 |   46 us | 小规模 attention，latency-bound 明显    |

详细数据见：[`docs/benchmarks.md`](docs/benchmarks.md)

---

## Prefill vs Decode 实验

| Seq Len | Prefill Latency | Decode Latency | Ratio |
| ------: | --------------: | -------------: | ----: |
|     128 |         2138 us |          17 us |  129× |
|     256 |         9339 us |          36 us |  259× |
|     512 |        38143 us |          72 us |  529× |
|    1024 |       154078 us |         145 us | 1060× |

核心结论：

> Prefill 阶段需要对完整 prompt 做 attention，复杂度接近 `O(S²·D)`。
> Decode 阶段每次只处理一个新 token，复杂度接近 `O(S·D)`。
> 因此，随着序列长度增长，Prefill 和 Decode 的性能差距会快速扩大。

这也是为什么现代 LLM 推理系统需要：

* Prefill / Decode 分离
* Chunked Prefill
* Continuous Batching
* KV Cache 复用
* PagedAttention 显存管理

---

## Nsight Compute 分析

### 1. 高 SM 利用率不等于高计算效率

在 `matmul_tiled_kernel` 中，Nsight Compute 显示：

```text
sm__throughput                 ≈ 96.4%
smsp__warps_active             ≈ 97.9%
stalled_long_scoreboard        ≈ 8.09 inst/issue
stalled_barrier                ≈ 5.46 inst/issue
```

表面上看，SM 利用率很高，warp 活跃度也很高。
但从 stall 指标看，kernel 仍然有大量时间在等待内存加载。

因此，这个 kernel 的主要问题不是 occupancy 不够，而是：

```text
global memory latency 较高
shared memory tile 复用度不够
每个线程计算量偏少
register-level reuse 不足
```

---

### 2. Naive Matmul 的瓶颈在访存模式

naive matmul 中，A 的访问相对连续，但 B 的访问经常是按列方向读取，global memory 访问局部性较差。

虽然矩阵乘法理论上 arithmetic intensity 较高，但 naive 实现没有充分利用 shared memory 和 register tiling，因此实际性能远低于 GPU 峰值算力。

---

### 3. Tiled Matmul 有提升，但 16×16 tile 仍然不够

当前 tiled matmul 已经通过 shared memory 降低了部分 global memory 访问，但 tile size 较小，每个线程只负责一个 output element，计算复用度仍然有限。

后续优化方向：

```text
16×16 one-thread-one-output
→ 64×64 block tile
→ register tiling
→ vectorized load
→ double buffering
→ Tensor Core path
```

---

## 构建与运行

### CPU 版本

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

./cpu/minitensor
./cpu/minitensor_cpu_benchmark
./cpu/minitensor_cpu_inference_benchmark
```

### CUDA 版本

```bash
cd tests/cuda
mkdir -p build
cd build
cmake ..
make -j$(nproc) cuda_kernel_benchmark

./benchmark/cuda_kernel_benchmark
```

---

## Roadmap

| 版本   | 内容                                    | 状态          |
| ---- | ------------------------------------- | ----------- |
| v0.1 | Tensor 基础：Rule of 5、模板 Tensor         | Done        |
| v0.2 | Shape、Stride、Broadcast、基础算子           | Done        |
| v0.3 | Compute Graph、两层 MLP 推理 demo          | Done        |
| v0.4 | CUDA Kernel 实现                        | Done        |
| v0.5 | KV Cache、Prefill / Decode / Generate  | Done        |
| v0.6 | PagedAttention、Continuous Batching    | Done        |
| v0.7 | Benchmark 框架、Nsight Profiling         | In Progress |
| v0.8 | Register-tiled Matmul、Vectorized Load | Planned     |
| v0.9 | 更大序列长度 Attention Benchmark            | Planned     |
| v1.0 | Minimal LLM Inference Runtime Demo    | Planned     |

---

## 项目收获

通过 MiniTensor，我重点理解了以下 AI Infra / LLM Inference 问题：

* 为什么 Decode 阶段容易 memory-bound
* KV Cache 为什么会成为推理系统的核心资源
* PagedAttention 如何减少显存碎片
* Prefill / Decode 为什么需要分离调度
* Continuous Batching 如何提升吞吐
* 高 occupancy 为什么不一定代表高 FLOPS
* Nsight Compute 如何定位 memory stall、barrier stall、warp active 等问题
* CUDA kernel 中 shared memory、coalescing、bank conflict、warp reduce 的实际影响

---

## 相关方向

* CUDA Kernel Optimization
* LLM Inference Runtime
* KV Cache
* PagedAttention
* Continuous Batching
* Prefill / Decode Separation
* Memory-bound vs Compute-bound Analysis
* Nsight Compute Profiling

---

## 作者

作者：Xiaoda / Jace Lee

项目方向：

```text
LLM Inference · CUDA Kernels · Runtime Systems · AI Infrastructure
```

该项目是我面向 AI Infra / LLM 推理优化方向的底层系统实践。

🌐 个人主页：[xiaoda.cloud](https://xiaoda.cloud)
