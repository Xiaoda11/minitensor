# minitensor/cuda — Phase 1: CUDA Kernel 重写

minitensor 的 GPU 后端实现。目标是理解 CUDA 编程模型并重写 v0.3 的核心算子。

## 目录结构

```
cuda/
├── CMakeLists.txt        # CMake 构建配置
├── README.md             # 本文件
├── vector_add.cu         # Week 1: Grid/Block/Thread 入门
├── matmul_naive.cu       # Week 1 (续): 朴素矩阵乘法
├── matmul_tiled.cu       # Week 2: Tiling + Shared Memory
├── softmax.cu            # Week 3: Softmax (Warp Reduce)
├── layernorm.cu          # Week 3: LayerNorm CUDA
├── attention.cu          # Week 4: Fused Attention (QK^T + Softmax + PV)
├── kv_cache.h            # Phase 2 Day 2: KV Cache 数据结构
├── phase2_main.cu        # Phase 2 Day 2: KV Cache 测试
├── prefill.cu            # Phase 2 Day 3: Prefill 阶段
├── decode.cu             # Phase 2 Day 4: Decode 阶段
├── generate.cu           # Phase 2 Day 5: generate() 自回归循环
├── fragmentation.cu      # Phase 3 Day 1: KV Cache 碎片分析
├── paged_attention.cu    # Phase 3 Day 2: PagedAttention 实现
└── tests/                # 单元测试 (future)
```

## 构建

```bash
# 单文件编译（快速验证）
nvcc -O2 vector_add.cu -o vector_add && ./vector_add

# CMake 构建（benchmark binary，用于 profiling）
cd tests/cuda && mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75"
make -j$(nproc)
# binary: ./benchmark/cuda_kernel_benchmark
```

## GPU Profiling — Nsight Compute 三刀流

> 方法论：先看 SM% 判断有没有问题，再看 Top 1 Stall 判断问题在哪，最后查表找药方。

### 第一刀：SpeedOfLight（10 秒）

```bash
sudo $(which ncu) --kernel-name <kernel名> --launch-count 1 \
  --section SpeedOfLight ./tests/cuda/build/benchmark/cuda_kernel_benchmark
```

看 `Compute (SM) Throughput` 和 `DRAM Throughput`。

### 第二刀：Top 3 Stalls（30 秒）

```bash
sudo $(which ncu) --kernel-name <kernel名> --launch-count 1 \
  --metrics sm__throughput...,smsp__warps_active...,\
smsp__average_warps_issue_stalled_long_scoreboard...,\
smsp__average_warps_issue_stalled_short_scoreboard...,\
smsp__average_warps_issue_stalled_wait...,\
smsp__average_warps_issue_stalled_barrier...,\
smsp__average_warps_issue_stalled_math_pipe_throttle...,\
smsp__average_warps_issue_stalled_not_selected...,\
smsp__average_warps_issue_stalled_membar...,\
smsp__average_warps_issue_stalled_dispatch_stall...,\
smsp__average_warps_issue_stalled_other... \
  binary
```

排序最大的 3 个。

### 第三刀：分类 + 药方

| Top 1 Stall | 瓶颈 | 药方 |
|-------------|------|------|
| Long Scoreboard | 等全局内存 | tiling / shared memory / 合并访存 |
| Barrier | 等同步 | 减少 reduce 串行步骤 / 提 occupancy |
| Short Scoreboard | 等 shared mem/L1 | 解决 bank conflict / float4 宽 load |
| Wait | SFU 延迟 | 砍依赖链 / 减少 exp/div/sqrt |
| Not Selected | occupancy 不够 | 减寄存器/SMEM 用量 |
| Math Pipe Throttle | ALU 饱和 | 考虑混合精度 |

### 已 Profile 的 Kernel（RTX 2060, CC 7.5）

| Kernel | SM% | Warp% | Top 1 Stall | 类型 |
|--------|-----|-------|-------------|------|
| matmul_naive | 97.46 | 97.88 | Long Scoreboard 10.92 | 内存瓶颈 |
| matmul_tiled | 96.41 | 97.84 | Long Scoreboard 8.12 | 内存换同步 |
| softmax_warp_reduce | 41.78 | 58.57 | Barrier 6.72 | 同步瓶颈 |
| layernorm_welford | 48.06 | 63.85 | Barrier 5.35 | 依赖链 |
| attention_fused_v2 | 59.70 | 70.78 | Wait 3.38 | 并行度不足 |

### 一键脚本

```bash
# 单 kernel 完整 profiling
scripts/profile_kernel.sh matmul_tiled_kernel
```

## 学习路线 (4 周)

| 周次 | 主题 | 文件 | 关键概念 |
|------|------|------|----------|
| Week 1 | CUDA 编程模型 | `vector_add.cu`, `matmul_naive.cu` | Grid/Block/Thread, 内存层次, Warp |
| Week 2 | GEMM 优化 | `matmul_tiled.cu` | Tiling, Shared Memory, Bank Conflict, 循环展开 |
| Week 3 | Softmax/LayerNorm | `softmax.cu`, `layernorm.cu` | Warp Reduce, 数值稳定性 |
| Week 4 | Fused Attention | `attention.cu` | Kernel Fusion, Online Softmax, QK^T+Softmax+PV |

## 面试考点覆盖

- GEMM 优化：Tiling, Shared Memory, 内存合并访问, occupancy 调优
- Softmax/LayerNorm：Warp-level reduction, 数值稳定技巧
- Fused Attention：Kernel fusion, Online softmax, QK^T+Softmax+PV 融合
- Kernel launch overhead vs grid-stride loop 的权衡

## 环境要求

- NVIDIA GPU (Compute Capability ≥ 7.0, 即 Volta 或更新)
- CUDA Toolkit ≥ 11.0
- nvidia-smi + nvcc 可用
- 推荐：nsight-compute (ncu) / nsight-systems (nsys)
- 自动化 profiling 工具：`tests/cuda/perf/` (Python, 封装 ncu + 瓶颈分类 + roofline)

## Phase 2: 微型推理引擎 (v0.5)

| 天次 | 主题 | 文件 | 关键概念 |
|------|------|------|----------|
| Day 1 | 自回归生成流程 | — | Prefill vs Decode, 计算量对比 |
| Day 2 | KV Cache 数据结构 | `kv_cache.h`, `phase2_main.cu` | 静态池, (layer,head,pos) 偏移, memcpy |
| Day 3 | Prefill 阶段 | `prefill.cu` | 缓存所有 K/V, [S×S] attention |
| Day 4 | Decode 阶段 | `decode.cu` | 单 token Q 对全量 K/V, [1×S] attention |
| Day 5 | generate() 循环 | `generate.cu` | Prefill+Decode 串联, 自回归生成 |
| Week 2 | 显存分析 + CUDA 集成 | — | KV Cache 显存占比, GQA/MQA, CUDA kernel 集成 |
| Week 3-4 | PagedAttention + Batching | — | 分块分配, Block Table, Continuous Batching |

## Phase 3: PagedAttention + Continuous Batching (进行中)

| 天次 | 主题 | 文件 | 关键概念 |
|------|------|------|----------|
| Day 1 | KV Cache 碎片分析 | `fragmentation.cu` | 连续 vs 分页浪费对比 (75%→1.6%) |
| Day 2 | PagedAttention 实现 | `paged_attention.cu` | Block Table, 逻辑→物理翻译, 按需分配 | ✅ |
| Day 3 | PagedAttention Attention | TBD | 分页 attention 计算, block 遍历 | ← 当前 |
| Day 4 | Continuous Batching | TBD | 请求调度, 动态增删请求 |
| Day 5 | 端到端推理 | TBD | Prefill+Decode+PagedAttention 集成 |