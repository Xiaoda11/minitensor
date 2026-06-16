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
└── tests/                # 单元测试 (future)
```

## 构建

```bash
# 单文件编译（快速验证）
nvcc -O2 vector_add.cu -o vector_add && ./vector_add

# CMake 构建（多文件）
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75"   # 根据你的 GPU 调整
make -j$(nproc)
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