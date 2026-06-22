# MiniTensor Benchmarks

> Phase 4 Week 1: 面试用性能数据 — CPU 算子 + CUDA kernel + Prefill/Decode 延迟对比

## CPU — 基础算子

| Op | Shape | Latency (us) | Bandwidth (GB/s) | Bottleneck |
| --- | --- | ---: | ---: | --- |
| add | [1024×1024] | 789.59 | 15.94 | memory bandwidth |
| mul | [1024×1024] | 726.16 | 17.33 | memory bandwidth |
| matmul | [256×256]×[256×256] | 21013.39 | 0.04 | compute / cache locality |
| matmul_blocked | [256×256]×[256×256], tile=32 | 3930.51 | 0.20 | compute / cache reuse |
| softmax | [1024×1024] | 6552.10 | 1.28 | exp latency + row reductions |
| layernorm | [1024×1024] | 3423.47 | 2.45 | row reductions + sqrt |
| transpose | [256×512] → [512×256] | 4147.53 | 0.25 | strided writes |
| reshape | [1024×1024] → [512×2048] | 548.68 | 15.29 | full copy |

**面试要点**: 算子分两类 —
- **memory-bound**: add/mul/reshape (15-17 GB/s, 瓶颈在内存带宽)
- **compute-bound**: matmul (0.04 GB/s, 瓶颈在计算), blocked matmul 5× 加速证明 cache reuse 的价值

## CPU — Prefill vs Decode 延迟对比

### Prefill (Full Attention Q[S×D] @ K[S×D]^T)

| Seq Len (S) | Head Dim (D) | Latency (us) | GFLOPS | Bottleneck |
| ---: | ---: | ---: | ---: | --- |
| 128 | 64 | 2138.06 | 2.00 | O(S²·D) compute-bound (Q@K^T dominates) |
| 256 | 64 | 9338.76 | 1.83 | O(S²·D) compute-bound |
| 512 | 64 | 38142.92 | 1.79 | O(S²·D) compute-bound |
| 1024 | 64 | 154078.21 | 1.78 | O(S²·D) compute-bound |

### Decode (Single Token Q[D] @ K_cache[S×D]^T)

| Cache Len (S) | Head Dim (D) | Latency (us) | Bandwidth (GB/s) | Bottleneck |
| ---: | ---: | ---: | ---: | --- |
| 128 | 64 | 16.56 | 3.99 | O(S·D) memory-bound (K/V cache load dominates) |
| 256 | 64 | 36.04 | 3.65 | O(S·D) memory-bound |
| 512 | 64 | 72.13 | 3.64 | O(S·D) memory-bound |
| 1024 | 64 | 145.32 | 3.61 | O(S·D) memory-bound |

### 交叉对比

| Seq Len (S) | Prefill (us) | Decode (us) | Ratio |
| ---: | ---: | ---: | ---: |
| 128 | 2138 | 17 | 129× |
| 256 | 9339 | 36 | 259× |
| 512 | 38143 | 72 | 529× |
| 1024 | 154078 | 145 | 1060× |

**面试要点**: 
- Prefill latency 随 S **平方增长** (O(S²·D)): S 翻倍 → 延迟 4×
- Decode latency 随 S **线性增长** (O(S·D)): S 翻倍 → 延迟 2×  
- S=1024 时 prefill 比 decode 慢 1000+ 倍 → 为什么 prefill chunking 和 prefill/decode 混合调度很重要
- Decode 是 memory-bound (3-4 GB/s bandwidth)，KV cache 越大越慢 → PagedAttention 的价值

## CUDA — Kernel 性能对比

> 需要在 WSL2 (RTX 2060, CC 7.5) 上运行。结果待采集。

| Op | Shape | Config | Latency (us) | Bandwidth (GB/s) | GFLOPS | Bottleneck |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vector_add | [16M] | 256 thr × 65536 blk | pending | pending | pending | memory bandwidth |
| matmul_naive | [1024³] | 16×16 thr × 64×64 blk | pending | pending | pending | global memory traffic |
| matmul_tiled | [1024³] | 16×16 thr × 64×64 blk, tile=16 | pending | pending | pending | shared memory reuse + occupancy |
| softmax | [1024×1024] | 1024 thr × 1024 blk | pending | pending | pending | memory bandwidth |
| layernorm | [1024×1024] | 1024 thr × 1024 blk | pending | pending | pending | memory bandwidth |
| attention | [S=128,D=64] | 256 thr × 128 blk, smem≈17KB | pending | pending | pending | QK reduction + softmax |

**面试要点**: 
- matmul_naive vs matmul_tiled: tiling 通过 shared memory 复用减少 global memory 访问，预期 5-10× 加速
- matmul_tiled GFLOPS 远低于 RTX 2060 理论峰值 (~6.5 TFLOPS FP32): 原因 = 无 Tensor Core、tile=16 太保守、occupancy 低
- softmax/layernorm 是 memory-bound: compute 简单(reduction)，瓶颈在数据搬运
- attention fused kernel 省去中间 tensor 写回 HBM（online softmax），比三阶段分开写快

## Build And Run

### CPU Benchmarks

```bash
cd /root/projects/minitensor
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 基础算子
./cpu/minitensor_cpu_benchmark

# Prefill vs Decode
./cpu/minitensor_cpu_inference_benchmark
```

### CUDA Benchmarks (WSL2 only)

```bash
# On WSL2 (潇达笔记本):
cd ~/minitensor/tests/cuda
mkdir -p build && cd build
cmake ..
make -j$(nproc) cuda_kernel_benchmark
./benchmark/cuda_kernel_benchmark

# Copy output back to docs/benchmarks.md
```
