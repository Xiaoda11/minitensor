# MiniTensor Benchmarks

> Phase 4 Week 1: 面试用性能数据 — CPU 算子 + CUDA kernel + Nsight Profiling + Prefill/Decode 延迟对比
>
> 采集环境: RTX 3060 (CC 8.6, 12GB VRAM, 28 SMs) / 腾讯云服务器 (Intel Xeon, g++)

---

## CUDA — Kernel 性能 (RTX 3060, nvcc -O3)

| Op | Shape | Config | Latency (us) | Bandwidth (GB/s) | GFLOPS | vs Peak FP32 | Bottleneck |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| vector_add | [16M] | 256 thr × 65536 blk | 609 | 330.85 | 27.57 | — | memory bandwidth (92% of 360 GB/s HBM) |
| matmul_naive | [1024³] | 16×16 thr × 64×64 blk | 2858 | 4.40 | 751 | 5.9% | global memory traffic (no tiling) |
| matmul_tiled | [1024³] | 16×16 thr × 64×64 blk, tile=16 | 2116 | 5.95 | 1015 | 8.0% | **memory latency** (见 Nsight 分析) |
| softmax | [1024×1024] | 1024 thr × 1024 blk, smem=1 warps×4B | 59 | 142.87 | 142.87 | — | memory bandwidth (3-pass row reduction) |
| layernorm | [1024×1024] | 1024 thr × 1024 blk, smem=1 warps×Welford | 208 | 40.33 | 40.33 | — | memory bandwidth + Welford 3-field shuffle |
| attention | [B=1,S=128,D=64] | 256 thr × 128 blk, smem≈17KB | 46 | 2.83 | 93.31 | — | latency-bound (128KB total, launch overhead) |

### Roofline 分析

RTX 3060 理论峰值: ~12.7 TFLOPS FP32, ~360 GB/s HBM bandwidth. Ridge point ≈ 35 FLOP/byte.

| Kernel | AI (FLOP/byte) | 带宽利用率 | 算力利用率 | Roofline 位置 |
|--------|---------------|-----------|-----------|-------------|
| vector_add | 0.08 | 92% | 0.2% | 极左 → memory-bound ✓ |
| matmul_naive | ~1024 | 1.2% | 5.9% | 极右 → 理论 compute-bound，实际 memory-latency 吃掉 |
| matmul_tiled | ~1024 | 1.7% | 8.0% | 极右 → 同上，tile 太小没缓解 latency |
| softmax | 0.42 | 40% | 1.1% | 左侧 → memory-bound |
| layernorm | 0.25 | 11% | 0.3% | 左侧 → memory-bound, Welford 3× shuffle |
| attention | ~128 | 0.8% | 0.7% | latency-bound (S=128 太小) |

### 面试三句话模板

**vector_add**: 3 数组 streaming，memory-bound，92% HBM 带宽利用率，无需优化。

**matmul_naive → tiled**: tile=16 shared memory 复用，bandwidth 4.4→6.0 GB/s (+35%)，GFLOPS 751→1015 (+35%)。提升远低于预期的 16× 因为 **DRAM read 实际 271 MB vs 理想 8 MB = 34×**——L2 cache 对 1024³ 无效，每次 tile load 都是 fresh DRAM fetch。

**softmax vs layernorm**: 同 shape [1024×1024] memory-bound，但 layernorm 慢 3.5×。根因: WelfordState 12 字节 vs softmax 的 4 字节，warp shuffle 开销 3×。

**attention**: S=128 太小 (128KB total)，kernel launch overhead ~5-10us 占 46us 大头。真实推理 S≥2048 时 bandwidth 会回升。

---

## Nsight Compute — matmul_tiled_kernel 深度 Profiling

> 工具: NVIDIA Nsight Compute 2026.1.1, RTX 3060 (CC 8.6)
> Kernel: `matmul_tiled_kernel<<<(64,64), (16,16)>>>(d_A, d_B, d_C, 1024, 1024, 1024)`

### 关键 Metric (多次 run 均值)

| Metric | Value | 含义 |
|--------|-------|------|
| `dram__bytes_read.sum` | **271 MB** | 实际从 HBM 读取量 |
| `dram__bytes_write.sum` | **5.1 MB** | 实际写回 HBM 量 |
| `sm__throughput (% peak)` | **96.4%** | SM 指令发射活跃度 |
| `l1tex__throughput (% peak)` | **96.4%** | L1/Texture Cache 吞吐活跃度 |
| `smsp__warps_active (% peak)` | **97.9%** | Warp 活跃度 (≈occupancy) |
| `stalled_barrier` | **5.46** inst/issue | 等 `__syncthreads()` 的周期 |
| `stalled_long_scoreboard` | **8.09** inst/issue | 等 global memory load 的周期 |

### 核心发现: 推翻 Occupancy 假设

**之前的错误判断**: "SM 利用率只有 8%，tile=16 太小 occupancy 低"\
**Nsight 实测**: SM throughput 96.4%, warp active 97.9% — **occupancy 根本不是瓶颈**

### DRAM Read 放大 34× 分析

```
理想 global traffic:  A(4MB) + B(4MB) + C(4MB write) = ~12 MB
实测 dram__bytes_read:  271 MB
实测 dram__bytes_write: 5.1 MB
读放大系数:             271 / 8 = 34×
写效率:                 5.1 / 4 = 1.28× (接近完美)
```

放大原因:
- 算法级 global load 请求总量 = 4096 blocks × 64 iterations × 2 tiles × 256 floats × 4B = **512 MB**
- 实测 DRAM read 271 MB → **L2 hit rate ≈ 1 - 271/512 ≈ 47%**
- tile=16 每 block 只复用 16×16=256 个元素，K=1024 需要 64 轮 — 每轮的 tile 数据在下轮前已被驱逐出 L2
- 跨 block 的 A/B tile 复用完全靠 L2 运气 — 同一行 A 被 64 个 block 读，只有 L2 能救，实测只救了约一半
- **本质**: L2 cache 对 1024³ working set 挡掉约一半，但仍有一半是 fresh DRAM fetch

### Scoreboard Stall > Barrier Stall

```
stalled_long_scoreboard:  8.09  inst/issue  ← 等 global memory load
stalled_barrier:          5.46  inst/issue  ← 等 __syncthreads()
```

Scoreboard 跟踪 in-flight 的 global memory load — 线程发出 `LDG` 后等数据回来。8.09 vs 5.46 = 内存延迟是 barrier 的 **1.5×**。

之前认为 `__syncthreads()` 是主因，实测颠覆：**主因是 global memory latency，不是同步开销**。

### SM throughput 96.4% vs GFLOPS 8% — 不矛盾

"SM throughput" 算的是 SM 在发指令的时间占比——**包括 load/store 指令**。你的 kernel 大部分时间在发 `LDG`，这些指令占满流水线（96.4%）但不贡献 FLOPS。所以:

> 96.4% 活跃 + 1 TFLOPS (8% 峰值) = warp 在疯狂发 load 指令等内存，FMA 排队闲着

### 优化方向 (按优先级)

| 优先级 | 方向 | 预期效果 | 理由 |
|--------|------|---------|------|
| P0 | **Block tile 16×16→64×64** + Register tiling | 3-5× | 不增加线程数 (保持 256)，每线程算 4×4 或 8×4 个 C 元素。一次 shared load 服务 64+ FMA，DRAM 读从 271MB→~20MB |
| P1 | **Double buffering** | 1.3-1.5× | 大 tile 前提下：加载下一 tile 的同时计算当前 tile，隐藏 latency |
| P2 | **Vectorized load (float4)** | 1.2-1.5× | 减少 load 指令数，提高 bandwidth 利用率 |
| P3 | Tensor Core (FP16) | 4× | CC 8.6 支持, 但需要改精度 |

**核心原则**: 不要围绕 occupancy 优化；要围绕**提高每次 global/shared load 对应的 FMA 数量**优化。

**不要 tile=32 一线程一元素**: 32×32 block 需要 1024 threads，CUDA 上限就是 1024，无扩展空间。正确方向:
```
当前 16×16 one-thread-one-C
→ 64×64 block tile, 256 threads, 每线程 4×4=16 个 C
→ register tiling (累加在寄存器)
→ double buffering
→ bank conflict / vectorized load
```

> 优化路线参考: Codex (OpenAI) 独立分析 + Nsight 实测数据

---

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

**面试要点**: 算子分两类 — memory-bound (add/mul/reshape, 15-17 GB/s) vs compute-bound (matmul, 0.04 GB/s)。blocked matmul 5× 加速证明 cache reuse 的价值。

---

## CPU — Prefill vs Decode 延迟对比

### Prefill (Full Attention Q[S×D] @ K[S×D]^T)

| Seq Len (S) | Head Dim (D) | Latency (us) | GFLOPS | Bottleneck |
| ---: | ---: | ---: | ---: | --- |
| 128 | 64 | 2138.06 | 2.00 | O(S²·D) compute-bound |
| 256 | 64 | 9338.76 | 1.83 | O(S²·D) compute-bound |
| 512 | 64 | 38142.92 | 1.79 | O(S²·D) compute-bound |
| 1024 | 64 | 154078.21 | 1.78 | O(S²·D) compute-bound |

### Decode (Single Token Q[D] @ K_cache[S×D]^T)

| Cache Len (S) | Head Dim (D) | Latency (us) | Bandwidth (GB/s) | Bottleneck |
| ---: | ---: | ---: | ---: | --- |
| 128 | 64 | 16.56 | 3.99 | O(S·D) memory-bound |
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

**面试要点**: Prefill O(S²·D) vs Decode O(S·D)。S=1024 时 prefill 比 decode 慢 1000+ 倍 → prefill chunking 和 prefill/decode 混合调度的价值。

---

## Build And Run

### CPU Benchmarks

```bash
cd ~/minitensor
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./cpu/minitensor_cpu_benchmark
./cpu/minitensor_cpu_inference_benchmark
```

### CUDA Benchmarks (WSL2 only)

```bash
cd ~/minitensor/tests/cuda
mkdir -p build && cd build
cmake ../benchmark
make -j$(nproc)
./benchmark/cuda_kernel_benchmark
```

### Nsight Compute Profiling (WSL2, requires perf counter access)

```bash
# 前置: NVIDIA Control Panel → Developer → Allow GPU perf counters for all users
#        + reg add HKLM\...\nvlddmkm\Global /v RmtSvcAllowAmpCdrv /d 1 /f + 重启

# 全量 profile
sudo $(which ncu) --set full ./benchmark/cuda_kernel_benchmark

# 聚焦 matmul_tiled 关键 metric
sudo $(which ncu) \
  --kernel-name matmul_tiled_kernel \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
smsp__warps_active.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
  ./benchmark/cuda_kernel_benchmark
```

### Automated Profiling with minitensor-perf

The manual ncu workflow above is automated by `tests/cuda/perf/` — a Python profiling
framework that wraps ncu, classifies bottlenecks, computes arithmetic intensity, and
generates roofline plots.

```bash
cd tests/cuda/perf

# Profile all kernels in the benchmark binary (auto-discovery)
python3 main.py --bin ../build_cuda/benchmark/cuda_kernel_benchmark --auto --sudo

# Full analysis: profile + roofline + JSON export
python3 main.py --bin ../build_cuda/benchmark/cuda_kernel_benchmark --auto \
    --sudo --roofline roofline.png --output report.json
```

See `tests/cuda/perf/README.md` for full documentation on bottleneck classification
rules, FLOP estimation formulas, and GPU specs database.
