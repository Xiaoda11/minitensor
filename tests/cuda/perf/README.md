# minitensor-perf — CUDA Kernel Performance Profiling & Roofline Analysis

Automated performance analysis for CUDA kernels built on Nsight Compute CLI (`ncu`).

## Quick Start

```bash
# Profile specific kernels
python3 main.py --bin ../build_cuda/benchmark/cuda_kernel_benchmark \
    --kernels matmul_tiled_kernel,softmax_warp_reduce_kernel

# Auto-discover all kernels in a binary
python3 main.py --bin ../build_cuda/benchmark/cuda_kernel_benchmark --auto

# Full analysis: profile + roofline plot + JSON export
python3 main.py --bin ../build_cuda/benchmark/cuda_kernel_benchmark --auto \
    --sudo --roofline roofline.png --output report.json

# List kernels without profiling
python3 main.py --bin ../build_cuda/benchmark/cuda_kernel_benchmark --list-kernels
```

**WSL2 users**: Always add `--sudo` (GPU perf counters require elevated access).

## Features

| Feature | Description |
|---------|-------------|
| **ncu profiling** | Collects 7 key GPU metrics per kernel via Nsight Compute |
| **kernel discovery** | Auto-finds all `__global__` functions via `cuobjdump` |
| **bottleneck classification** | Rule-based: compute/memory/latency/sync/balanced |
| **FLOP estimation** | Per-workload FLOP formulas (matmul, softmax, attention, layernorm) |
| **arithmetic intensity** | AI = estimated FLOPs / actual DRAM bytes |
| **workload grouping** | Groups by type: matmul, softmax, attention, layernorm, etc. |
| **roofline plot** | Log-log roofline with memory bandwidth + compute ceilings |
| **JSON export** | Machine-readable output for CI/regression detection |

## Architecture

```
tests/cuda/perf/
├── main.py              # CLI entry point
├── core/
│   ├── types.py         # KernelProfile dataclass + GPU specs database
│   ├── classifier.py    # Rule-based bottleneck classification
│   ├── ncu_runner.py    # ncu CLI wrapper + kernel discovery + batch runner
│   ├── analyzer.py      # Workload grouping + FLOP/AI computation
│   └── runner.py        # CPU timing utility
├── viz/
│   └── roofline.py      # Roofline plot (matplotlib, log-log)
└── README.md
```

## Collected Metrics

| Metric | ncu Counter | Use |
|--------|------------|-----|
| SM throughput % | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM busyness (incl. load/store) |
| DRAM read (MB) | `dram__bytes_read.sum` | Actual DRAM traffic |
| DRAM write (MB) | `dram__bytes_write.sum` | Actual DRAM traffic |
| Active warps % | `smsp__warps_active.avg.pct_of_peak_sustained_elapsed` | Occupancy proxy |
| Barrier stall ratio | `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | `__syncthreads()` bottleneck |
| Scoreboard stall ratio | `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | Memory latency bottleneck |
| L1/TEX throughput % | `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1 cache efficiency |

## Bottleneck Classification Rules

| Category | Condition | Meaning |
|----------|-----------|---------|
| `compute_bound` | SM util > 80% AND DRAM read < 200 MB | ALU is the bottleneck |
| `memory_bound` | DRAM read > 200 MB | HBM bandwidth limited |
| `latency_bound` | scoreboard stall > 5% | Waiting for global memory |
| `sync_bound` | barrier stall > 3% | `__syncthreads()` overhead |
| `balanced` | none of the above | No single bottleneck dominates |

## FLOP Estimation Formulas

| Workload | Formula | Example (default shape) |
|----------|---------|------------------------|
| matmul | 2 × M × K × N | 2 × 1024³ ≈ 2.15G FLOPs |
| attention | B × (4S²D + 8S²) | 1 × (4×128²×64 + 8×128²) ≈ 4.3M FLOPs |
| softmax | 8 × R × C | 8 × 1024² ≈ 8.4M FLOPs |
| layernorm | 8 × R × C | 8 × 1024² ≈ 8.4M FLOPs |
| vector_add | N | 16M FLOPs |

## GPU Specs Database

Pre-configured specs for roofline ceiling lines:

| GPU | FP32 TFLOPS | HBM Bandwidth | Ridge (FLOP/byte) |
|-----|------------|---------------|-------------------|
| RTX 2060 | 6.5 | 336 GB/s | 19.3 |
| RTX 3060 | 12.7 | 360 GB/s | 35.3 |
| RTX 4090 | 82.6 | 1008 GB/s | 82.0 |
| A100 | 19.5 | 1555 GB/s | 12.5 |

Auto-detection via `nvidia-smi`. Override with `--gpu "RTX 2060"`.

## Dependencies

- **Required**: Python 3.8+, CUDA Toolkit (provides `ncu`, `cuobjdump`)
- **Optional**: matplotlib (for `--roofline` plots)

## Example Output

```
GPU: RTX 3060 (12.7 TFLOPS FP32, 360 GB/s HBM)

Profiling 6 kernel(s) ...

[1/6] profiling matmul_naive_kernel ...
[2/6] profiling matmul_tiled_kernel ...
...

======================================================================
  PER-KERNEL RESULTS
======================================================================
Kernel                              SM%   DRAM(MB)       AI  Tag
-------------------------------------------------------------------
matmul_naive_kernel                 45.3%    7156.2   1024.0  compute_bound
matmul_tiled_kernel                 96.4%    1015.3   1024.0  memory_bound
softmax_warp_reduce_kernel          38.2%     192.0      0.4  memory_bound
attention_fused_kernel              12.1%       2.8    128.0  latency_bound
layernorm_welford_kernel            11.0%      48.0      0.3  memory_bound
vector_add_kernel                   92.0%     192.0      0.1  memory_bound

======================================================================
  GROUP SUMMARY
======================================================================
Group           Count  Avg SM%   Avg AI  Tags
-------------------------------------------------------------------
attention           1     12.1%    128.0  latency_bound(1)
layernorm           1     11.0%      0.3  memory_bound(1)
matmul              2     70.9%   1024.0  compute_bound(1), memory_bound(1)
softmax             1     38.2%      0.4  memory_bound(1)
vector_add          1     92.0%      0.1  memory_bound(1)
```

## Integration with minitensor

This tool sits alongside the benchmark binary under `tests/cuda/`:

```
tests/cuda/
├── benchmark/         # Benchmark binary (CMake + benchmark.cu + kernels.h)
│   ├── CMakeLists.txt
│   ├── benchmark.cu
│   └── kernels.h
└── perf/              # This tool
    ├── main.py
    ├── core/
    └── viz/
```

Build the benchmark, then run perf from this directory. The `--bin` path is relative to where you invoke `main.py`.
