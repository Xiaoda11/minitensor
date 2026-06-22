# MiniTensor Benchmarks

CPU results were measured on this server with `std::chrono` via `minitensor_cpu_benchmark`.
CUDA rows are placeholders for WSL2 runs.

## CPU

| Device | Op | Shape | Latency (us) | Bandwidth (GB/s) | Bottleneck |
| --- | --- | --- | ---: | ---: | --- |
| CPU | add | [1024x1024] | 732.26 | 17.18 | memory bandwidth |
| CPU | mul | [1024x1024] | 696.86 | 18.06 | memory bandwidth |
| CPU | matmul | [256x256]x[256x256] | 21261.91 | 0.04 | compute / cache locality |
| CPU | matmul_blocked | [256x256]x[256x256], tile=32 | 3931.86 | 0.20 | compute / cache reuse |
| CPU | softmax | [1024x1024] | 6517.11 | 1.29 | exp latency + row reductions |
| CPU | layernorm | [1024x1024] | 3387.50 | 2.48 | row reductions + sqrt |
| CPU | transpose | [256x512] -> [512x256] | 4180.60 | 0.25 | strided writes |
| CPU | reshape | [1024x1024] -> [512x2048] | 548.49 | 15.29 | full copy |

## CUDA

| Device | Op | Shape | Latency (us) | Bandwidth (GB/s) | Bottleneck |
| --- | --- | --- | ---: | ---: | --- |
| CUDA | vector_add | [16M] | pending | pending | memory bandwidth |
| CUDA | matmul_naive | [1024x1024]x[1024x1024] | pending | pending | global memory traffic / no tiling |
| CUDA | matmul_tiled | [1024x1024]x[1024x1024] | pending | pending | shared memory reuse / occupancy |
| CUDA | softmax | [1024x1024] | pending | pending | row reductions + exp |
| CUDA | layernorm | [1024x1024] | pending | pending | row reductions + sqrt |
| CUDA | attention | [B=1,H=1,S=128,D=64] | pending | pending | QK reduction + softmax + V accumulation |

## Build And Run

CPU:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
./cpu/minitensor_cpu_benchmark
```

CUDA benchmark skeleton:

```bash
cmake --build build --target cuda_kernel_benchmark
./tests/cuda/benchmark/cuda_kernel_benchmark
```
