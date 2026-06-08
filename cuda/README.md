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
| Week 4 | 性能分析 | 分析报告 | ncu/nsys profiling, roofline analysis |

## 面试考点覆盖

- GEMM 优化：Tiling, Shared Memory, 内存合并访问, occupancy 调优
- Softmax/LayerNorm：Warp-level reduction, 数值稳定技巧
- Kernel launch overhead vs grid-stride loop 的权衡

## 环境要求

- NVIDIA GPU (Compute Capability ≥ 7.0, 即 Volta 或更新)
- CUDA Toolkit ≥ 11.0
- nvidia-smi + nvcc 可用
- 推荐：nsight-compute (ncu) / nsight-systems (nsys)