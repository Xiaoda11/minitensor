#pragma once
#include <cuda_runtime.h>

// ---- WelfordState for layernorm_welford_kernel shared memory ----
struct WelfordState {
    int count;
    float mean;
    float m2;
};

// ---- Phase 1 Kernels ----
__global__ void vector_add_kernel_grid_stride(const float *a, const float *b,
                                               float *c, int n);

// ---- Matmul Kernels ----
__global__ void matmul_naive_kernel(const float *a, const float *b, float *c,
                                     int M, int K, int N);
__global__ void matmul_tiled_kernel(const float *a, const float *b, float *c,
                                     int M, int K, int N);

// ---- Softmax Kernels ----
__global__ void softmax_warp_reduce_kernel(const float *input, float *output,
                                            int rows, int cols);

// ---- LayerNorm Kernels ----
__global__ void layernorm_welford_kernel(const float *input, float *output,
                                          int rows, int cols, float eps);

// ---- Attention Kernels ----
__global__ void attention_fused_kernel_v2(const float *Q, const float *K,
                                           const float *V, float *O,
                                           int seq_len, int head_dim);
