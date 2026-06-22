/**
 * @file matmul_naive.cu
 * @brief CUDA Week 1 续 — 朴素矩阵乘法 C = A × B
 *
 * Learning objectives:
 *  - 2D Grid/Block 布局 (dim3)
 *  - 内存合并访问 (coalesced memory access)
 *  - 为什么全局内存重复读取导致 bottleneck
 *  - GPU vs CPU 性能对比，理解"朴素 GPU ≠ 一定快"
 *
 * Usage:
 *   nvcc -Wno-deprecated-gpu-targets matmul_naive.cu -o matmul_naive
 *   ./matmul_naive [M] [K] [N]
 *
 * 默认: 1024×1024 × 1024×1024
 *
 * 面试考点：
 *   Q: "为什么朴素 GPU matmul 有时候比 CPU 还慢？"
 *   A: 每个输出元素需要读 2*K 次全局内存（A 的一行 + B 的一列），
 *      而全局内存延迟 ~300-500 cycles。CPU 有 L1/L2/L3 cache
 *      自动缓存，GPU 需要手动用 shared memory 做 tiling。
 *      → Week 2: matmul_tiled.cu 解决这个问题。
 *
 * Reference:
 *   CUDA C++ Programming Guide §3.2.1, §5.4
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ============================================================================
// 1. Error checking
// ============================================================================

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t e = (err);                                                 \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// 2. CPU baseline — 标准三重循环 matmul
// ============================================================================

/// @brief CPU 矩阵乘法: C[M×N] = A[M×K] × B[K×N]
///
/// O(M*K*N) 标准实现。CPU 因为 Cache 自动工作，
/// 对中等尺寸矩阵（M,K,N ~ 1K）性能还不错。
static void matmul_cpu(const float *a, const float *b, float *c,
                int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// ============================================================================
// 3. GPU kernel: 朴素 matmul — 每个线程算输出矩阵的一个元素
// ============================================================================

/**
 * @brief GPU 朴素矩阵乘法
 *
 * 二维线程布局:
 *   - 每个线程负责输出矩阵的 1 个元素 C[row][col]
 *   - Grid: (N/16, M/16) 个 block
 *   - Block: (16, 16) 个线程
 *   - row = blockIdx.y * blockDim.y + threadIdx.y
 *   - col = blockIdx.x * blockDim.x + threadIdx.x
 *
 * 为什么慢？（明天 Week 2 tiling 要解决的核心问题）
 *   A[row][k] 访问模式: 同一行内 k 变化 → 内存连续 → 合并访问 ✓
 *   B[k][col] 访问模式: 不同 k, 同一 col → 内存跳跃 N 个元素 → 非合并访问 ✗
 *
 *   每个线程读 2*K 次全局内存（延迟约 300 cycles/次）。
 *   以 K=1024 为例: 每个线程约 2048 次全局内存访问。
 *   RTX 3060 全局内存带宽 ~360 GB/s，但延迟瓶颈让带宽利用率可能只有 10-20%。
 */
__global__ void matmul_naive_kernel(const float *a, const float *b, float *c,
                                     int M, int K, int N) {
    // ---- 线程的全局坐标: 对应输出矩阵 C 的 (row, col) ----
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界保护
    if (row >= M || col >= N) return;

    // ---- 内积: C[row][col] = Σ A[row][k] * B[k][col] ----
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        // A[row][k]: 行主序，连续访问 → coalesced ✓
        // B[k][col]: 内存跨度 = N，跳跃访问 → uncoalesced ✗
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}

// ============================================================================
// 4. 验证函数
// ============================================================================

static bool verify(const float *cpu, const float *gpu, int n, float eps = 1e-2f) {
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(cpu[i] - gpu[i]) > eps) {
            if (mismatches < 5) {
                fprintf(stderr, "  Mismatch[%d]: CPU=%f, GPU=%f\n",
                        i, cpu[i], gpu[i]);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "  Total mismatches: %d / %d\n", mismatches, n);
        return false;
    }
    return true;
}

// ============================================================================
// 5. Main
// ============================================================================

#ifndef KERNEL_EXPORT
int main(int argc, char **argv) {
    // ---- 参数 ----
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 1024;

    int size_a = M * K;
    int size_b = K * N;
    int size_c = M * N;
    size_t bytes_a = size_a * sizeof(float);
    size_t bytes_b = size_b * sizeof(float);
    size_t bytes_c = size_c * sizeof(float);

    // ---- Device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("========================================\n");
    printf("CUDA Week 1: Naive Matrix Multiply\n");
    printf("========================================\n");
    printf("Dimensions:    A[%d×%d] × B[%d×%d] = C[%d×%d]\n", M, K, K, N, M, N);
    printf("Total FLOPs:   %.2f GFLOP\n", 2.0 * M * K * N / 1e9);
    printf("Memory:        A+B=%.1f MB  C=%.1f MB\n",
           (bytes_a + bytes_b) / 1e6, bytes_c / 1e6);
    printf("GPU:           %s (CC %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Tile size:     %d×%d threads per block\n\n", 16, 16);

    // ---- Host alloc & init ----
    float *h_a = new float[size_a];
    float *h_b = new float[size_b];
    float *h_c_cpu = new float[size_c];
    float *h_c_gpu = new float[size_c];

    for (int i = 0; i < size_a; ++i) h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < size_b; ++i) h_b[i] = static_cast<float>(rand()) / RAND_MAX;

    // ---- Device alloc ----
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

    // ---- CPU baseline ----
    printf("Running CPU baseline... ");
    fflush(stdout);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("%.2f ms (%.2f GFLOP/s)\n", cpu_ms, 2.0 * M * K * N / (cpu_ms * 1e6));

    // ---- GPU launch config ----
    // Block: 16×16 = 256 threads (warp = 32 threads — 一个 block = 8 warps)
    // Grid:  ceil(N/16) × ceil(M/16) blocks
    dim3 block_dim(16, 16);
    dim3 grid_dim((N + 15) / 16, (M + 15) / 16);

    printf("Launch config:  grid(%d,%d), block(%d,%d)\n",
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    printf("Running GPU kernel... ");
    fflush(stdout);

    // ---- GPU timing ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes_c, cudaMemcpyDeviceToHost));

    printf("%.2f ms (%.2f GFLOP/s)\n", gpu_ms, 2.0 * M * K * N / (gpu_ms * 1e6));

    // ---- Results ----
    printf("\n========== Results ==========\n");
    printf("CPU:              %.2f ms\n", cpu_ms);
    printf("GPU (naive):      %.2f ms", gpu_ms);
    if (gpu_ms < cpu_ms)
        printf("  (%.1fx speedup)\n", cpu_ms / gpu_ms);
    else
        printf("  (%.1fx SLOWER than CPU!)\n", gpu_ms / cpu_ms);
    printf("Verification:     %s\n",
           verify(h_c_cpu, h_c_gpu, size_c) ? "PASSED ✓" : "FAILED ✗");

    // ---- Analysis ----
    printf("\n========== Analysis ==========\n");
    if (gpu_ms > cpu_ms) {
        printf("GPU slower than CPU! Reason:\n");
        printf("  1. Each output element reads 2×%d global memory values\n", K);
        printf("  2. Global memory latency ~300-500 cycles per access\n");
        printf("  3. B[k][col] access is NOT coalesced (jumps %d elements per loop)\n\n", N);
        printf("  → Week 2: matmul_tiled.cu uses shared memory tiling to fix this\n");
    } else {
        printf("GPU faster! But still bottlenecked by global memory.\n");
        printf("  Each thread reads 2×%d global memory values.\n", K);
        printf("  With tiling (Week 2), we can reduce this to ~2×K/tile_size.\n\n");
        printf("  Expected >2-5x further speedup with shared memory tiling.\n");
    }

    printf("Next:  nvcc matmul_tiled.cu -o matmul_tiled && ./matmul_tiled\n");

    // ---- Cleanup ----
    delete[] h_a; delete[] h_b; delete[] h_c_cpu; delete[] h_c_gpu;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
#endif // KERNEL_EXPORT