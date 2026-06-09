/**
 * @file matmul_tiled.cu
 * @brief CUDA Week 2 — Tiled Matrix Multiply with Shared Memory
 *
 * Learning objectives:
 *  - Shared memory 声明和使用 (__shared__)
 *  - Tiling: 把大矩阵切成小块，逐块计算
 *  - __syncthreads() 线程同步屏障
 *  - 为什么 tiling 能把全局内存访问减少 16 倍
 *  - Bank conflict 的概念（本次不做优化，先理解概念）
 *
 * Usage:
 *   nvcc -O2 matmul_tiled.cu -o matmul_tiled
 *   ./matmul_tiled [M] [K] [N]
 *
 * 默认: 1024×1024 × 1024×1024
 *
 * Core idea（一句话）:
 *   把 A 和 B 的 16×16 小块搬到片上 shared memory（~20 cycles 延迟），
 *   在 shared memory 上做乘加，最后写回全局内存。
 *   全局内存访问次数从 2×K 降到 2×K/TILE_SIZE。
 *
 * Reference:
 *   CUDA C++ Programming Guide §3.2.3 (Shared Memory)
 *   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ============================================================================
// 0. Macros
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

// Tile size: 16×16 是经典选择
//   - 16×16 = 256 threads = 8 warps → 一个 block 刚好
//   - 16×16×2×float = 2 KB (A tile + B tile) → 远小于 48 KB shared memory
//   - 16 是 warp size (32) 的一半 → coalesced 访问友好
#define TILE_SIZE 16

// ============================================================================
// 1. CPU baseline — 同上一个文件，这里用 i-k-j 顺序（更快）
// ============================================================================

/// @brief CPU 矩阵乘法（i-k-j 循环顺序，cache 友好）
///
/// 与 naive 版本的 i-j-k 不同，这里把 k 提到中层：
///   for i:         遍历 A 的每一行
///     for k:        遍历累加维度
///       for j:      累加到 C[i][j]
///
/// 为什么更快：内层 j 循环中，C[i][j] 连续访问（行主序），
/// B[k][j] 也连续访问。A[i][k] 在内层不变，可以放寄存器。
/// CPU L1 cache 命中率大幅提升。
void matmul_cpu(const float *a, const float *b, float *c,
                int M, int K, int N) {
    // 先清零 C
    for (int i = 0; i < M * N; ++i) c[i] = 0.0f;

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float aik = a[i * K + k];          // A[i][k] 放寄存器，内层循环不变
            for (int j = 0; j < N; ++j) {
                c[i * N + j] += aik * b[k * N + j];
                //                 ^^^^^^^^^^^^^^^^  连续访问 ✓
                // ^^^^^^^^^^^^  连续访问 ✓
            }
        }
    }
}

// ============================================================================
// 2. GPU 朴素版本（用于对比）
// ============================================================================

__global__ void matmul_naive_kernel(const float *a, const float *b, float *c,
                                     int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}

// ============================================================================
// 3. GPU Tiled 版本 — 核心！
// ============================================================================

/**
 * @brief Tiled 矩阵乘法 kernel
 *
 * ============================================================
 * 执行流程（每个 block 负责一个 16×16 的 C tile）
 * ============================================================
 *
 *   for (int phase = 0; phase < ceil(K/TILE_SIZE); ++phase) {
 *
 *       // Step 1: 所有线程协作，把 A 和 B 各一个 tile 搬到 shared memory
 *       //         —— 每个线程搬 A 的 1 个元素 + B 的 1 个元素
 *       As[threadIdx.y][threadIdx.x] = A[row][phase*TILE + threadIdx.x]
 *       Bs[threadIdx.y][threadIdx.x] = B[phase*TILE + threadIdx.y][col]
 *
 *       // Step 2: 等所有线程搬完 —— 关键！
 *       __syncthreads();
 *
 *       // Step 3: 在 shared memory 上做 16 次乘加（延迟 ~20 cycles）
 *       for (int k = 0; k < TILE_SIZE; ++k)
 *           sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]
 *
 *       // Step 4: 等所有线程算完再搬下一块
 *       __syncthreads();
 *   }
 *
 *   // 写回全局内存
 *   C[row][col] = sum
 *
 * ============================================================
 * 为什么快 16 倍？
 * ============================================================
 *
 *   全局内存访问次数对比（K=1024, TILE=16）：
 *     Naive:  每个线程读 2×1024 = 2048 次全局内存
 *     Tiled:  每个线程读 2×1024/16 = 128 次全局内存
 *
 *   因为 shared memory 延迟 ~20 cycles vs 全局内存 ~300 cycles，
 *   而且同一个 tile 内的数据被 block 内 256 个线程共享复用。
 *
 * ============================================================
 * __syncthreads() 的作用
 * ============================================================
 *
 *   一个 block 内的线程不是同时执行的 —— SM 按 warp (32 线程) 调度。
 *   如果不加 __syncthreads()，可能 warp 0 还在搬数据，
 *   warp 7 已经在读了 —— 读到的是旧数据。
 *
 *   __syncthreads() = "所有线程到这里停下，等 block 内所有线程都到了再继续"
 */
__global__ void matmul_tiled_kernel(const float *a, const float *b, float *c,
                                     int M, int K, int N) {
    // ---- 全局坐标：这个线程负责 C[row][col] ----
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // ---- Shared memory: 整个 block 共享 ----
    //      __shared__ 关键字 = 存在片上 SRAM，整个 block 可见
    //      大小: 2 × 16×16 × 4 bytes = 2 KB
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // ---- 累加器 ----
    float sum = 0.0f;

    // ---- 主循环：每次处理一个 16×16 的 tile ----
    //       K=1024, TILE=16 → 循环 64 次（64 phases）
    for (int phase = 0; phase < K; phase += TILE_SIZE) {

        // ============================================================
        // Phase 1: 协作加载 —— 每个线程搬一个 A 元素和一个 B 元素
        // ============================================================
        //
        // 加载 A tile:
        //   从全局内存 A[row][phase + threadIdx.x] 搬到 As[threadIdx.y][threadIdx.x]
        //   注意: threadIdx.x 对应 A 的列索引，同一个 warp 内 threadIdx.x 递增
        //         → 相邻线程读相邻地址 → coalesced ✓
        //
        // 加载 B tile:
        //   从全局内存 B[phase + threadIdx.y][col] 搬到 Bs[threadIdx.y][threadIdx.x]
        //   注意: threadIdx.y 对应 B 的行索引
        //         → 同一个 warp 内 threadIdx.y 相同，所以 32 个线程读同一个 B 行
        //         → 但不同 threadIdx.x 导致不同列 → 需要检查是否 coalesced
        //         （实际上 B[phase+ty][col] 固定 ty，col 随 tx 变化 = 同一行连续列 = coalesced ✓）
        //
        // 一个 warp (32 threads) 的加载模式：
        //   Thread(ty=0, tx=0..31):
        //     As[0][0..31] ← A[row][phase+0..phase+31]    ← 同一行，连续 → coalesced ✓
        //     Bs[0][0..31] ← B[phase+0][col..col+31]      ← 同一行，连续 → coalesced ✓
        //
        // 边界检查：row/col 在 block 边界上，或者 phase 超出 K
        if (row < M && (phase + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = a[row * K + (phase + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (phase + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = b[(phase + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ============================================================
        // Phase 2: 同步 —— 等所有线程把 tile 搬完
        // ============================================================
        __syncthreads();

        // ============================================================
        // Phase 3: 在 shared memory 上计算
        // ============================================================
        //   每个线程做 TILE_SIZE 次乘加：
        //     sum += As[threadIdx.y][0] * Bs[0][threadIdx.x]
        //          + As[threadIdx.y][1] * Bs[1][threadIdx.x]
        //          + ...
        //          + As[threadIdx.y][15] * Bs[15][threadIdx.x]
        //
        //   注意访问模式：
        //     As[ty][k]  — ty 固定，k 递增 → 同一行连续列 → 无 bank conflict ✓
        //     Bs[k][tx]  — k 递增，tx 固定 → 同一列不同行
        //
        //   Bank conflict 分析（32 banks, 4-byte words, Bs[16][16] 行主序）：
        //     给定 k，warp 内 32 个线程访问 Bs[k][0..15]（前 16 线程）
        //     + Bs[k][0..15]（后 16 线程，完全相同的地址）。
        //     相同地址 = broadcast，不冲突。
        //     k=0: bank 0..15  |  k=1: bank 16..31  → 完美均匀分布 ✓
        //
        //   16×16 恰好是 32 bank 的一半，warp 覆盖 2 行 × 16 列，
        //   每行 16 个元素刚好占满半个 bank 集，自动无冲突。
        //   （如果 TILE_SIZE=32，就会出现 2-way bank conflict）
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // ============================================================
        // Phase 4: 同步 —— 等所有线程算完，再覆写 As/Bs 加载下一块
        // ============================================================
        __syncthreads();
    }

    // ---- 写回结果 ----
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

// ============================================================================
// 3b. GPU Tiled + 循环展开版本
// ============================================================================

/**
 * @brief Tiled 矩阵乘法 — 循环展开优化版
 *
 * 和 matmul_tiled_kernel 结构完全一样，唯一区别：
 * 内层的 16 次乘加循环被手动展开，消除循环开销。
 *
 * 为什么循环展开能加速：
 *   1. 消除 k++ 比较和分支（每次迭代省 1-2 条指令）
 *   2. 编译器更容易做指令重排和流水线优化
 *   3. As[ty][k] 和 Bs[k][tx] 的地址对编译器完全透明 → 可能合并为 float4 向量加载
 *
 * 手动展开 vs #pragma unroll：
 *   #pragma unroll 让编译器展开，但不保证——取决于启发式。
 *   手动展开是强制的，不依赖编译器判断。
 */
__global__ void matmul_tiled_unroll_kernel(const float *a, const float *b, float *c,
                                            int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    // 小技巧：存到寄存器变量，减少重复索引计算
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    for (int phase = 0; phase < K; phase += TILE_SIZE) {
        // ---- 协作加载（和原版一致） ----
        if (row < M && (phase + tx) < K) {
            As[ty][tx] = a[row * K + (phase + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        if (col < N && (phase + ty) < K) {
            Bs[ty][tx] = b[(phase + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        // ---- 手动展开：16 次乘加，无循环 ----
        // 编译器看到这 16 行连续的乘加，可以做：
        //   - 寄存器重命名，消除假依赖
        //   - 把连续的 As[ty][k] 合并为 float4 load (128-bit)
        //   - 指令级并行：load As[ty][k+1] 的同时算 As[ty][k] * Bs[k][tx]
        sum += As[ty][0]  * Bs[0][tx];
        sum += As[ty][1]  * Bs[1][tx];
        sum += As[ty][2]  * Bs[2][tx];
        sum += As[ty][3]  * Bs[3][tx];
        sum += As[ty][4]  * Bs[4][tx];
        sum += As[ty][5]  * Bs[5][tx];
        sum += As[ty][6]  * Bs[6][tx];
        sum += As[ty][7]  * Bs[7][tx];
        sum += As[ty][8]  * Bs[8][tx];
        sum += As[ty][9]  * Bs[9][tx];
        sum += As[ty][10] * Bs[10][tx];
        sum += As[ty][11] * Bs[11][tx];
        sum += As[ty][12] * Bs[12][tx];
        sum += As[ty][13] * Bs[13][tx];
        sum += As[ty][14] * Bs[14][tx];
        sum += As[ty][15] * Bs[15][tx];

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

// ============================================================================
// 4. 验证 + 工具函数
// ============================================================================

bool verify(const float *cpu, const float *gpu, int n, float eps = 5e-2f) {
    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = std::fabs(cpu[i] - gpu[i]);
        if (err > max_err) max_err = err;
        if (err > eps) {
            if (mismatches < 5) {
                fprintf(stderr, "  Mismatch[%d]: CPU=%.6f, GPU=%.6f, diff=%.6f\n",
                        i, cpu[i], gpu[i], err);
            }
            mismatches++;
        }
    }
    fprintf(stderr, "  Max error: %e\n", max_err);
    return mismatches == 0;
}

// ============================================================================
// 5. Main
// ============================================================================

int main(int argc, char **argv) {
    // ---- 参数 ----
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 1024;

    size_t bytes_a = M * K * sizeof(float);
    size_t bytes_b = K * N * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);

    // ---- Device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("========================================\n");
    printf("CUDA Week 2: Tiled Matrix Multiply\n");
    printf("========================================\n");
    printf("Dimensions:    A[%d×%d] × B[%d×%d] = C[%d×%d]\n", M, K, K, N, M, N);
    printf("Total FLOPs:   %.2f GFLOP\n", 2.0 * M * K * N / 1e9);
    printf("GPU:           %s (CC %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Tile size:     %d×%d\n", TILE_SIZE, TILE_SIZE);
    printf("Shared mem:    %.1f KB/block (limit: %zu KB)\n\n",
           2.0f * TILE_SIZE * TILE_SIZE * sizeof(float) / 1024.0f,
           prop.sharedMemPerBlock / 1024);

    // ---- Host alloc & init ----
    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c_cpu = new float[M * N];
    float *h_c_gpu = new float[M * N];

    srand(42);
    for (int i = 0; i < M * K; ++i) h_a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_b[i] = (float)rand() / RAND_MAX;

    // ---- Device alloc + copy ----
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

    // ---- Launch config ----
    dim3 block_dim(TILE_SIZE, TILE_SIZE);       // 16×16 = 256 threads
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Launch config:  grid(%d,%d), block(%d,%d)\n",
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    printf("Phases:         %d (K/TILE_SIZE)\n\n", K / TILE_SIZE);

    // ---- CUDA events for timing ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ================================================================
    // Test 1: CPU baseline (i-k-j loop, cache-friendly)
    // ================================================================
    printf("Running CPU (i-k-j)... ");
    fflush(stdout);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("%.2f ms (%.2f GFLOP/s)\n", cpu_ms, 2.0 * M * K * N / (cpu_ms * 1e6));

    // ================================================================
    // Test 2: GPU naive kernel
    // ================================================================
    printf("Running GPU (naive)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
    printf("%.2f ms (%.2f GFLOP/s)\n", naive_ms, 2.0 * M * K * N / (naive_ms * 1e6));

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes_c, cudaMemcpyDeviceToHost));
    bool naive_pass = verify(h_c_cpu, h_c_gpu, M * N);
    printf("  Verification: %s\n\n", naive_pass ? "PASSED ✓" : "FAILED ✗");

    // ================================================================
    // Test 3: GPU tiled kernel
    // ================================================================
    printf("Running GPU (tiled)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float tiled_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));
    printf("%.2f ms (%.2f GFLOP/s)\n", tiled_ms, 2.0 * M * K * N / (tiled_ms * 1e6));

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes_c, cudaMemcpyDeviceToHost));
    bool tiled_pass = verify(h_c_cpu, h_c_gpu, M * N);
    printf("  Verification: %s\n\n", tiled_pass ? "PASSED ✓" : "FAILED ✗");

    // ================================================================
    // Test 4: GPU tiled + unrolled kernel
    // ================================================================
    printf("Running GPU (tiled+unroll)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled_unroll_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float unroll_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&unroll_ms, start, stop));
    printf("%.2f ms (%.2f GFLOP/s)\n", unroll_ms, 2.0 * M * K * N / (unroll_ms * 1e6));

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes_c, cudaMemcpyDeviceToHost));
    bool unroll_pass = verify(h_c_cpu, h_c_gpu, M * N);
    printf("  Verification: %s\n\n", unroll_pass ? "PASSED ✓" : "FAILED ✗");

    // ================================================================
    // Summary
    // ================================================================
    printf("========== Results ==========\n");
    printf("CPU (i-k-j):         %.2f ms  (%.2f GFLOP/s)\n",
           cpu_ms, 2.0 * M * K * N / (cpu_ms * 1e6));
    printf("GPU (naive):         %.2f ms  (%.2f GFLOP/s)\n",
           naive_ms, 2.0 * M * K * N / (naive_ms * 1e6));
    printf("GPU (tiled):         %.2f ms  (%.2f GFLOP/s)\n",
           tiled_ms, 2.0 * M * K * N / (tiled_ms * 1e6));
    printf("GPU (tiled+unroll):  %.2f ms  (%.2f GFLOP/s)\n",
           unroll_ms, 2.0 * M * K * N / (unroll_ms * 1e6));

    printf("\n--- Speedups ---\n");
    printf("Tiled vs CPU:        %.1fx\n", cpu_ms / tiled_ms);
    printf("Tiled vs Naive:      %.1fx\n", naive_ms / tiled_ms);
    printf("Unroll vs Tiled:     %.1fx\n", tiled_ms / unroll_ms);

    printf("\n--- Efficiency ---\n");
    // RTX 3060 peak: ~12.7 TFLOPS (FP32)
    double peak_tflops = 12.7;
    printf("Best (unroll) GFLOPS: %.2f\n", 2.0 * M * K * N / (unroll_ms * 1e6));
    printf("Peak %%:             %.1f%% of %.1f TFLOPS\n",
           100.0 * (2.0 * M * K * N / (unroll_ms * 1e6)) / (peak_tflops * 1000), peak_tflops);

    // ---- Cleanup ----
    delete[] h_a; delete[] h_b; delete[] h_c_cpu; delete[] h_c_gpu;
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
