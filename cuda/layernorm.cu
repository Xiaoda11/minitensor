/**
 * @file layernorm.cu
 * @brief CUDA Week 3 — LayerNorm: Warp Reduce 实战应用
 *
 * Learning objectives:
 *  - 用 warp reduce 计算均值和方差（两次归约）
 *  - LayerNorm 的前向传播公式
 *  - Welford 在线算法（单遍计算方差，避免数值精度问题）
 *  - 与 softmax 对比：都是"归约 + 归一化"，结构类似
 *
 * Usage:
 *   nvcc -arch=sm_75 -O2 layernorm.cu -o layernorm
 *   ./layernorm [rows] [cols]
 *
 * 默认: 4096×256
 *
 * Core idea（一句话）:
 *   LayerNorm = 两次 warp_reduce_sum（算 mean 和 variance）
 *             + element-wise 归一化
 *   和 softmax 的"max reduce + sum reduce"结构完全相同。
 *
 * Formula:
 *   mean     = (1/N) * Σ x_i
 *   variance = (1/N) * Σ (x_i - mean)²
 *   y_i      = (x_i - mean) / sqrt(variance + ε)
 *
 * Reference:
 *   Ba, Kiros, Hinton (2016) "Layer Normalization"
 *   https://arxiv.org/abs/1607.06450
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ============================================================================
// 0. Macros + Warp Primitives（与 softmax.cu 完全相同，复用）
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

/**
 * Warp Reduce Sum — 和 softmax.cu 里一模一样的函数
 *
 * 这是你 CUDA 工具箱里的"标准件"——写一次，到处用。
 * softmax 里用它求 sum(exp)，LayerNorm 里用它求 sum(x) 和 sum((x-mean)²)。
 */
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// 1. CPU: LayerNorm
// ============================================================================

/**
 * @brief CPU LayerNorm（行级归一化，无 gamma/beta 简化版）
 *
 * LayerNorm vs BatchNorm:
 *   - BatchNorm: 对同一通道跨 batch 归一化（受 batch size 影响）
 *   - LayerNorm: 对同一样本跨特征归一化（batch size 无关，适合 NLP）
 *
 * Transformer 里每个 token 的 hidden state 做一次 LayerNorm。
 * 例如 hidden_dim=768，那就是对 768 个值求 mean/variance 然后归一化。
 */
void layernorm_cpu(const float *input, float *output, int rows, int cols,
                   float eps = 1e-5f) {
    for (int r = 0; r < rows; ++r) {
        const float *row_in = input + r * cols;
        float *row_out = output + r * cols;

        // Step 1: 计算 mean — 一次归约
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) sum += row_in[c];
        float mean = sum / cols;

        // Step 2: 计算 variance — 第二次归约
        float var_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = row_in[c] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / cols;

        // Step 3: 归一化
        float inv_std = 1.0f / sqrtf(variance + eps);
        for (int c = 0; c < cols; ++c) {
            row_out[c] = (row_in[c] - mean) * inv_std;
        }
    }
}

// ============================================================================
// 2. GPU Kernel 1: Naive（每线程独立算，同 softmax naive）
// ============================================================================

/**
 * @brief Naive GPU LayerNorm — 和 naive softmax 同样的结构性问题
 *
 * 每个线程遍历整行来算 mean 和 variance — 数据被读了 threads 次。
 */
__global__ void layernorm_naive_kernel(const float *input, float *output,
                                        int rows, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows || tid >= cols) return;

    const float *row_in = input + row * cols;

    // Step 1: mean（每个线程独立遍历整行）
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) sum += row_in[c];
    float mean = sum / cols;

    // Step 2: variance（再次遍历整行）
    float var_sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float diff = row_in[c] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / cols;

    // Step 3: normalize
    float inv_std = rsqrtf(variance + eps);
    output[row * cols + tid] = (row_in[tid] - mean) * inv_std;
}

// ============================================================================
// 3. GPU Kernel 2: Warp Reduce LayerNorm（核心）
// ============================================================================

/**
 * @brief Warp Reduce LayerNorm — Week 3 核心应用
 *
 * 和 softmax_warp_reduce_kernel 的结构完全相同：
 *
 *   Softmax:                         LayerNorm:
 *     warp_reduce_max  → max           warp_reduce_sum  → sum → mean
 *     warp_reduce_sum  → sum(exp)      warp_reduce_sum  → sum(diff²) → var
 *     normalize                        normalize
 *
 * 学完这个你手里有两套"归约零件":
 *   - warp_reduce_sum (LayerNorm + softmax 的 sum 部分)
 *   - warp_reduce_max (softmax 的 max 部分)
 *
 * 后续 Transformer 里的 Attention Softmax + LayerNorm 就是这两个的组合。
 */
__global__ void layernorm_warp_reduce_kernel(const float *input, float *output,
                                              int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float *warp_results = smem;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    if (row >= rows) return;

    const float *row_in = input + row * cols;
    float val = (tid < cols) ? row_in[tid] : 0.0f;

    // ============================================================
    // Step 1: 归约求和 → 得到 mean
    // ============================================================

    // 1a. Warp 内求和
    float warp_sum = warp_reduce_sum(val);

    // 1b. 各 warp 的 lane 0 写入 shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = warp_sum;
    }
    __syncthreads();

    // 1c. 第一个 warp 归约所有 warp 的和
    float block_sum = (warp_id == 0 && lane_id < num_warps)
                      ? warp_results[lane_id] : 0.0f;
    if (warp_id == 0) {
        block_sum = warp_reduce_sum(block_sum);
    }

    // 1d. Broadcast mean 给所有线程
    if (lane_id == 0 && warp_id == 0) {
        warp_results[0] = block_sum;
    }
    __syncthreads();
    float mean = warp_results[0] / cols;

    // ============================================================
    // Step 2: 归约求方差（需要先用 mean 算 diff²）
    // ============================================================
    float diff = (tid < cols) ? (val - mean) : 0.0f;
    float diff_sq = diff * diff;

    // 2a. Warp 内求和（diff²）
    float warp_var = warp_reduce_sum(diff_sq);

    // 2b. 各 warp 的 lane 0 写入
    if (lane_id == 0) {
        warp_results[warp_id] = warp_var;
    }
    __syncthreads();

    // 2c. 第一个 warp 归约
    float block_var = (warp_id == 0 && lane_id < num_warps)
                      ? warp_results[lane_id] : 0.0f;
    if (warp_id == 0) {
        block_var = warp_reduce_sum(block_var);
    }

    // 2d. Broadcast variance
    if (lane_id == 0 && warp_id == 0) {
        warp_results[0] = block_var;
    }
    __syncthreads();
    float variance = warp_results[0] / cols;

    // ============================================================
    // Step 3: 归一化输出
    // ============================================================
    if (tid < cols) {
        float inv_std = rsqrtf(variance + eps);  // rsqrtf = 1/sqrtf，GPU 硬件指令
        output[row * cols + tid] = diff * inv_std;
    }
}

// ============================================================================
// 4. GPU Kernel 3: Welford 单遍算法（进阶）
// ============================================================================

/**
 * @brief Welford 在线方差算法 — 只需要一次归约
 *
 * 上面的 kernel 需要两次归约（先算 mean，再算 variance）。
 * Welford 算法可以在一次遍历中同时算出 mean 和 variance，
 * 数值精度也更好。
 *
 * Welford 递推公式（对序列 x_1, x_2, ..., x_n）:
 *   M_1 = x_1                           (n=1 时的 mean)
 *   S_1 = 0                             (n=1 时的 M2)
 *
 *   对 n ≥ 2:
 *     delta = x_n - M_{n-1}
 *     M_n   = M_{n-1} + delta / n       (更新 mean)
 *     S_n   = S_{n-1} + delta * (x_n - M_n)   (更新 M2)
 *
 *   最终: mean = M_N,  variance = S_N / N
 *
 * 不过这里我们不用递推（因为所有 x 已知），用并行 Welford：
 *   每个线程算自己的局部 (count, mean, M2)
 *   然后 warp/block reduce 合并这些局部统计量
 *
 * 合并公式（两组 (n_a, m_a, s_a) 和 (n_b, m_b, s_b) 合并）:
 *   delta = m_b - m_a
 *   n_ab  = n_a + n_b
 *   m_ab  = m_a + delta * n_b / n_ab
 *   s_ab  = s_a + s_b + delta * delta * n_a * n_b / n_ab
 *
 * 这个 kernel 的教学目的不是让你背公式，而是展示:
 *   "warp reduce 不仅可以归约标量，也可以归约结构体（count, mean, M2）"
 *
 * 实际工程中，PyTorch 的 LayerNorm CUDA kernel 用的就是 Welford。
 */
struct WelfordState {
    int count;      // 样本数
    float mean;     // 当前均值
    float m2;       // 二阶矩（sum of squared diffs）
};

__device__ WelfordState welford_combine(WelfordState a, WelfordState b) {
    if (b.count == 0) return a;
    if (a.count == 0) return b;

    int total = a.count + b.count;
    float delta = b.mean - a.mean;
    float mean = a.mean + delta * b.count / total;
    float m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / total;
    return {total, mean, m2};
}

__global__ void layernorm_welford_kernel(const float *input, float *output,
                                          int rows, int cols, float eps) {
    extern __shared__ float smem[];
    WelfordState *warp_states = (WelfordState *)smem;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    if (row >= rows) return;

    const float *row_in = input + row * cols;

    // Step 1: 每个线程初始化自己的 Welford 状态
    WelfordState state;
    if (tid < cols) {
        state = {1, row_in[tid], 0.0f};  // count=1, mean=x, M2=0
    } else {
        state = {0, 0.0f, 0.0f};         // 无效状态
    }

    // Step 2: Warp 内归约 Welford 状态（不是简单标量相加！）
    for (int offset = 16; offset > 0; offset /= 2) {
        WelfordState other;
        other.count = __shfl_down_sync(0xffffffff, state.count, offset);
        other.mean  = __shfl_down_sync(0xffffffff, state.mean, offset);
        other.m2    = __shfl_down_sync(0xffffffff, state.m2, offset);
        state = welford_combine(state, other);
    }

    // Step 3: 各 warp 的 lane 0 写入 shared memory
    if (lane_id == 0) {
        warp_states[warp_id] = state;
    }
    __syncthreads();

    // Step 4: 第一个 warp 合并所有 warp 的结果
    WelfordState block_state = {0, 0.0f, 0.0f};
    if (warp_id == 0) {
        block_state = (lane_id < num_warps) ? warp_states[lane_id]
                                            : WelfordState{0, 0.0f, 0.0f};
        for (int offset = 16; offset > 0; offset /= 2) {
            WelfordState other;
            other.count = __shfl_down_sync(0xffffffff, block_state.count, offset);
            other.mean  = __shfl_down_sync(0xffffffff, block_state.mean, offset);
            other.m2    = __shfl_down_sync(0xffffffff, block_state.m2, offset);
            block_state = welford_combine(block_state, other);
        }
    }

    // Step 5: Broadcast mean 和 variance
    if (lane_id == 0 && warp_id == 0) {
        warp_states[0] = block_state;
    }
    __syncthreads();

    float mean = warp_states[0].mean;
    float variance = warp_states[0].m2 / cols;

    // Step 6: 归一化
    if (tid < cols) {
        float inv_std = rsqrtf(variance + eps);
        output[row * cols + tid] = (row_in[tid] - mean) * inv_std;
    }
}

// ============================================================================
// 5. 验证 + 工具函数
// ============================================================================

static bool verify(const float *cpu, const float *gpu, int n, float eps = 1e-3f) {
    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = fabsf(cpu[i] - gpu[i]);
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
// 6. Main
// ============================================================================

#ifndef KERNEL_EXPORT
int main(int argc, char **argv) {
    // ---- 参数 ----
    int rows = (argc > 1) ? atoi(argv[1]) : 4096;
    int cols = (argc > 2) ? atoi(argv[2]) : 256;
    float eps = 1e-5f;

    if (cols % 32 != 0 || cols > 1024) {
        fprintf(stderr, "Error: cols must be multiple of 32 and ≤ 1024\n");
        return 1;
    }

    int n = rows * cols;
    size_t bytes = n * sizeof(float);

    // ---- Device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("========================================\n");
    printf("CUDA Week 3: LayerNorm — Warp Reduce\n");
    printf("========================================\n");
    printf("Dimensions:      [%d × %d] = %d elements (%.2f MB)\n",
           rows, cols, n, bytes / 1e6);
    printf("GPU:             %s (CC %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Threads/block:   %d (%d warps)\n\n", cols, cols / 32);

    // ---- Host alloc & init ----
    float *h_input  = new float[n];
    float *h_cpu    = new float[n];
    float *h_gpu    = new float[n];

    // 模拟 Transformer hidden states（零均值附近，小方差）
    srand(42);
    for (int i = 0; i < n; ++i) {
        // 正态分布近似: Box-Muller 太慢，用均匀分布近似
        h_input[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;  // [-2, 2]
    }

    // ---- Device alloc ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ================================================================
    // Test 1: CPU baseline
    // ================================================================
    printf("Running CPU... ");
    fflush(stdout);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    layernorm_cpu(h_input, h_cpu, rows, cols, eps);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("%.2f ms\n", cpu_ms);

    // 验证: 每行 LayerNorm 输出应为 mean≈0, std≈1
    {
        float max_mean_err = 0.0f, max_std_err = 0.0f;
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int c = 0; c < cols; ++c) {
                sum += h_cpu[r * cols + c];
                sum_sq += h_cpu[r * cols + c] * h_cpu[r * cols + c];
            }
            float m = sum / cols;
            float s = sqrtf(sum_sq / cols - m * m);
            if (fabsf(m) > max_mean_err) max_mean_err = fabsf(m);
            if (fabsf(s - 1.0f) > max_std_err) max_std_err = fabsf(s - 1.0f);
        }
        printf("  Mean check: max |mean| = %.2e  (should be < 1e-5)\n", max_mean_err);
        printf("  Std check:  max |std-1| = %.2e  (should be < 1e-4)\n\n", max_std_err);
    }

    // ================================================================
    // Test 2: GPU Naive
    // ================================================================
    printf("Running GPU (naive)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    layernorm_naive_kernel<<<rows, cols>>>(d_input, d_output, rows, cols, eps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
    printf("%.2f ms", naive_ms);
    if (naive_ms < cpu_ms)
        printf("  (%.2fx speedup vs CPU)", cpu_ms / naive_ms);
    else
        printf("  (%.2fx SLOWER than CPU)", naive_ms / cpu_ms);
    printf("\n");

    CUDA_CHECK(cudaMemcpy(h_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("  Verification:   %s\n\n",
           verify(h_cpu, h_gpu, n) ? "PASSED ✓" : "FAILED ✗");

    // ================================================================
    // Test 3: GPU Warp Reduce
    // ================================================================
    int num_warps = cols / 32;
    int smem_size = num_warps * sizeof(float);

    printf("Running GPU (warp reduce)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    layernorm_warp_reduce_kernel<<<rows, cols, smem_size>>>(
        d_input, d_output, rows, cols, eps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float warp_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&warp_ms, start, stop));
    printf("%.2f ms", warp_ms);
    if (warp_ms < cpu_ms)
        printf("  (%.2fx speedup vs CPU)", cpu_ms / warp_ms);
    printf("\n");

    CUDA_CHECK(cudaMemcpy(h_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("  Verification:   %s\n\n",
           verify(h_cpu, h_gpu, n) ? "PASSED ✓" : "FAILED ✗");

    // ================================================================
    // Test 4: GPU Welford（单遍算法）
    // ================================================================
    int welford_smem = num_warps * sizeof(WelfordState);

    printf("Running GPU (Welford)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    layernorm_welford_kernel<<<rows, cols, welford_smem>>>(
        d_input, d_output, rows, cols, eps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float welford_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&welford_ms, start, stop));
    printf("%.2f ms", welford_ms);
    if (welford_ms < cpu_ms)
        printf("  (%.2fx speedup vs CPU)", cpu_ms / welford_ms);
    printf("\n");

    CUDA_CHECK(cudaMemcpy(h_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("  Verification:   %s\n\n",
           verify(h_cpu, h_gpu, n, 2e-3f) ? "PASSED ✓" : "FAILED ✗");

    // ================================================================
    // Summary
    // ================================================================
    printf("========== Results ==========\n");
    printf("%-25s %8s  %s\n", "Version", "Time", "vs CPU");
    printf("%-25s %6.2f ms  %7s\n", "CPU", cpu_ms, "—");
    printf("%-25s %6.2f ms  %6.2fx\n",
           "GPU (naive)", naive_ms, cpu_ms / naive_ms);
    printf("%-25s %6.2f ms  %6.2fx\n",
           "GPU (warp reduce)", warp_ms, cpu_ms / warp_ms);
    printf("%-25s %6.2f ms  %6.2fx\n",
           "GPU (Welford)", welford_ms, cpu_ms / welford_ms);

    printf("\n--- Relative Speedups ---\n");
    printf("Warp Reduce vs Naive:  %.2fx\n", naive_ms / warp_ms);
    printf("Welford vs Warp:       %.2fx\n", warp_ms / welford_ms);

    printf("\n--- Key Takeaways ---\n");
    printf("1. LayerNorm = 两次 warp_reduce_sum（mean + variance）\n");
    printf("   + element-wise normalize。和 softmax 结构完全相同！\n");
    printf("2. warp_reduce_sum 是通用标准件——softmax 求和用它，\n");
    printf("   LayerNorm 求和也用它。\n");
    printf("3. Welford 单遍算法：只做一次归约同时得到 mean 和 variance，\n");
    printf("   数值精度更好。PyTorch 内部用的就是这个。\n");
    printf("4. 你现在掌握了两组 warp reduce 原语:\n");
    printf("   - warp_reduce_sum  (LayerNorm + softmax sum)\n");
    printf("   - warp_reduce_max  (softmax max)\n");
    printf("   后续 Attention 的 softmax + LayerNorm 就是这两组合。\n");

    // ---- Cleanup ----
    delete[] h_input; delete[] h_cpu; delete[] h_gpu;
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
#endif // KERNEL_EXPORT
