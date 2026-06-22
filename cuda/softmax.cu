/**
 * @file softmax.cu
 * @brief CUDA Week 3 — Softmax: Warp Reduce + Numerical Stability
 *
 * Learning objectives:
 *  - Softmax 数值稳定性: 为什么 exp(x-max) 而非直接 exp(x)
 *  - Warp Reduce: __shfl_down_sync 在 32 线程内做树形归约
 *  - Shared Memory 跨 warp 归约: warp 结果合并
 *  - 为什么 warp shuffle 比 shared memory 快（寄存器直接交换，无 bank conflict）
 *
 * Usage:
 *   nvcc -arch=sm_75 -O2 softmax.cu -o softmax
 *   ./softmax [rows] [cols]
 *
 * 默认: 4096×256 (每个 block 处理一行，256 threads = 8 warps)
 *
 * Core idea（一句话）:
 *   Softmax = find max → subtract → exp → sum → divide
 *   其中 "find max" 和 "sum" 是归约操作——warp reduce 用树形结构
 *   在 5 步内（log2(32)）完成 32 个值的归约，比 per-thread loop 快很多。
 *
 * Reference:
 *   CUDA C++ Programming Guide §B.25 (Warp Shuffle Functions)
 *   https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions
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

// ============================================================================
// 1. CPU: 数值稳定的 Softmax
// ============================================================================

/**
 * @brief CPU 行级 Softmax（数值稳定版本）
 *
 * 核心公式（稳定版）:
 *   softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
 *
 * 为什么不直接用 exp(x_i) / sum(exp(x))?
 *   - float 能表示的范围: ~1.2e-38 到 ~3.4e38
 *   - exp(89) ≈ 4.5e38  → 接近上限
 *   - exp(90) ≈ 1.2e39  → 溢出 → inf!
 *   - 如果输入向量是 [1000, 1001, 1002]:
 *     exp(1000) = inf, exp(1001) = inf, exp(1002) = inf
 *     → softmax = inf/inf = nan ❌
 *
 * 减去 max 后:
 *   - 输入变成 [-2, -1, 0]
 *   - exp(-2) = 0.135, exp(-1) = 0.368, exp(0) = 1.0
 *   - softmax = [0.09, 0.24, 0.67] ✓
 *
 * 数学上等价（分子分母同除以 exp(max)）:
 *   exp(x_i) / sum(exp(x_j)) = exp(x_i - max) / sum(exp(x_j - max))
 *
 * 时间复杂度: O(rows * cols) — 两遍扫描（找 max + 计算 sum）
 */
void softmax_cpu(const float *input, float *output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float *row_in = input + r * cols;
        float *row_out = output + r * cols;

        // Step 1: 找这一行的最大值（防止 exp 溢出）
        float max_val = row_in[0];
        for (int c = 1; c < cols; ++c) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        // Step 2: 计算 exp(x - max) 的累加和（分母）
        float sum_exp = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum_exp += expf(row_in[c] - max_val);
        }

        // Step 3: 归一化
        for (int c = 0; c < cols; ++c) {
            row_out[c] = expf(row_in[c] - max_val) / sum_exp;
        }
    }
}

// ============================================================================
// 2. GPU Kernel 1: Naive — 每线程独自做 loop（无 warp 协作）
// ============================================================================

/**
 * @brief Naive GPU Softmax: 每个线程自己遍历整行找 max 和 sum
 *
 * 并行策略:
 *   - 每个 block 处理一行 → gridDim.x = rows
 *   - blockDim.x = cols（假设 cols ≤ 1024）
 *   - 每个线程负责输出的一个元素
 *   - 但每个线程需要自己遍历整行来找 max 和 sum
 *
 * 问题: 每个线程都要读整行数据 — 重复读取，浪费带宽
 *   - 如果有 256 个线程，同一行被读了 256 遍
 *   - 每个线程 O(cols) 而不是 O(1) 的全局内存读取
 */
__global__ void softmax_naive_kernel(const float *input, float *output,
                                      int rows, int cols) {
    int row = blockIdx.x;       // 每一行一个 block
    int tid = threadIdx.x;      // 线程在行内的位置

    if (row >= rows || tid >= cols) return;

    const float *row_in = input + row * cols;

    // Step 1: 找 max — 每个线程独立遍历整行
    float max_val = row_in[0];
    for (int c = 1; c < cols; ++c) {
        if (row_in[c] > max_val) max_val = row_in[c];
    }

    // Step 2: 计算 sum — 每个线程独立遍历整行
    float sum_exp = 0.0f;
    for (int c = 0; c < cols; ++c) {
        sum_exp += expf(row_in[c] - max_val);
    }

    // Step 3: 写入自己的结果
    float val = expf(row_in[tid] - max_val) / sum_exp;
    output[row * cols + tid] = val;
}

// ============================================================================
// 3. Warp Reduce Primitives — Week 3 核心！
// ============================================================================

/**
 * @brief Warp-level Sum Reduce
 *
 * 工作原理（树形归约，"蝴蝶"模式）:
 *
 *   假设 8 个线程的值: [a, b, c, d, e, f, g, h]
 *
 *   Step 1 (offset=4):  线程 0..3 从线程 4..7 获取值并累加
 *     lane 0: a + e     lane 1: b + f
 *     lane 2: c + g     lane 3: d + h
 *     lane 4..7: 原值不变（但后续不再使用）
 *
 *   Step 2 (offset=2):  线程 0..1 从线程 2..3 获取累加
 *     lane 0: (a+e) + (c+g)     lane 1: (b+f) + (d+h)
 *
 *   Step 3 (offset=1):  线程 0 从线程 1 获取最终结果
 *     lane 0: (a+e+c+g) + (b+f+d+h) = 总和 ✓
 *
 *   32 线程的 warp 只需要 log2(32) = 5 步。
 *
 * __shfl_down_sync(mask, val, delta):
 *   - mask: 参与线程的掩码，0xffffffff = 全部 32 个线程
 *   - val: 当前线程要共享的值
 *   - delta: 从比自己高 delta 号的线程获取值
 *   - 返回: lane_id + delta 号线程的 val
 *
 *   lane 0 调 __shfl_down_sync(0xffffffff, my_val, 4) 返回 lane 4 的 my_val
 *   lane 30 调 __shfl_down_sync(0xffffffff, my_val, 4) 返回 0（越界）
 *
 * 为什么比 shared memory 快？
 *   - 寄存器直接传递，延迟 ~1 cycle（vs shared memory ~20 cycles）
 *   - 不需要 __syncthreads()
 *   - 无 bank conflict
 */
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // 只有 lane 0 有完整结果
}

/// @brief Warp-level Max Reduce（同上，只是把 + 换成 fmaxf）
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) val = other;
    }
    return val;  // 只有 lane 0 有完整结果
}

// ============================================================================
// 4. GPU Kernel 2: Warp Reduce + Shared Memory
// ============================================================================

/**
 * @brief Warp Reduce Softmax — Week 3 核心 kernel
 *
 * 策略（一个 block 处理一行，分成多步）:
 *
 *   Step 1: 每个线程加载一个元素
 *   Step 2: 每个 warp 内做 warp_reduce_max → 8 个 warp 各产出一个 max
 *   Step 3: 8 个 warp_max 写入 shared memory，第一个 warp 再做一次 reduce
 *           → 得到全局 max
 *   Step 4: 每个线程计算 exp(x - max)
 *   Step 5: warp_reduce_sum → 8 个 warp 各产出一个 sum → 再次 reduce → 全局 sum
 *   Step 6: 每个线程 exp(x - max) / sum → 写回
 *
 * 对比 naive kernel:
 *   Naive: 每个线程遍历整行 → O(cols) per thread = cols × threads 次全局读
 *   Warp:  每个线程只读 1 次 → O(1) per thread
 *
 * 全局内存读取量: 256（warps=8时）vs 256×256=65536（naive）
 */
__global__ void softmax_warp_reduce_kernel(const float *input, float *output,
                                            int rows, int cols) {
    // ---- 动态 shared memory: 每个 warp 的 lane 0 写入部分结果 ----
    //      warp_count 个 float，用于跨 warp 归约
    extern __shared__ float smem[];  // 大小在 kernel launch 时指定
    float *warp_results = smem;     // 前 warp_count 个元素存部分结果

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;      // 当前线程属于哪个 warp
    int lane_id = tid % 32;      // 在 warp 内的位置 (0..31)
    int num_warps = blockDim.x / 32;

    if (row >= rows) return;

    const float *row_in = input + row * cols;

    // ============================================================
    // Step 1: 加载数据
    // ============================================================
    //   如果 cols > blockDim.x，需要 stride loop 处理多元素
    //   这里简化：cols 必须是 256（= blockDim.x）
    float val = (tid < cols) ? row_in[tid] : -INFINITY;

    // ============================================================
    // Step 2: 找全局 max
    // ============================================================

    // 2a. Warp 内归约：每个 warp 的 lane 0 拿到该 warp 的最大值
    float warp_max = warp_reduce_max(val);

    // 2b. 每个 warp 的 lane 0 把结果写入 shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = warp_max;
    }
    __syncthreads();  // 等所有 warp 的结果都写入

    // 2c. 第一个 warp 从 shared memory 读所有 warp 结果，再做一次 reduce
    //     （如果 num_warps ≤ 32，第一个 warp 可以内部搞定）
    float block_max = warp_results[lane_id];  // 只对第一个 warp 有效
    if (warp_id == 0) {
        // 第一个 warp 的每个 lane 加载一个 warp 的部分结果
        block_max = (lane_id < num_warps) ? warp_results[lane_id] : -INFINITY;
        block_max = warp_reduce_max(block_max);  // 归约得到全局 max
    }
    // Broadcast: 把 block_max 从 warp 0 lane 0 广播给所有线程
    // （通过 shared memory）
    if (lane_id == 0 && warp_id == 0) {
        warp_results[0] = block_max;  // 存到 smem[0]
    }
    __syncthreads();
    block_max = warp_results[0];  // 所有线程读到同一个 max
    __syncthreads();  // 确保都读完了，接下来要覆写 warp_results

    // ============================================================
    // Step 3: 计算 exp(x - max)
    // ============================================================
    float exp_val = (tid < cols) ? expf(val - block_max) : 0.0f;

    // ============================================================
    // Step 4: 归约求和（与 Step 2 完全相同的模式，只是用 sum 代替 max）
    // ============================================================

    // 4a. Warp 内求和
    float warp_sum = warp_reduce_sum(exp_val);

    // 4b. 每个 warp 的 lane 0 写入
    if (lane_id == 0) {
        warp_results[warp_id] = warp_sum;
    }
    __syncthreads();

    // 4c. 第一个 warp 再求和
    float block_sum = (warp_id == 0 && lane_id < num_warps)
                      ? warp_results[lane_id] : 0.0f;
    if (warp_id == 0) {
        block_sum = warp_reduce_sum(block_sum);
    }

    // 4d. Broadcast block_sum
    if (lane_id == 0 && warp_id == 0) {
        warp_results[0] = block_sum;
    }
    __syncthreads();
    block_sum = warp_results[0];

    // ============================================================
    // Step 5: 归一化输出
    // ============================================================
    if (tid < cols) {
        output[row * cols + tid] = exp_val / block_sum;
    }
}

// ============================================================================
// 5. GPU Kernel 3: Block Reduce（一个 warp 搞定全部——当 num_warps=1 时）
// ============================================================================

/**
 * @brief Single Warp Softmax — 每行一个 warp，线程内循环处理多个元素
 *
 * 与 multi-warp 版本的对比:
 *   Multi-warp: grid(rows, cols) — 一个 block 256 线程处理一行
 *               需要 shared memory 做跨 warp 归约
 *   Single-warp: grid(rows, 32) — 一个 warp 32 线程处理一整行
 *               每个线程通过局部循环处理 cols/32 个元素，
 *               然后 warp reduce 归约这些局部结果。
 *               不需要 shared memory！
 *
 * 为什么 cols=256, warp=32 时每个线程处理 8 个元素:
 *   线程 0: 处理 input[0..7]   → 找局部 max/sum
 *   线程 1: 处理 input[8..15]  → 找局部 max/sum
 *   ...
 *   线程 31: 处理 input[248..255]
 *   然后 warp_reduce_max/sum 把 32 个局部结果归约成全局结果。
 *
 * 这种模式在 attention 里非常常见——每个 head 的维度通常 ≤ 128，
 * 用 1-4 个 warp 就能覆盖。
 */
__global__ void softmax_single_warp_kernel(const float *input, float *output,
                                            int rows, int cols) {
    int row = blockIdx.x;        // 每行一个 block
    int lane = threadIdx.x;      // 0..31

    if (row >= rows) return;

    const float *row_in = input + row * cols;
    int elems_per_lane = cols / 32;   // 每个线程处理多少个元素

    // ============================================================
    // Step 1: 每个线程先在自己的 cols/32 个元素里找局部 max
    // ============================================================
    float local_max = -INFINITY;
    for (int i = 0; i < elems_per_lane; ++i) {
        float v = row_in[lane * elems_per_lane + i];
        if (v > local_max) local_max = v;
    }

    // ============================================================
    // Step 2: Warp reduce 找全局 max
    // ============================================================
    float max_val = warp_reduce_max(local_max);
    max_val = __shfl_sync(0xffffffff, max_val, 0);  // lane 0 广播给所有线程

    // ============================================================
    // Step 3: 每个线程算局部 sum(exp)
    // ============================================================
    float local_sum = 0.0f;
    for (int i = 0; i < elems_per_lane; ++i) {
        local_sum += expf(row_in[lane * elems_per_lane + i] - max_val);
    }

    // ============================================================
    // Step 4: Warp reduce 求全局 sum
    // ============================================================
    float sum_val = warp_reduce_sum(local_sum);
    sum_val = __shfl_sync(0xffffffff, sum_val, 0);

    // ============================================================
    // Step 5: 归一化写回
    // ============================================================
    for (int i = 0; i < elems_per_lane; ++i) {
        int idx = lane * elems_per_lane + i;
        float v = row_in[idx];
        output[row * cols + idx] = expf(v - max_val) / sum_val;
    }
}

// ============================================================================
// 6. 验证 + 工具函数
// ============================================================================

static bool verify(const float *cpu, const float *gpu, int n, float eps = 1e-4f) {
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
// 7. Main
// ============================================================================

#ifndef KERNEL_EXPORT
int main(int argc, char **argv) {
    // ---- 参数 ----
    int rows = (argc > 1) ? atoi(argv[1]) : 4096;
    int cols = (argc > 2) ? atoi(argv[2]) : 256;

    // cols 必须是 32 的倍数（warp size），且 ≤ 1024（max threads per block）
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
    printf("CUDA Week 3: Softmax — Warp Reduce\n");
    printf("========================================\n");
    printf("Dimensions:      [%d × %d] = %d elements (%.2f MB)\n",
           rows, cols, n, bytes / 1e6);
    printf("GPU:             %s (CC %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Warp size:       %d\n", prop.warpSize);
    printf("Max threads/blk: %d\n", prop.maxThreadsPerBlock);
    printf("Threads/block:   %d (%d warps)\n\n", cols, cols / 32);

    // ---- Host alloc & init ----
    float *h_input  = new float[n];
    float *h_cpu    = new float[n];
    float *h_gpu    = new float[n];

    // 用较大的随机数范围，展示数值稳定性的作用
    srand(42);
    for (int i = 0; i < n; ++i) {
        h_input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // [-10, 10]
    }

    // ---- Device alloc ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // ---- CUDA events ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ================================================================
    // Test 1: CPU baseline
    // ================================================================
    printf("Running CPU (stable)... ");
    fflush(stdout);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    softmax_cpu(h_input, h_cpu, rows, cols);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("%.2f ms\n", cpu_ms);

    // 验证: 每行 softmax 之和应该 ≈ 1.0
    {
        float max_row_sum_err = 0.0f;
        for (int r = 0; r < rows; ++r) {
            float row_sum = 0.0f;
            for (int c = 0; c < cols; ++c) row_sum += h_cpu[r * cols + c];
            float err = fabsf(row_sum - 1.0f);
            if (err > max_row_sum_err) max_row_sum_err = err;
        }
        printf("  Row sum check:  max |sum-1| = %.2e  (should be < 1e-5)\n\n",
               max_row_sum_err);
    }

    // ================================================================
    // Test 2: GPU Naive
    // ================================================================
    printf("Running GPU (naive)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    softmax_naive_kernel<<<rows, cols>>>(d_input, d_output, rows, cols);
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
    int smem_size = num_warps * sizeof(float);  // 每个 warp 一个 float

    printf("Running GPU (warp reduce)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    softmax_warp_reduce_kernel<<<rows, cols, smem_size>>>(
        d_input, d_output, rows, cols);
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
    // Test 4: GPU Single Warp（当 block=32 时，演示最简形式）
    // ================================================================
    printf("Running GPU (single warp, block=32)... ");
    fflush(stdout);
    CUDA_CHECK(cudaEventRecord(start));
    // 每行一个 block，每个 block 32 threads（一个 warp）
    // 每个线程通过局部循环处理 cols/32 个元素
    softmax_single_warp_kernel<<<rows, 32>>>(
        d_input, d_output, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float sw_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&sw_ms, start, stop));
    printf("%.2f ms", sw_ms);
    if (sw_ms < cpu_ms)
        printf("  (%.2fx speedup vs CPU)", cpu_ms / sw_ms);
    printf("\n");

    CUDA_CHECK(cudaMemcpy(h_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    printf("  Verification:   %s\n\n",
           verify(h_cpu, h_gpu, n) ? "PASSED ✓" : "FAILED ✗");

    // ================================================================
    // Summary
    // ================================================================
    printf("========== Results ==========\n");
    printf("%-25s %8s  %s\n", "Version", "Time", "vs CPU");
    printf("%-25s %6.2f ms  %7s\n", "CPU (stable)", cpu_ms, "—");
    printf("%-25s %6.2f ms  %6.2fx\n",
           "GPU (naive)", naive_ms, cpu_ms / naive_ms);
    printf("%-25s %6.2f ms  %6.2fx\n",
           "GPU (warp reduce)", warp_ms, cpu_ms / warp_ms);
    printf("%-25s %6.2f ms  %6.2fx\n",
           "GPU (single warp)", sw_ms, cpu_ms / sw_ms);

    printf("\n--- Relative Speedups ---\n");
    printf("Warp Reduce vs Naive:       %.2fx\n", naive_ms / warp_ms);
    printf("Single Warp vs Warp Reduce: %.2fx\n", warp_ms / sw_ms);

    printf("\n--- Key Takeaways ---\n");
    printf("1. Naive GPU may be SLOWER than CPU — each thread reads the\n");
    printf("   entire row, wasting memory bandwidth.\n");
    printf("2. Warp reduce lets threads collaborate: each reads 1 element,\n");
    printf("   then __shfl_down_sync does tree reduction in 5 steps.\n");
    printf("3. Single-warp kernel is simplest: no shared memory, no\n");
    printf("   __syncthreads, just registers and warp shuffles.\n");
    printf("4. These primitives (warp_reduce_sum, warp_reduce_max) are\n");
    printf("   reusable — you'll use them again for LayerNorm.\n");

    // ---- Cleanup ----
    delete[] h_input; delete[] h_cpu; delete[] h_gpu;
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
#endif // KERNEL_EXPORT
