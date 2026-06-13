/**
 * @file attention.cu
 * @brief CUDA Week 4 — Scaled Dot-Product Attention (Fused Kernel)
 *
 * Learning objectives:
 *  - 把 Week 2 (tiled matmul) + Week 3 (warp reduce softmax) 拼成一个 kernel
 *  - Online softmax: 逐 tile 更新 running max/sum，避免存储整个 [S×S] 矩阵
 *  - 核融合 (kernel fusion): 消除中间矩阵的显存读写
 *
 * Formula:
 *   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 *   其中 Q, K, V ∈ R^{S × D}，S=序列长度，D=head_dim
 *
 * 为什么不用三个独立 kernel？
 *   - Q @ K^T 产生 [S×S] 的中间矩阵，显存占用 O(S²)
 *   - 三个 kernel 各读/写一次这个 S×S 矩阵 → 带宽浪费
 *   - 融合 kernel: S×S 矩阵只在寄存器和 shared memory 里存在，不写回显存
 *
 * Usage:
 *   nvcc -arch=sm_75 -O2 attention.cu -o attention
 *   ./attention [S] [D]
 *
 * 默认: S=128, D=64（典型单头注意力尺寸）
 *
 * Reference:
 *   Vaswani et al. (2017) "Attention Is All You Need"
 *   FlashAttention (Dao et al., 2022) — online softmax 的灵感来源
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
// 1. CPU: Reference Attention
// ============================================================================

/**
 * @brief CPU 参考实现 — 分三步走，便于理解和验证
 *
 * Step 1: S = Q @ K^T           → [S × S]
 * Step 2: P = softmax(S / √D)   → 每行独立
 * Step 3: O = P @ V             → [S × D]
 */
void attention_cpu(const float *Q, const float *K, const float *V,
                   float *output, int S, int D) {
    float scale = 1.0f / sqrtf((float)D);

    float *scores = new float[S * S];
    float *probs  = new float[S * S];

    // Step 1: Q @ K^T
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < D; ++k) {
                dot += Q[i * D + k] * K[j * D + k];
            }
            scores[i * S + j] = dot * scale;
        }
    }

    // Step 2: row-wise softmax
    for (int i = 0; i < S; ++i) {
        float max_val = scores[i * S];
        for (int j = 1; j < S; ++j) {
            if (scores[i * S + j] > max_val) max_val = scores[i * S + j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < S; ++j) {
            float e = expf(scores[i * S + j] - max_val);
            probs[i * S + j] = e;
            sum_exp += e;
        }
        for (int j = 0; j < S; ++j) {
            probs[i * S + j] /= sum_exp;
        }
    }

    // Step 3: O = P @ V
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < D; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < S; ++k) {
                sum += probs[i * S + k] * V[k * D + j];
            }
            output[i * D + j] = sum;
        }
    }

    delete[] scores;
    delete[] probs;
}

// ============================================================================
// 2. Warp Reduce Primitives（复用 Week 3 的标准件）
// ============================================================================

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) val = other;
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// 3. GPU Kernel: Fused Attention（核心）
// ============================================================================

/**
 * @brief 完整 Fused Attention Kernel — Online Softmax
 *
 * 配置:
 *   每个 block 处理一个 query → gridDim = S
 *   TILE_KV = 32（K/V 每次加载 32 行，一个 warp 的行数）
 *   blockDim.x = 256（8 warps）
 *
 * 线程分工:
 *   - 每个 warp 负责 TILE_KV 个 K/V 位置中的一个或几个
 *   - 一个 warp 内 32 个 lane 并行做点积 Q·K[pos]（stride loop 覆盖 D 维度）
 *   - 然后 warp reduce sum 得到 score
 *   - 跨 warp 协作: warp reduce max/sum 收集 tile 内的全局 max 和 sum
 *
 * Shared memory layout:
 *   q_shared   [D]            — 当前 query 向量
 *   k_shared   [TILE_KV * D]  — K 的 tile
 *   v_shared   [TILE_KV * D]  — V 的 tile
 *   scores_tile [TILE_KV]     — 当前 tile 的 attention scores（公共读写）
 *   out_acc    [D]            — 输出累加器（最终 /= running_sum）
 *
 * Online Softmax 流程（每个 tile）:
 *   scores = Q @ K_tile^T / sqrt(D)
 *   new_max = max(running_max, max(scores))
 *   rescale = exp(running_max - new_max)
 *
 *   out_acc    *= rescale         ← 旧的输出等比缩小
 *   running_sum *= rescale        ← 旧的 exp 和等比缩小
 *   running_max = new_max
 *
 *   exp_scores = exp(scores - new_max)
 *   out_acc    += exp_scores @ V_tile   ← 新 tile 的贡献（分子）
 *   running_sum += sum(exp_scores)
 *
 *   最终: output = out_acc / running_sum
 */
__global__ void attention_fused_kernel(const float *Q, const float *K,
                                        const float *V, float *output,
                                        int S, int D) {
    const int TILE_KV = 32;

    extern __shared__ float smem[];
    float *q_shared    = smem;                                    // [D]
    float *k_shared    = smem + D;                                // [TILE_KV * D]
    float *v_shared    = smem + D + TILE_KV * D;                  // [TILE_KV * D]
    float *scores_tile = smem + D + 2 * TILE_KV * D;              // [TILE_KV]
    float *out_acc     = smem + D + 2 * TILE_KV * D + TILE_KV;   // [D]

    int q_idx     = blockIdx.x;       // 处理第几个 query
    int tid       = threadIdx.x;      // 0 .. 255
    int warp_id   = tid / 32;         // 0 .. 7
    int lane_id   = tid % 32;         // 0 .. 31
    int num_warps = blockDim.x / 32;  // = 8

    if (q_idx >= S) return;

    float scale = 1.0f / sqrtf((float)D);
    int num_tiles = (S + TILE_KV - 1) / TILE_KV;

    // ---- 加载 Q[q_idx] 到 shared memory ----
    for (int d = tid; d < D; d += blockDim.x) {
        q_shared[d] = Q[q_idx * D + d];
    }

    // ---- Output 累加器清零 ----
    for (int d = tid; d < D; d += blockDim.x) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();

    // ---- Online softmax 状态 ----
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // ---- Tile 遍历 K 和 V ----
    for (int tile = 0; tile < num_tiles; ++tile) {
        int tile_start = tile * TILE_KV;

        // ----------------------------------------------------------
        // Step A: 加载 K_tile 和 V_tile 到 shared memory
        //         每个线程加载几个元素，确保合并访问（连续地址）
        // ----------------------------------------------------------
        int total = TILE_KV * D;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            int t = idx / D;     // K/V 行号 (0..TILE_KV-1)
            int col = idx % D;   // 列号
            int row = tile_start + t;
            if (row < S) {
                k_shared[idx] = K[row * D + col];
                v_shared[idx] = V[row * D + col];
            } else {
                k_shared[idx] = 0.0f;
                v_shared[idx] = 0.0f;
            }
        }
        __syncthreads();

        // ----------------------------------------------------------
        // Step B: 计算这个 tile 的每个 score
        //         每个 warp 负责 ceiling(TILE_KV / num_warps) 个位置
        //         一个 warp 内部: 32 个 lane 并行做点积 → warp reduce sum
        // ----------------------------------------------------------
        for (int pos = warp_id; pos < TILE_KV; pos += num_warps) {
            int row = tile_start + pos;
            if (row >= S) {
                // 无效位置，填入 -INFINITY（不会影响 max）
                if (lane_id == 0) scores_tile[pos] = -INFINITY;
                continue;
            }

            // 点积 q_shared · k_shared[pos]
            // 每个 lane 负责 D/32 个维度的乘积累加
            float dot = 0.0f;
            for (int d = lane_id; d < D; d += 32) {
                dot += q_shared[d] * k_shared[pos * D + d];
            }
            dot = warp_reduce_sum(dot);   // warp 内归约 → lane 0 有完整点积
            if (lane_id == 0) {
                scores_tile[pos] = dot * scale;  // 缩放到 shared memory
            }
        }
        __syncthreads();  // 确保 scores_tile[] 全部写完

        // ----------------------------------------------------------
        // Step C: 找这个 tile 的最大 score
        //         先每个 warp 找自己的 max，再跨 warp 汇总
        // ----------------------------------------------------------

        // C1. 每个线程扫描自己负责的 scores_tile 元素
        float local_max = -INFINITY;
        for (int pos = tid; pos < TILE_KV; pos += blockDim.x) {
            int row = tile_start + pos;
            if (row < S && scores_tile[pos] > local_max) {
                local_max = scores_tile[pos];
            }
        }

        // C2. Warp 内归约 → 每个 warp 的 lane 0 持有该 warp 的 max
        float warp_max = warp_reduce_max(local_max);

        // C3. 存到 k_shared[warp_id]（临时借用 k_shared 头部）
        if (lane_id == 0) {
            k_shared[warp_id] = warp_max;
        }
        __syncthreads();

        // C4. Warp 0 归约所有 warp 的 max → 全局 tile_max
        float tile_max = -INFINITY;
        if (warp_id == 0) {
            if (lane_id < num_warps) {
                tile_max = k_shared[lane_id];
            }
            tile_max = warp_reduce_max(tile_max);
        }
        tile_max = __shfl_sync(0xffffffff, tile_max, 0);  // broadcast

        // ----------------------------------------------------------
        // Step D: Online Softmax Update
        // ----------------------------------------------------------
        float new_max = (tile_max > running_max) ? tile_max : running_max;
        float rescale = expf(running_max - new_max);

        // D1. Rescale 旧的 out_acc 和 running_sum
        for (int d = tid; d < D; d += blockDim.x) {
            out_acc[d] *= rescale;
        }
        __syncthreads();  // 确保所有 rescale 完成后再读 out_acc

        running_sum *= rescale;
        running_max = new_max;

        // D2. 算 exp(scores - new_max)，存入 scores_tile，同时累加 tile_sum_exp
        float local_sum_exp = 0.0f;
        for (int pos = tid; pos < TILE_KV; pos += blockDim.x) {
            int row = tile_start + pos;
            if (row < S) {
                float e = expf(scores_tile[pos] - new_max);
                scores_tile[pos] = e;  // 覆写为 attention weight（分子）
                local_sum_exp += e;
            } else {
                scores_tile[pos] = 0.0f;
            }
        }

        // D3. 归约 tile 的 sum(exp)
        float warp_exp_sum = warp_reduce_sum(local_sum_exp);
        if (lane_id == 0) {
            k_shared[warp_id] = warp_exp_sum;  // 复用 k_shared 头部
        }
        __syncthreads();

        float tile_sum_exp = 0.0f;
        if (warp_id == 0) {
            if (lane_id < num_warps) {
                tile_sum_exp = k_shared[lane_id];
            }
            tile_sum_exp = warp_reduce_sum(tile_sum_exp);
        }
        tile_sum_exp = __shfl_sync(0xffffffff, tile_sum_exp, 0);
        running_sum += tile_sum_exp;

        // ----------------------------------------------------------
        // Step E: 累加 V 的加权贡献
        //         out_acc[d] += Σ_pos scores_tile[pos] * v_shared[pos][d]
        // ----------------------------------------------------------
        for (int pos = 0; pos < TILE_KV; ++pos) {
            int row = tile_start + pos;
            if (row >= S) continue;
            float weight = scores_tile[pos];  // = exp(score - new_max)
            for (int d = tid; d < D; d += blockDim.x) {
                out_acc[d] += weight * v_shared[pos * D + d];
            }
        }
        __syncthreads();
    }

    // ---- 最终归一化: output = out_acc / running_sum ----
    for (int d = tid; d < D; d += blockDim.x) {
        output[q_idx * D + d] = out_acc[d] / running_sum;
    }
}

// ============================================================================
// 4. 验证 + 工具函数
// ============================================================================

bool verify(const float *cpu, const float *gpu, int n, float eps = 1e-3f) {
    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = fabsf(cpu[i] - gpu[i]);
        if (err > max_err) max_err = err;
        if (err > eps) {
            if (mismatches < 5) {
                fprintf(stderr, "  Mismatch[%d]: CPU=%.6f GPU=%.6f diff=%.6f\n",
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
    int S = (argc > 1) ? atoi(argv[1]) : 128;
    int D = (argc > 2) ? atoi(argv[2]) : 64;

    printf("========================================\n");
    printf("CUDA Week 4: Fused Attention\n");
    printf("========================================\n");
    printf("Dimensions: [%d × %d] → Q,K,V ≡ S×D\n", S, D);
    printf("Attention matrix [S×S]: %d × %d = %.2f KB\n",
           S, S, (float)(S * S * sizeof(float)) / 1024.0f);

    int n_qkv = S * D;
    int n_out = S * D;
    size_t bytes_qkv = n_qkv * sizeof(float);
    size_t bytes_out = n_out * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ---- Alloc host ----
    float *h_Q = new float[n_qkv];
    float *h_K = new float[n_qkv];
    float *h_V = new float[n_qkv];
    float *h_cpu = new float[n_out];
    float *h_gpu = new float[n_out];

    srand(42);
    for (int i = 0; i < n_qkv; ++i) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // ---- Alloc device ----
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_K, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_V, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_O, bytes_out));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes_qkv, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ================================================================
    // Test 1: CPU
    // ================================================================
    printf("Running CPU... ");
    fflush(stdout);
    auto t0 = std::chrono::high_resolution_clock::now();
    attention_cpu(h_Q, h_K, h_V, h_cpu, S, D);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("%.2f ms\n\n", cpu_ms);

    // ================================================================
    // Test 2: GPU Fused Attention
    // ================================================================
    int smem_size = (D + 2 * 32 * D + 32 + D) * sizeof(float);
    // block_dim 不依赖 D——kernel 内部用 stride loop 处理任意 D
    int block_dim = 256;

    printf("Running GPU (fused attention)...\n");
    printf("  Block dim:          %d (%d warps)\n", block_dim, block_dim / 32);
    printf("  Shared memory:      %.2f KB\n", smem_size / 1024.0f);
    printf("  Max shared/blk:     %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    fflush(stdout);

    CUDA_CHECK(cudaEventRecord(start));
    attention_fused_kernel<<<S, block_dim, smem_size>>>(
        d_Q, d_K, d_V, d_O, S, D);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    printf("  Time: %.2f ms", gpu_ms);
    if (gpu_ms > 0 && cpu_ms > 0)
        printf("  (%.2fx vs CPU)", cpu_ms / gpu_ms);
    printf("\n");

    CUDA_CHECK(cudaMemcpy(h_gpu, d_O, bytes_out, cudaMemcpyDeviceToHost));
    printf("  Verification: %s\n\n",
           verify(h_cpu, h_gpu, n_out) ? "PASSED ✓" : "FAILED ✗");

    // ================================================================
    // Summary
    // ================================================================
    printf("========== Results ==========\n");
    printf("%-25s %8s  %s\n", "Version", "Time", "vs CPU");
    printf("%-25s %6.2f ms  %7s\n", "CPU (3-step)", cpu_ms, "—");
    printf("%-25s %6.2f ms  %6.2fx\n",
           "GPU (fused)", gpu_ms, cpu_ms / gpu_ms);

    printf("\n--- Key Takeaways ---\n");
    printf("1. Fused kernel avoids materializing [%d×%d] attention matrix\n",
           S, S);
    printf("   in GPU global memory — scores stay in registers + smem.\n");
    printf("2. Online softmax rescales old partial output when a larger\n");
    printf("   score is found, avoiding a second K/V scan.\n");
    printf("3. Building blocks combined:\n");
    printf("   - Tiled global memory loads (Week 2)\n");
    printf("   - Warp reduce sum + max (Week 3 softmax)\n");
    printf("   - Shared memory as cross-warp communication (Week 3)\n");

    // ---- Cleanup ----
    delete[] h_Q; delete[] h_K; delete[] h_V;
    delete[] h_cpu; delete[] h_gpu;
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
