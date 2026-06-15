/**
 * @file prefill.cu
 * @brief Phase 2 Week 1 Day 3: Prefill 阶段实现
 *
 * Prefill 做的事:
 *   1. 接收 prompt 的 Q, K, V（L 层 × H 头 × S 个 token × D 维）
 *   2. 逐层逐头: 将 K 和 V 存入 KV Cache
 *   3. 逐层逐头: 运行 attention(Q, K, V) → 输出
 *
 * 为什么先存再读而不是直接算 attention？
 *   因为 decode 阶段需要这些 K/V — 存进缓存是给后面用的。
 *   Prefill 本身可以直接用原始的 K/V 数组算 attention（不需要从 Cache 读），
 *   但写入 Cache 是必不可少的步骤。
 *
 * 编译 (纯 CPU):
 *   g++ -std=c++17 -O2 -x c++ prefill.cu -o prefill && ./prefill
 */

#include "kv_cache.h"
#include <cmath>
#include <cstring>
#include <cassert>

// ============================================================================
// 0. CPU Attention（复用 Phase 1 的三步走实现）
// ============================================================================
void attention_cpu(const float *Q, const float *K, const float *V,
                   float *output, int S, int D) {
    float scale = 1.0f / sqrtf((float)D);

    float *scores = new float[S * S];
    float *probs  = new float[S * S];

    // Step 1: S = Q @ K^T
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < D; ++k)
                dot += Q[i * D + k] * K[j * D + k];
            scores[i * S + j] = dot * scale;
        }
    }

    // Step 2: row-wise softmax
    for (int i = 0; i < S; ++i) {
        float max_val = scores[i * S];
        for (int j = 1; j < S; ++j)
            if (scores[i * S + j] > max_val) max_val = scores[i * S + j];

        float sum_exp = 0.0f;
        for (int j = 0; j < S; ++j) {
            float e = expf(scores[i * S + j] - max_val);
            probs[i * S + j] = e;
            sum_exp += e;
        }
        for (int j = 0; j < S; ++j)
            probs[i * S + j] /= sum_exp;
    }

    // Step 3: O = P @ V
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < D; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < S; ++k)
                sum += probs[i * S + k] * V[k * D + j];
            output[i * D + j] = sum;
        }
    }

    delete[] scores;
    delete[] probs;
}

// ============================================================================
// 1. 数据访问辅助 — 从扁平数组中提取某个 head 的数据块
// ============================================================================
//
// 输入 Q/K/V 的布局: [L][H][S][D] — 和 KV Cache 完全相同的布局
//
//   Q[layer][head] 是一个 [S × D] 的矩阵
//   在扁平数组中从偏移 (layer * H + head) * S * D 开始，连续 S*D 个 float
//
// 例如: L=2, H=3, S=4, D=2
//   Q[0] = [head0_data(8 floats)][head1_data(8)][head2_data(8)]
//   Q[1] = ...
//   Q[0][head1] 从 Q[8] 开始, 连续 8 个 float:
//     [Q_0,1,0,0  Q_0,1,0,1  Q_0,1,1,0  Q_0,1,1,1  Q_0,1,2,0  Q_0,1,2,1  Q_0,1,3,0  Q_0,1,3,1]
//       pos=0              pos=1              pos=2              pos=3
// ============================================================================

/**
 * @brief 获取 Q[layer][head] 的指针（单头的 [S×D] 矩阵起点）
 *
 * @param data   扁平化的 Q/K/V 数组 [L * H * S * D]
 * @param layer  层号
 * @param head   头号
 * @param H      每层 head 数
 * @param S      序列长度
 * @param D      head 维度
 * @return       指向 data[layer][head][0][0] 的指针
 */
inline const float* get_head_ptr(const float *data, int layer, int head,
                                  int H, int S, int D) {
    // TODO: 填偏移计算
    // 提示: 一层占的大小 = H * S * D
    //       一个头在一层中占的大小 = S * D
    return ((layer * H * S * D + head * S * D)+ data);
}

/**
 * @brief 获取 K[layer][head] 中第 pos 个 token 的指针
 *
 * 这个函数用于 store_kv 循环中提取单个 token 的 K/V 向量。
 *
 * @param k_data  扁平化的 K 数组 [L * H * S * D]
 * @param layer   层号
 * @param head    头号
 * @param pos     token 位置
 * @param H       每层 head 数
 * @param S       序列长度
 * @param D       head 维度
 * @return        指向 K[layer][head][pos][0] 的指针
 */
inline const float* get_token_k_ptr(const float *k_data, int layer, int head,
                                     int pos, int H, int S, int D) {
    // TODO: 填偏移计算 — 先跳到 head，再跳到 pos
    // 提示: 在 get_head_ptr 的基础上加 pos * D
    return ((layer * H * S * D + head * S * D + pos * D)+ k_data);
}

// ============================================================================
// 2. Prefill 主函数
// ============================================================================

/**
 * @brief 执行 prefill: 缓存 K/V + 运行 attention，得到每层的输出
 *
 * @param cache    KV Cache（会被写入）
 * @param Q        所有层的 Query  [L * H * S * D]
 * @param K        所有层的 Key    [L * H * S * D]
 * @param V        所有层的 Value  [L * H * S * D]
 * @param output   输出 [L * H * S * D] — attention 的结果
 * @param L        层数
 * @param H        每层 head 数
 * @param S        序列长度 (prompt 的 token 数)
 * @param D        head 维度
 */
void prefill(KVCache &cache,
             const float *Q, const float *K, const float *V,
             float *output,
             int L, int H, int S, int D) {

    printf("[Prefill] Processing prompt of %d tokens (%d layers × %d heads × dim=%d)\n",
           S, L, H, D);

    // ---- 逐层处理 ----
    for (int l = 0; l < L; ++l) {

        // ---- 逐头处理 ----
        for (int h = 0; h < H; ++h) {

            // --------------------------------------------------------------
            // Step A: 缓存 K 和 V（为 decode 阶段做准备）
            //         把当前 head 的每个 token 位置的 K/V 存入 Cache
            // --------------------------------------------------------------
            for (int pos = 0; pos < S; ++pos) {
                // TODO: 填下面两行 — 获取 K[l][h][pos] 和 V[l][h][pos] 的指针
                const float *k_ptr = get_token_k_ptr(K, l, h,
                                     pos, H, S, D);
                const float *v_ptr = get_token_k_ptr(V, l, h,
                                     pos, H, S, D);
                cache.store_kv(l, h, pos, k_ptr, v_ptr);
            }

            // --------------------------------------------------------------
            // Step B: 运行 attention
            //         Q[l][h] @ K[l][h]^T / √D → softmax → @ V[l][h]
            // --------------------------------------------------------------
            // TODO: 填下面三行 — 获取 Q[l][h], K[l][h], V[l][h] 的指针
            const float *q_head =get_head_ptr(Q, l, h,H,  S,  D) ;
            const float *k_head = get_head_ptr(K, l, h,H,  S,  D);
            const float *v_head =get_head_ptr(V, l, h,H,  S,  D);

            // 输出位置: output[l][h] 的 [S×D] 矩阵
            int out_offset = ((l * H + h) * S) * D;
            attention_cpu(q_head, k_head, v_head, output + out_offset, S, D);
        }
    }

    // ---- 推进序列长度 ----
    cache.advance(S);
}

// ============================================================================
// 2b. 验证函数 — 用 cache 中的数据重新算 attention，结果应一致
// ============================================================================

/**
 * @brief 从 KV Cache 中提取单个 head 的 K 矩阵，用原始 Q 算 attention
 *
 * 这个函数验证: cache 里存的 K/V 和原始输入的 K/V 一样。
 * 重新计算 attention: Q @ cached_K^T → softmax → @ cached_V
 * 与 prefill 的直接计算结果对比。
 */
bool verify_prefill_output(const KVCache &cache,
                           const float *Q,
                           const float *prefill_output,
                           int L, int H, int S, int D) {

    // 临时存储从 cache 提取的 K[head] 和 V[head]
    float *K_from_cache = new float[S * D];
    float *V_from_cache = new float[S * D];
    float *recomputed   = new float[S * D];

    bool all_ok = true;

    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {

            // 从 cache 中逐 token 提取 K 和 V
            for (int pos = 0; pos < S; ++pos) {
                const float *k_ptr = cache.get_k_ptr(l, h, pos);
                const float *v_ptr = cache.get_v_ptr(l, h, pos);
                memcpy(K_from_cache + pos * D, k_ptr, D * sizeof(float));
                memcpy(V_from_cache + pos * D, v_ptr, D * sizeof(float));
            }

            // 用原始 Q 和 cache 中的 K/V 重新算 attention
            int q_offset = ((l * H + h) * S) * D;
            attention_cpu(Q + q_offset, K_from_cache, V_from_cache,
                          recomputed, S, D);

            // 对比 prefill 的输出
            const float *expected = prefill_output + q_offset;
            for (int i = 0; i < S * D; ++i) {
                if (fabsf(recomputed[i] - expected[i]) > 1e-5f) {
                    printf("  FAIL: layer=%d head=%d idx=%d  expected=%.6f got=%.6f\n",
                           l, h, i, expected[i], recomputed[i]);
                    all_ok = false;
                    break;
                }
            }
        }
    }

    delete[] K_from_cache;
    delete[] V_from_cache;
    delete[] recomputed;
    return all_ok;
}

// ============================================================================
// 3. 测试
// ============================================================================

bool test_prefill_basic() {
    printf("--- Test 1: Prefill 2 layers × 2 heads × 4 tokens × dim=4 ---\n");

    int L = 2, H = 2, S = 4, D = 4;
    int total = L * H * S * D;

    // ---- 构造测试用的 Q, K, V ----
    float *Q = new float[total];
    float *K = new float[total];
    float *V = new float[total];
    float *output = new float[total];

    for (int i = 0; i < total; ++i) {
        Q[i] = (float)((i * 7 + 13) % 100) / 100.0f;  // 伪随机但可复现
        K[i] = (float)((i * 3 + 17) % 100) / 100.0f;
        V[i] = (float)((i * 11 + 5) % 100) / 100.0f;
    }

    // ---- 初始化 KV Cache ----
    KVCache cache(L, H, D, S);
    assert(cache.current_len() == 0);

    // ---- 执行 prefill ----
    prefill(cache, Q, K, V, output, L, H, S, D);

    // ---- 验证 ----
    assert(cache.current_len() == S);

    // 抽查几个 cache 位置：K[0][0][2] 应等于原始 K 中对应位置
    {
        const float *cached_k = cache.get_k_ptr(0, 0, 2);
        const float *original_k = K + ((0 * H + 0) * S + 2) * D;
        for (int d = 0; d < D; ++d) {
            assert(cached_k[d] == original_k[d]);
        }
    }

    // 验证输出: 用 cache 里的 K/V 重算 attention，结果应一致
    bool ok = verify_prefill_output(cache, Q, output, L, H, S, D);

    printf("  Test 1: %s\n\n", ok ? "PASSED" : "FAILED");

    delete[] Q; delete[] K; delete[] V; delete[] output;
    return ok;
}

bool test_prefill_single_token() {
    printf("--- Test 2: Prefill 单 token 边界 (S=1) ---\n");

    int L = 1, H = 1, S = 1, D = 8;
    int total = L * H * S * D;

    float *Q = new float[total];
    float *K = new float[total];
    float *V = new float[total];
    float *output = new float[total];

    for (int i = 0; i < total; ++i) {
        Q[i] = (float)(i + 1) / 10.0f;
        K[i] = (float)(i + 1) / 10.0f;
        V[i] = (float)((i + 1) * 2) / 10.0f;
    }

    KVCache cache(L, H, D, S);
    prefill(cache, Q, K, V, output, L, H, S, D);

    bool ok = verify_prefill_output(cache, Q, output, L, H, S, D);
    printf("  Test 2: %s\n\n", ok ? "PASSED" : "FAILED");

    delete[] Q; delete[] K; delete[] V; delete[] output;
    return ok;
}

// ============================================================================
// 4. Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Phase 2 Day 3: Prefill Phase\n");
    printf("========================================\n\n");

    int passed = 0;

    if (test_prefill_basic())         passed++;
    if (test_prefill_single_token())  passed++;

    printf("========================================\n");
    printf("Result: %d/2 tests passed\n", passed);
    printf("========================================\n");

    return (passed == 2) ? 0 : 1;
}
