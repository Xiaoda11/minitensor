/**
 * @file decode.cu
 * @brief Phase 2 Week 1 Day 4: Decode 阶段实现
 *
 * Decode 阶段 vs Prefill 阶段:
 *
 *   Prefill (Day 3):                       Decode (Day 4):
 *   ┌────────────────────────┐             ┌────────────────────────┐
 *   │ Q: [S_prompt × D]      │             │ Q: [1 × D]   ← 只一个新 token!
 *   │ K: [S_prompt × D]      │             │ K: [S_cache × D]  ← 从 KV Cache 读
 *   │ scores: [S × S]        │             │ scores: [1 × S_cache] ← 一个向量
 *   │ 计算量: O(S² × D)      │             │ 计算量: O(S × D)
 *   └────────────────────────┘             └────────────────────────┘
 *
 * 为什么 decode 更快:
 *   - 每次只生成 1 个新 token → Q 只有 1 行
 *   - K 和 V 是复用之前缓存的所有 token 的，不需要重新算
 *   - Attention 从矩阵×矩阵 变成 向量×矩阵
 *
 * 自回归生成循环:
 *   [BOS] → prefill(prompt) → decode → "我" → decode → "爱" → decode → ...
 *           缓存 K/V_0..S-1     +1 token 缓存        +1 token 缓存
 *
 * 编译 (纯 CPU):
 *   g++ -std=c++17 -O2 -x c++ decode.cu -o decode && ./decode
 */

#include "kv_cache.h"
#include <cmath>
#include <cstring>
#include <cassert>

// ============================================================================
// 0. 单 token Attention (CPU) — 用于 decode 阶段
// ============================================================================
//
// 和 prefill 的 attention_cpu 的区别:
//   prefill: Q[S×D] @ K[S×D]^T → scores[S×S]
//   decode:  Q[1×D] @ K[S×D]^T → scores[1×S]
//
// 数学上一样，但 decode 版更简单——Q 只有一行，所以没有外层 i 循环。
//
// 流程:
//   1. scores[j] = Q[0] · K[j] / √D               → S 次点积
//   2. softmax(scores) → probs[0..S-1]             → 标椎稳定 softmax
//   3. output[d] = Σ(j) probs[j] * V[j][d]         → S×D 次乘加

void decode_attention_cpu(const float *Q,     // [D]  — 单 token 的 query
                          const float *K,     // [S × D]  — 缓存的所有 key
                          const float *V,     // [S × D]  — 缓存的所有 value
                          float *output,      // [D]  — 单 token 的输出
                          int S, int D) {

    float scale = 1.0f / sqrtf((float)D);

    // ---- Step 1: scores = Q @ K^T (1×S 向量) ----
    float *scores = new float[S];
    for (int j = 0; j < S; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[d] * K[j * D + d];
        }
        scores[j] = dot * scale;
    }

    // ---- Step 2: softmax (标准三步: max → exp → normalize) ----
    float max_val = scores[0];
    for (int j = 1; j < S; ++j) {
        if (scores[j] > max_val) max_val = scores[j];
    }

    float *probs = new float[S];
    float sum_exp = 0.0f;
    for (int j = 0; j < S; ++j) {
        probs[j] = expf(scores[j] - max_val);
        sum_exp += probs[j];
    }
    for (int j = 0; j < S; ++j) {
        probs[j] /= sum_exp;
    }

    // ---- Step 3: output = probs @ V (1×D 向量) ----
    for (int d = 0; d < D; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < S; ++j) {
            sum += probs[j] * V[j * D + d];
        }
        output[d] = sum;
    }

    delete[] scores;
    delete[] probs;
}

// ============================================================================
// 1. Decode 主函数
// ============================================================================

/**
 * @brief 执行一次 decode 步: 用新 token 的 Q/K/V + 缓存的 K/V 算出输出
 *
 * 一次 decode 步做的事情:
 *   1. 从 KV Cache 中读取所有历史 K 和 V
 *   2. 对每个 (layer, head):
 *      a. 取 Q_new[layer][head] — 单 token 的 query 向量 [D]
 *      b. 取 K_cache[layer][head] — 缓存的所有历史的 key [S × D]
 *      c. 取 V_cache[layer][head] — 缓存的所有历史的 value [S × D]
 *      d. 运行 decode_attention_cpu(Q_new, K_cache, V_cache) → output [D]
 *      e. 把新 token 的 K/V 存入 KV Cache（为下一步 decode 做准备）
 *   3. 推进 cache.current_len()
 *
 * @param cache    KV Cache（已有 prefill 阶段缓存的 K/V，会被追加写入）
 * @param Q_new    新 token 所有层的 Query  [L * H * D]
 * @param K_new    新 token 所有层的 Key    [L * H * D]
 * @param V_new    新 token 所有层的 Value  [L * H * D]
 * @param output   输出 [L * H * D] — 每个 (layer, head) 的 attention 结果
 * @param L        层数
 * @param H        每层 head 数
 * @param S_cache  缓存的序列长度（即这一 step 开始时的 current_len）
 * @param D        head 维度
 */
void decode(KVCache &cache,
            const float *Q_new, const float *K_new, const float *V_new,
            float *output,
            int L, int H, int S_cache, int D) {

    printf("[Decode] Processing 1 token at pos=%d (cached %d tokens, %d layers × %d heads × dim=%d)\n",
           S_cache, S_cache, L, H, D);

    // 临时缓冲区：从 cache 提取一个 head 的完整 K/V 矩阵
     float *K_head = new float[S_cache * D];
     float *V_head = new float[S_cache * D];

    // ---- 逐层处理 ----
    for (int l = 0; l < L; ++l) {

        // ---- 逐头处理 ----
        for (int h = 0; h < H; ++h) {

            // ================================================================
            // Step A: 从 KV Cache 提取第 l 层第 h 个 head 的所有历史 K 和 V
            // ================================================================
            // 循环 S_cache 次，每次从 cache.get_k_ptr(l, h, pos) 读 D 个 float
            // 复制到 K_head[pos * D] 和 V_head[pos * D]
            //
            // 提示: cache.get_k_ptr(l, h, pos) 返回 const float*，指向 D 个连续 float
            //       用 memcpy(K_head + pos*D, ptr, D*sizeof(float)) 复制

            // TODO: 填下面循环体 (约 3-4 行)
            // 提示: cache.get_k_ptr(l,h,pos) 获取 K 指针, memcpy 到 K_head + pos*D
            //       cache.get_v_ptr(l,h,pos) 获取 V 指针, memcpy 到 V_head + pos*D
            for (int pos = 0; pos < S_cache; ++pos) {
                const  float *K_head_temp = cache.get_k_ptr(l,h,pos);
                const  float *v_head_temp = cache.get_v_ptr(l,h,pos);
                memcpy(K_head + pos*D, K_head_temp, D*sizeof(float));
                memcpy(V_head + pos*D, v_head_temp, D*sizeof(float));
            }

            // ================================================================
            // Step B: 获取新 token 的 Q — 单 token 向量 [D]
            // ================================================================
            // Q_new 布局: [L][H][D]
            //   每个 (layer, head) 有 D 个连续的 float
            //   偏移 = (layer * H + head) * D
            //
            // TODO: 填下面这行 — 获取 Q_new[l][h] 的指针 (单 token, 只有 D 个元素)
            const float *q_ptr = Q_new + ((l * H + h) * D);

            // ================================================================
            // Step C: 运行单 token attention
            // ================================================================
            // 填入 Q_ptr(单token), K_head(全历史), V_head(全历史), output位置, S_cache, D
            //
            // TODO: 填下面两行 — 输出位置偏移 和 decode_attention_cpu 调用
            int out_offset = (l * H + h) * D;
            decode_attention_cpu(q_ptr,K_head,V_head ,output+out_offset,S_cache,D);

            // ================================================================
            // Step D: 把新 token 的 K/V 存入 KV Cache
            // ================================================================
            // 注意: 新 token 要写到 pos = S_cache（最后一个位置的下一个）
            //       因为 prefill 阶段已经缓存了 pos 0..S_cache-1
            //
            // TODO: 填下面三行 — 获取 K_new[l][h], V_new[l][h] 的指针，然后 store_kv
            const float *k_new_ptr = K_new + (l * H + h) * D;
            const float *v_new_ptr = V_new + (l * H + h) * D;
            cache.store_kv(l, h, S_cache, k_new_ptr, v_new_ptr);
        }
    }

    // ---- 推进序列长度 ----
    cache.advance(1);

    delete[] K_head;
    delete[] V_head;
}

// ============================================================================
// 2. 验证函数
// ============================================================================

/**
 * @brief 验证 decode 输出: 手动重做同样计算，对比结果
 *
 * 这个验证比 prefill 的 verify 简单 — 不需要从 cache 提取完整的 K/V 矩阵
 * (上一步 decode 已经提取过了)，我们直接手工算一次相同的 attention。
 */
bool verify_decode_output(const KVCache &cache,
                          const float *Q_new,
                          const float *decode_output,
                          int L, int H, int S, int D) {

    float *K_manual = new float[S * D];
    float *V_manual = new float[S * D];
    float *expected  = new float[D];

    bool all_ok = true;

    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {

            // 手工从 cache 提取 K_head, V_head
            for (int pos = 0; pos < S; ++pos) {
                const float *k_ptr = cache.get_k_ptr(l, h, pos);
                const float *v_ptr = cache.get_v_ptr(l, h, pos);
                memcpy(K_manual + pos * D, k_ptr, D * sizeof(float));
                memcpy(V_manual + pos * D, v_ptr, D * sizeof(float));
            }

            // 用 Q_new 手工算 attention
            int q_offset = (l * H + h) * D;
            decode_attention_cpu(Q_new + q_offset, K_manual, V_manual,
                                 expected, S, D);

            // 对比 decode 输出
            int out_offset = (l * H + h) * D;
            const float *got = decode_output + out_offset;
            for (int d = 0; d < D; ++d) {
                if (fabsf(got[d] - expected[d]) > 1e-5f) {
                    printf("  FAIL: layer=%d head=%d dim=%d  expected=%.6f got=%.6f\n",
                           l, h, d, expected[d], got[d]);
                    all_ok = false;
                    break;
                }
            }
        }
    }

    delete[] K_manual;
    delete[] V_manual;
    delete[] expected;
    return all_ok;
}

// ============================================================================
// 3. 测试
// ============================================================================

/**
 * Test 1: 基本 decode — prefill 3 个 token → decode 1 个 token
 *
 * 场景: prompt="A B C" (3 tokens), 模型生成 "D" (1 token)
 *   Step 1: prefill 缓存 "A B C" 的 K/V
 *   Step 2: decode "D": 用 "D" 的 Q 对 "A B C" 的 K/V 做 attention
 */
bool test_decode_basic() {
    printf("--- Test 1: Decode after 3-token prefill ---\n");

    int L = 1, H = 2, D = 4, S_max = 8;
    int S_prompt = 3;

    KVCache cache(L, H, D, S_max);

    // ---- 模拟 prefill: 直接在 cache 里写入 3 个 token 的 K/V ----
    // (简化: 不调 prefill()，手动写 cache，只测 decode 逻辑)
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            for (int pos = 0; pos < S_prompt; ++pos) {
                float k[4], v[4];
                for (int d = 0; d < D; ++d) {
                    // 编码: K = pos*100 + h*10 + d, V = K + 1000
                    k[d] = pos * 100.0f + h * 10.0f + d;
                    v[d] = k[d] + 1000.0f;
                }
                cache.store_kv(l, h, pos, k, v);
            }
        }
    }
    cache.advance(S_prompt);
    assert(cache.current_len() == 3);

    // ---- 构造新 token "D" 的 Q_new, K_new, V_new ----
    int per_layer = H * D;  // 8
    float *Q_new = new float[L * per_layer];
    float *K_new = new float[L * per_layer];
    float *V_new = new float[L * per_layer];

    for (int i = 0; i < L * per_layer; ++i) {
        Q_new[i] = (float)((i * 7 + 3) % 50) / 100.0f;
        K_new[i] = (float)((i * 5 + 2) % 50) / 100.0f;
        V_new[i] = (float)((i * 3 + 1) % 50) / 100.0f;
    }

    float *output = new float[L * per_layer];

    // ---- 执行 decode ----
    int S_cache_before = cache.current_len();  // = 3
    decode(cache, Q_new, K_new, V_new, output, L, H, S_cache_before, D);

    // ---- 验证 ----
    assert(cache.current_len() == 4);  // prefill 3 + decode 1

    // 验证 cache 中新增的 K/V（pos=3）
    {
        const float *kc = cache.get_k_ptr(0, 0, 3);
        const float *vc = cache.get_v_ptr(0, 0, 3);
        for (int d = 0; d < D; ++d) {
            assert(kc[d] == K_new[d]);           // K_new[0][0][d]
            assert(vc[d] == V_new[d]);           // V_new[0][0][d]
        }
    }

    // 验证 attention 输出
    bool ok = verify_decode_output(cache, Q_new, output, L, H, S_cache_before, D);

    printf("  Test 1: %s\n\n", ok ? "PASSED" : "FAILED");

    delete[] Q_new; delete[] K_new; delete[] V_new; delete[] output;
    return ok;
}

/**
 * Test 2: decode 第一步（prefill 后紧跟第一个 decode）
 *
 * 边界: S_cache=2, 即 decode 时缓存中只有 2 个 token。
 * 验证 decode 在 prompt 很短时也能正确工作。
 */
bool test_decode_first_step() {
    printf("--- Test 2: Decode after 2-token prefill (short prompt) ---\n");

    int L = 2, H = 1, D = 4, S_max = 4;
    int S_prompt = 2;

    KVCache cache(L, H, D, S_max);

    // 模拟 prefill: 写入 2 个 token
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            for (int pos = 0; pos < S_prompt; ++pos) {
                float k[4], v[4];
                for (int d = 0; d < D; ++d) {
                    k[d] = (float)(l * 100 + pos * 10 + d + 1);
                    v[d] = k[d] * 2.0f;
                }
                cache.store_kv(l, h, pos, k, v);
            }
        }
    }
    cache.advance(S_prompt);

    // 新 token 的 Q/K/V
    int total = L * H * D;
    float *Q_new = new float[total];
    float *K_new = new float[total];
    float *V_new = new float[total];
    float *output = new float[total];

    for (int i = 0; i < total; ++i) {
        Q_new[i] = (float)(i + 1) / 20.0f;
        K_new[i] = (float)(i + 1) / 15.0f;
        V_new[i] = (float)(i + 1) / 10.0f;
    }

    int S_cache_before = cache.current_len();
    decode(cache, Q_new, K_new, V_new, output, L, H, S_cache_before, D);

    assert(cache.current_len() == 3);

    bool ok = verify_decode_output(cache, Q_new, output, L, H, S_cache_before, D);

    // 验证旧数据没被覆盖
    {
        float ek00[] = {1, 2, 3, 4};     // layer=0 pos=0: 0*100+0*10+d+1
        float ev00[] = {2, 4, 6, 8};     // V = K * 2
        ok = cache.verify_write_read(0, 0, 0, ek00, ev00) && ok;
    }

    printf("  Test 2: %s\n\n", ok ? "PASSED" : "FAILED");

    delete[] Q_new; delete[] K_new; delete[] V_new; delete[] output;
    return ok;
}

/**
 * Test 3: 多次连续 decode — 模拟生成 3 个 token
 *
 * 场景: prompt 1 token → decode → decode → decode
 * 验证 cache 的序列长度正确递增，每次产出正确的 attention 输出。
 */
bool test_decode_multiple_steps() {
    printf("--- Test 3: Multi-step decode (3 generation steps) ---\n");

    int L = 1, H = 1, D = 4, S_max = 8;
    KVCache cache(L, H, D, S_max);

    // 模拟 prefill: 写入 1 个 token（"BOS"）
    {
        float k[] = {1, 1, 1, 1};
        float v[] = {2, 2, 2, 2};
        cache.store_kv(0, 0, 0, k, v);
    }
    cache.advance(1);

    bool all_ok = true;

    // 连续 decode 3 步
    for (int step = 0; step < 3; ++step) {
        int S_before = cache.current_len();  // 1, 2, 3

        float Q_new[4], K_new[4], V_new[4], output[4];
        for (int d = 0; d < D; ++d) {
            Q_new[d] = (float)(step * 10 + d + 3) / 10.0f;
            K_new[d] = (float)(step * 10 + d + 5) / 10.0f;
            V_new[d] = (float)(step * 10 + d + 7) / 10.0f;
        }

        decode(cache, Q_new, K_new, V_new, output, L, H, S_before, D);

        // 验证序列长度
        assert(cache.current_len() == S_before + 1);

        // 验证新 token 的 K/V 正确存储
        const float *kc = cache.get_k_ptr(0, 0, S_before);
        const float *vc = cache.get_v_ptr(0, 0, S_before);
        for (int d = 0; d < D; ++d) {
            if (kc[d] != K_new[d] || vc[d] != V_new[d]) {
                printf("  FAIL at step=%d dim=%d: K(%f vs %f) V(%f vs %f)\n",
                       step, d, kc[d], K_new[d], vc[d], V_new[d]);
                all_ok = false;
            }
        }

        // 验证 attention 输出
        if (!verify_decode_output(cache, Q_new, output, L, H, S_before, D)) {
            printf("  FAIL at step=%d: attention output mismatch\n", step);
            all_ok = false;
        }
    }

    // 最终序列长度 = 1 (prefill) + 3 (decode) = 4
    assert(cache.current_len() == 4);

    printf("  Test 3: %s\n\n", all_ok ? "PASSED" : "FAILED");
    return all_ok;
}

// ============================================================================
// 4. Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Phase 2 Day 4: Decode Phase\n");
    printf("========================================\n\n");

    int passed = 0;

    if (test_decode_basic())           passed++;
    if (test_decode_first_step())      passed++;
    if (test_decode_multiple_steps())  passed++;

    printf("========================================\n");
    printf("Result: %d/3 tests passed\n", passed);
    printf("========================================\n");

    return (passed == 3) ? 0 : 1;
}
