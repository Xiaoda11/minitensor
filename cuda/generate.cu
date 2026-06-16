/**
 * @file generate.cu
 * @brief Phase 2 Week 1 Day 5: 完整 generate() 循环 — Prefill + Decode 串联
 *
 * Day 3 (prefill) + Day 4 (decode) 各自是独立模块。
 * 今天把它们拼成一个完整的自回归生成循环。
 *
 * 自回归生成流程:
 *
 *   User prompt: "今天天气" (3 tokens)
 *        │
 *        ▼
 *   ┌─────────────────────────────────────────┐
 *   │  Step 1: Prefill                        │
 *   │  - 输入: Q, K, V [L×H×S_prompt×D]       │
 *   │  - 缓存 K/V 到 KV Cache                 │
 *   │  - 算 [S×S] attention → 第一个输出      │
 *   │  - cache.current_len = S_prompt         │
 *   └─────────────────────────────────────────┘
 *        │
 *        ▼
 *   ┌─────────────────────────────────────────┐
 *   │  Step 2: Decode loop (重复 N 次)        │
 *   │  for step in 0..max_new_tokens-1:       │
 *   │    - 构造新 token 的 Q_new, K_new, V_new │
 *   │    - decode() → attention 输出 [L×H×D]  │
 *   │    - 缓存新 token 的 K/V                │
 *   │    - cache.advance(1)                   │
 *   │    - 用输出选下一个 token（简化: 直接    │
 *   │      用 attention 输出模拟 logits）      │
 *   └─────────────────────────────────────────┘
 *
 * 今天学什么:
 *   1. 理解 prefill → decode 的控制流
 *   2. current_len 如何随生成推进
 *   3. 每次 encode 新 token 后，缓存如何增长
 *
 * 编译:
 *   g++ -std=c++17 -O2 -x c++ generate.cu -o generate && ./generate
 */

#include "kv_cache.h"
#include <cmath>
#include <cstring>
#include <cassert>

// ============================================================================
// 0. 依赖函数 — 从 Day 3 (prefill.cu) 和 Day 4 (decode.cu) 移植
//    (保持每个 .cu 文件独立编译)
// ============================================================================

// ---- 0a. CPU Attention（完整版 — 用于 prefill）----
void attention_cpu(const float *Q, const float *K, const float *V,
                   float *output, int S, int D) {
    float scale = 1.0f / sqrtf((float)D);

    float *scores = new float[S * S];
    float *probs  = new float[S * S];

    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < D; ++k)
                dot += Q[i * D + k] * K[j * D + k];
            scores[i * S + j] = dot * scale;
        }
    }

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

// ---- 0b. 单 token Attention（简化版 — 用于 decode）----
void decode_attention_cpu(const float *Q, const float *K, const float *V,
                           float *output, int S, int D) {
    float scale = 1.0f / sqrtf((float)D);

    float *scores = new float[S];
    for (int j = 0; j < S; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[d] * K[j * D + d];
        }
        scores[j] = dot * scale;
    }

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

// ---- 0c. 数据访问辅助 ----
inline const float* get_head_ptr(const float *data, int layer, int head,
                                  int H, int S, int D) {
    return data + layer * H * S * D + head * S * D;
}

inline const float* get_token_k_ptr(const float *k_data, int layer, int head,
                                     int pos, int H, int S, int D) {
    return k_data + layer * H * S * D + head * S * D + pos * D;
}

// ---- 0d. Prefill (Day 3) ----
void prefill(KVCache &cache,
             const float *Q, const float *K, const float *V,
             float *output,
             int L, int H, int S, int D) {

    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            // Step A: 缓存 K/V
            for (int pos = 0; pos < S; ++pos) {
                const float *k_ptr = get_token_k_ptr(K, l, h, pos, H, S, D);
                const float *v_ptr = get_token_k_ptr(V, l, h, pos, H, S, D);
                cache.store_kv(l, h, pos, k_ptr, v_ptr);
            }
            // Step B: attention
            const float *q_head = get_head_ptr(Q, l, h, H, S, D);
            const float *k_head = get_head_ptr(K, l, h, H, S, D);
            const float *v_head = get_head_ptr(V, l, h, H, S, D);
            int out_offset = ((l * H + h) * S) * D;
            attention_cpu(q_head, k_head, v_head, output + out_offset, S, D);
        }
    }
    cache.advance(S);
}

// ---- 0e. Decode (Day 4) ----
void decode(KVCache &cache,
            const float *Q_new, const float *K_new, const float *V_new,
            float *output,
            int L, int H, int S_cache, int D) {

    float *K_head = new float[S_cache * D];
    float *V_head = new float[S_cache * D];

    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            // Step A: 从 cache 提取 K_head, V_head
            for (int pos = 0; pos < S_cache; ++pos) {
                const float *k_ptr = cache.get_k_ptr(l, h, pos);
                const float *v_ptr = cache.get_v_ptr(l, h, pos);
                memcpy(K_head + pos * D, k_ptr, D * sizeof(float));
                memcpy(V_head + pos * D, v_ptr, D * sizeof(float));
            }
            // Step B: 获取 Q_new
            const float *q_ptr = Q_new + (l * H + h) * D;
            // Step C: decode attention
            int out_offset = (l * H + h) * D;
            decode_attention_cpu(q_ptr, K_head, V_head, output + out_offset,
                                 S_cache, D);
            // Step D: 缓存新 token
            const float *k_new_ptr = K_new + (l * H + h) * D;
            const float *v_new_ptr = V_new + (l * H + h) * D;
            cache.store_kv(l, h, S_cache, k_new_ptr, v_new_ptr);
        }
    }
    cache.advance(1);
    delete[] K_head;
    delete[] V_head;
}

// ============================================================================
// 1. 模拟 "token 选择" — 从 attention 输出生成下一个 token 的 Q/K/V
// ============================================================================
//
// 真实推理中这一步是: attention → output projection → LM head → softmax →
// sample → embedding 查表 → 得 Q/K/V。
// 这里用简化: 把归一化后的 attention 输出直接映射成新 token 的 Q/K/V。
//

/**
 * @brief 简化版 "next token" 生成
 *
 * 输入: attention 的输出 [L * H * D]
 * 输出: 新 token 的 Q_new, K_new, V_new [L * H * D]
 *
 * 简化逻辑: Q_new[i] = output[i] * 0.5
 *          K_new[i] = output[i] * 0.3
 *          V_new[i] = output[i] * 0.2
 */
void next_token_from_output(const float *attn_output,
                            float *Q_new, float *K_new, float *V_new,
                            int total_elements) {
    for (int i = 0; i < total_elements; ++i) {
        float val = attn_output[i];
        Q_new[i] = val * 0.5f;
        K_new[i] = val * 0.3f;
        V_new[i] = val * 0.2f;
    }
}

// ============================================================================
// 2. generate() — 核心: Prefill + Decode Loop
// ============================================================================

/**
 * @brief 完整的自回归生成
 *
 * 流程:
 *   1. Prefill: 缓存 prompt 的 K/V + 算 attention
 *   2. 用 prefill 输出生成第一个新 token
 *   3. Decode loop: 每步算一次单 token attention
 *   4. 记录每一步的 "logits" (用 attention 输出的总和模拟)
 *
 * @param cache         KV Cache (初始为空)
 * @param prompt_Q/K/V  Prompt 的 Q/K/V [L * H * S_prompt * D]
 * @param L             层数
 * @param H             每层 head 数
 * @param S_prompt      Prompt token 数
 * @param D             Head 维度
 * @param max_new_tokens 最多生成多少个新 token
 * @param out_logits    输出: 每一步的 logit 值 [max_new_tokens]
 *                       每个值是该步 attention 输出所有元素的 sum
 *                       (真实场景应该是 LM head 的 logits，这里简化)
 */
void generate(KVCache &cache,
              const float *prompt_Q, const float *prompt_K, const float *prompt_V,
              int L, int H, int S_prompt, int D,
              int max_new_tokens,
              float *out_logits) {

    int per_layer = H * D;                // 每个 layer 的 Q_new/K_new/V_new 大小

    // ========================================================================
    // Step 1: Prefill — 缓存 prompt 的所有 K/V + 算 [S×S] attention
    // ========================================================================
    //
    // 输出 prefill_output 布局: [L][H][S_prompt][D]
    //   每个 (layer, head) 有 S_prompt × D 个元素
    //
    // TODO 1: 分配 prefill_output 并调用 prefill()
    // 提示: 总元素数 = L * H * S_prompt * D
    float *prefill_output =new float[L * H * S_prompt * D];   // TODO 1a: new float[???]

    // TODO 1b: 调用 prefill — 把 prompt 的 K/V 缓存下来 + 算 attention
    // prefill 签名: prefill(cache, Q, K, V, output, L, H, S, D)
    // ??? prefill(???);
    prefill(cache,prompt_Q, prompt_K, prompt_V, prefill_output, L, H, S_prompt, D);
    // ========================================================================
    // Step 2: 构造第一个新 token 的 Q/K/V
    // ========================================================================
    //
    // 从 prefill 输出中，取每个 (layer, head) 最后一个 token 的 [D] 向量
    // 作为该 (layer, head) 的新 token 输入
    //
    // 偏移公式: prefill_output[l][h][S_prompt-1]
    //   = (layer * H * S_prompt * D) + (head * S_prompt * D) + ((S_prompt-1) * D)
    //
    float *Q_new = new float[L * per_layer];
    float *K_new = new float[L * per_layer];
    float *V_new = new float[L * per_layer];

    // TODO 2: 遍历每个 (layer, head)，填偏移 + 复制
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            // TODO 2a: 计算 prefill_output[l][h][S_prompt-1] 的偏移
            // 提示: 一层 = H * S_prompt * D, 一个 head = S_prompt * D
            int src_offset = (l * H * S_prompt * D) + (h * S_prompt * D) + ((S_prompt-1) * D);
            int dst_base = (l * H + h) * D;

            // TODO 2b: 从 prefill_output 复制 D 个元素，映射到 Q_new/K_new/V_new
            // 映射系数: Q*=0.5, K*=0.3, V*=0.2
            for (int d = 0; d < D; ++d) {
                float val = prefill_output[d+src_offset];
                Q_new[d+dst_base] = val * 0.5f;
                K_new[d+dst_base] = val * 0.3f;
                V_new[d+dst_base] = val * 0.2f;
            }
        }
    }
    delete[] prefill_output;

    // ========================================================================
    // Step 3: Decode Loop — 逐 token 生成
    // ========================================================================
    //
    float *decode_output = new float[L * per_layer];

    for (int step = 0; step < max_new_tokens; ++step) {
        int S_before = cache.current_len();

        printf("[Generate] Step %d/%d: decoding 1 token (cached %d tokens)\n",
               step + 1, max_new_tokens, S_before);

        // --- 3a: 执行 decode ---
        // TODO 3: 调用 decode() 对新 token 做 attention
        // decode 签名: decode(cache, Q_new, K_new, V_new, output, L, H, S_cache, D)
        // ??? decode(???);
        decode(cache, Q_new, K_new, V_new,decode_output, L, H,S_before, D);
        // --- 3b: 记录 logit ---
        // 简化: attention 输出全体元素求和 → "logit 信号"
        // TODO 4: 计算 out_logits[step] = sum of all decode_output elements
        // ??? out_logits[step] = ???;
        float logit_sum = 0.0f;
        for (int i = 0; i < L * per_layer; ++i) {
            logit_sum += decode_output[i];
        }
        out_logits[step] = logit_sum;
        // --- 3c: 生成下一个 token 的 Q/K/V ---
        if (step < max_new_tokens - 1) {
            // TODO 5: 调用 next_token_from_output
            // 签名: next_token_from_output(attn_output, Q_new, K_new, V_new, total_elements)
            // ??? next_token_from_output(???);
            next_token_from_output(decode_output, Q_new, K_new, V_new, L * per_layer); 
        }
    }

    // ---- 清理 ----
    delete[] Q_new;
    delete[] K_new;
    delete[] V_new;
    delete[] decode_output;

    printf("[Generate] Done — generated %d tokens (cache has %d total)\n",
           max_new_tokens, cache.current_len());
}

// ============================================================================
// 3. 验证函数
// ============================================================================

/**
 * @brief 重新跑 generate，对比 logits 是否一致（确定性验证）
 */
bool verify_generate(const KVCache &original_cache,
                     const float *prompt_Q, const float *prompt_K, const float *prompt_V,
                     const float *expected_logits,
                     int L, int H, int S_prompt, int D,
                     int max_new_tokens) {

    KVCache verify_cache(L, H, D, original_cache.max_seq_len());

    float *verify_logits = new float[max_new_tokens];
    generate(verify_cache, prompt_Q, prompt_K, prompt_V,
             L, H, S_prompt, D, max_new_tokens, verify_logits);

    bool all_ok = true;
    for (int step = 0; step < max_new_tokens; ++step) {
        if (fabsf(verify_logits[step] - expected_logits[step]) > 1e-5f) {
            printf("  FAIL at step=%d: expected=%.6f got=%.6f\n",
                   step, expected_logits[step], verify_logits[step]);
            all_ok = false;
        }
    }

    delete[] verify_logits;
    return all_ok;
}

// ============================================================================
// 4. 测试
// ============================================================================

bool test_generate_basic() {
    printf("--- Test 1: Basic generate (prompt=3, new=2) ---\n");

    int L = 1, H = 2, D = 4, S_prompt = 3, max_new = 2;
    int S_max = 8;
    int total_prompt = L * H * S_prompt * D;

    float *prompt_Q = new float[total_prompt];
    float *prompt_K = new float[total_prompt];
    float *prompt_V = new float[total_prompt];

    for (int i = 0; i < total_prompt; ++i) {
        prompt_Q[i] = (float)((i * 7 + 3) % 50) / 100.0f;
        prompt_K[i] = (float)((i * 5 + 2) % 50) / 100.0f;
        prompt_V[i] = (float)((i * 3 + 1) % 50) / 100.0f;
    }

    KVCache cache(L, H, D, S_max);
    float *logits = new float[max_new];

    generate(cache, prompt_Q, prompt_K, prompt_V,
             L, H, S_prompt, D, max_new, logits);

    assert(cache.current_len() == S_prompt + max_new);

    bool ok = verify_generate(cache, prompt_Q, prompt_K, prompt_V,
                              logits, L, H, S_prompt, D, max_new);

    printf("  Test 1: %s\n\n", ok ? "PASSED" : "FAILED");

    delete[] prompt_Q; delete[] prompt_K; delete[] prompt_V;
    delete[] logits;
    return ok;
}

bool test_generate_minimal() {
    printf("--- Test 2: Minimal generate (prompt=1, new=1) ---\n");

    int L = 2, H = 1, D = 4, S_prompt = 1, max_new = 1;
    int S_max = 4;
    int total_prompt = L * H * S_prompt * D;

    float *prompt_Q = new float[total_prompt];
    float *prompt_K = new float[total_prompt];
    float *prompt_V = new float[total_prompt];

    for (int i = 0; i < total_prompt; ++i) {
        prompt_Q[i] = (float)(i + 1) / 10.0f;
        prompt_K[i] = (float)(i + 1) / 12.0f;
        prompt_V[i] = (float)(i + 1) / 15.0f;
    }

    KVCache cache(L, H, D, S_max);
    float *logits = new float[max_new];

    generate(cache, prompt_Q, prompt_K, prompt_V,
             L, H, S_prompt, D, max_new, logits);

    assert(cache.current_len() == 2);

    bool ok = verify_generate(cache, prompt_Q, prompt_K, prompt_V,
                              logits, L, H, S_prompt, D, max_new);

    printf("  Test 2: %s\n\n", ok ? "PASSED" : "FAILED");

    delete[] prompt_Q; delete[] prompt_K; delete[] prompt_V;
    delete[] logits;
    return ok;
}

bool test_generate_multistep() {
    printf("--- Test 3: Multi-step generate (prompt=2, new=4) ---\n");

    int L = 1, H = 1, D = 4, S_prompt = 2, max_new = 4;
    int S_max = 8;
    int total_prompt = L * H * S_prompt * D;

    float *prompt_Q = new float[total_prompt];
    float *prompt_K = new float[total_prompt];
    float *prompt_V = new float[total_prompt];

    for (int i = 0; i < total_prompt; ++i) {
        prompt_Q[i] = (float)((i * 11 + 7) % 30) / 20.0f;
        prompt_K[i] = (float)((i * 7 + 3) % 30) / 20.0f;
        prompt_V[i] = (float)((i * 5 + 1) % 30) / 20.0f;
    }

    KVCache cache(L, H, D, S_max);
    float *logits = new float[max_new];

    generate(cache, prompt_Q, prompt_K, prompt_V,
             L, H, S_prompt, D, max_new, logits);

    assert(cache.current_len() == S_prompt + max_new);

    bool ok = verify_generate(cache, prompt_Q, prompt_K, prompt_V,
                              logits, L, H, S_prompt, D, max_new);

    printf("  Test 3: %s\n\n", ok ? "PASSED" : "FAILED");

    delete[] prompt_Q; delete[] prompt_K; delete[] prompt_V;
    delete[] logits;
    return ok;
}

// ============================================================================
// 5. Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Phase 2 Day 5: Generate Loop\n");
    printf("========================================\n\n");

    int passed = 0;

    if (test_generate_basic())      passed++;
    if (test_generate_minimal())    passed++;
    if (test_generate_multistep())  passed++;

    printf("========================================\n");
    printf("Result: %d/3 tests passed\n", passed);
    printf("========================================\n");

    return (passed == 3) ? 0 : 1;
}
