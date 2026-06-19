/**
 * @file end_to_end.cu
 * @brief Phase 3 Day 5: 端到端推理 — Prefill + Decode + PagedAttention 集成
 *
 * 目标: 把 Day 2/3 的 PagedAttention 和 Phase 2 的 Prefill/Decode 流程
 *       焊在一起，跑一次完整的自回归生成。
 *
 * 三个核心函数:
 *   prefill_step  — 一次性处理所有 prompt tokens，缓存 K/V
 *   decode_step   — 每次生成 1 个 token，查 paged KV cache
 *   generate      — 串联 prefill + 若干次 decode 的自回归循环
 *
 * 编译:
 *   g++ -std=c++17 -O2 -x c++ end_to_end.cu -o end_to_end && ./end_to_end
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

// ============================================================================
// 常量
// ============================================================================
constexpr int BLOCK_SIZE            = 16;   // 每个物理 block 的 slot 数
constexpr int MAX_LOGICAL_BLOCKS    = 8;    // 每个请求最多 8 个逻辑块
constexpr int TOTAL_PHYSICAL_BLOCKS = 16;   // 物理 block 总数
constexpr int MAX_REQUESTS          = 4;    // 最多同时请求数
constexpr int H                     = 2;    // head 数
constexpr int D                     = 4;    // head 维度（小值，方便手算验证）
constexpr int D_MODEL               = H * D;// 模型维度 = 8

// ============================================================================
// Part 1: BlockTable
// ============================================================================
// 逻辑→物理地址翻译。物理 block 不连续，但通过 BlockTable 对上层呈现
// 为连续的"逻辑序列"。
struct BlockTable {
    int physical_blocks[MAX_LOGICAL_BLOCKS];
    int num_blocks;
    bool active;

    void init() {
        num_blocks = 0;
        active = false;
        for (int i = 0; i < MAX_LOGICAL_BLOCKS; ++i) physical_blocks[i] = -1;
    }

    // 逻辑位置 pos → (物理 block, block 内偏移)
    void translate(int pos, int *out_phys_blk, int *out_offset) const {
        int logical_block = pos / BLOCK_SIZE;
        *out_phys_blk = physical_blocks[logical_block];
        *out_offset    = pos % BLOCK_SIZE;
    }
};

// ============================================================================
// Part 2: PagedKVCache
// ============================================================================
// 分页 KV Cache 管理器，支持多请求并行。存储布局:
//   K_pool[phys_blk * H * BLOCK_SIZE * D + head * BLOCK_SIZE * D + offset * D]
class PagedKVCache {
private:
    float *K_pool;
    float *V_pool;
    bool   block_free[TOTAL_PHYSICAL_BLOCKS];
    BlockTable tables[MAX_REQUESTS];

public:
    void init() {
        int total = TOTAL_PHYSICAL_BLOCKS * H * BLOCK_SIZE * D;
        K_pool = new float[total]();
        V_pool = new float[total]();
        for (int i = 0; i < TOTAL_PHYSICAL_BLOCKS; ++i) block_free[i] = true;
        for (int i = 0; i < MAX_REQUESTS;       ++i) tables[i].init();
    }
    ~PagedKVCache() { delete[] K_pool; delete[] V_pool; }

    // 分配 num_blocks 个物理 block，返回请求 ID (rid)
    int allocate(int num_blocks) {
        int rid = 0;
        while (rid < MAX_REQUESTS && tables[rid].active) ++rid;
        if (rid == MAX_REQUESTS) return -1;

        int found = 0;
        for (int pb = 0; pb < TOTAL_PHYSICAL_BLOCKS && found < num_blocks; ++pb) {
            if (block_free[pb]) {
                tables[rid].physical_blocks[found++] = pb;
                block_free[pb] = false;
            }
        }
        if (found < num_blocks) {  // 不够，回滚
            for (int i = 0; i < found; ++i)
                block_free[tables[rid].physical_blocks[i]] = true;
            return -1;
        }
        tables[rid].num_blocks = num_blocks;
        tables[rid].active = true;
        return rid;
    }

    void free(int rid) {
        for (int i = 0; i < tables[rid].num_blocks; ++i)
            block_free[tables[rid].physical_blocks[i]] = true;
        tables[rid].init();
    }

    // 存储一个 K/V 向量到 (rid, head, pos)
    void store(int rid, int head, int pos, const float *k_vec, const float *v_vec) {
        int phys_blk, offset;
        tables[rid].translate(pos, &phys_blk, &offset);
        int base = phys_blk * H * BLOCK_SIZE * D + head * BLOCK_SIZE * D + offset * D;
        memcpy(K_pool + base, k_vec, D * sizeof(float));
        memcpy(V_pool + base, v_vec, D * sizeof(float));
    }

    // 读取一个 K/V 向量
    void load(int rid, int head, int pos, float *k_vec, float *v_vec) const {
        int phys_blk, offset;
        tables[rid].translate(pos, &phys_blk, &offset);
        int base = phys_blk * H * BLOCK_SIZE * D + head * BLOCK_SIZE * D + offset * D;
        memcpy(k_vec, K_pool + base, D * sizeof(float));
        memcpy(v_vec, V_pool + base, D * sizeof(float));
    }
};

// ============================================================================
// Part 3: 微型模型权重
// ============================================================================
// Wq/Wk/Wv/Wo: 每个都是 [D_MODEL][D_MODEL] 矩阵，存在一维数组里

float Wq[D_MODEL * D_MODEL];
float Wk[D_MODEL * D_MODEL];
float Wv[D_MODEL * D_MODEL];
float Wo[D_MODEL * D_MODEL];

void init_weights() {
    srand(42);
    for (int i = 0; i < D_MODEL * D_MODEL; ++i) {
        Wq[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;
        Wk[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;
        Wv[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;
        Wo[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

// 线性投影: out = x × W,   x: [D_MODEL], W: [D_MODEL][D_MODEL] 行主序
void linear(const float *x, const float *W, float *out) {
    memset(out, 0, D_MODEL * sizeof(float));
    for (int j = 0; j < D_MODEL; ++j)
        for (int i = 0; i < D_MODEL; ++i)
            out[j] += x[i] * W[i * D_MODEL + j];
}

// Softmax: 对长度为 len 的数组原地做 safe softmax (减最大值防溢出)
void softmax_inplace(float *x, int len) {
    float max_val = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > max_val) max_val = x[i];
    float sum = 0;
    for (int i = 0; i < len; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < len; ++i) x[i] /= sum;
}

// 返回最大值的索引（贪婪采样）
int argmax(const float *x, int len) {
    int best = 0;
    for (int i = 1; i < len; ++i) if (x[i] > x[best]) best = i;
    return best;
}

// ============================================================================
// prefill_step
// ============================================================================
// 一次性处理所有 prompt token:
//   (1) 对每个 token 做 Q/K/V 投影，K/V 存入 paged KV Cache
//   (2) 用最后一个 token 的 Q，对所有已缓存位置做 paged attention
//   (3) 拼接多头输出 × Wo → last_hidden
//
// 只输出最后一位的 hidden state，因为自回归生成只需要它来预测下一个 token。

void prefill_step(PagedKVCache &cache, int rid,
                  const int *prompt, int prompt_len,
                  const float *embedding,
                  float *last_hidden) {
    // ---- 第 1 步: 投影所有 prompt token，K/V 存入 cache，保留 last token 的 Q ----
    float q_last[H][D];  // 最后一个 token 的 Q，按 head 拆分

    for (int t = 0; t < prompt_len; ++t) {
        // 查 embedding 表得到 token 向量 x [D_MODEL]
        const float *x = embedding + prompt[t] * D_MODEL;

        // Q/K/V 投影
        float q_full[D_MODEL], k_full[D_MODEL], v_full[D_MODEL];
        linear(x, Wq, q_full);
        linear(x, Wk, k_full);
        linear(x, Wv, v_full);

        // 按 head 拆分，K/V 存入 cache
        for (int h = 0; h < H; ++h) {
            float k_h[D], v_h[D];
            for (int d = 0; d < D; ++d) {
                k_h[d] = k_full[h * D + d];
                v_h[d] = v_full[h * D + d];
            }
            cache.store(rid, h, t, k_h, v_h);

            // 最后一个 token 的 Q 单独保留，供第 2 步 attention 用
            if (t == prompt_len - 1) {
                for (int d = 0; d < D; ++d)
                    q_last[h][d] = q_full[h * D + d];
            }
        }
    }

    // ---- 第 2 步: 用 Q_last 对所有 K/V 位置做 paged attention ----
    float context[D_MODEL];

    for (int h = 0; h < H; ++h) {
        // 计算 Q·K^T 分数
        float *scores = new float[prompt_len];
        for (int s = 0; s < prompt_len; ++s) {
            float k_s[D], v_s[D];
            cache.load(rid, h, s, k_s, v_s);
            float dot = 0;
            for (int d = 0; d < D; ++d)
                dot += q_last[h][d] * k_s[d];
            scores[s] = dot;
        }

        // softmax 归一化
        softmax_inplace(scores, prompt_len);

        // 加权求和: Σ scores[s] × V[s]
        float attn_h[D] = {0};
        for (int s = 0; s < prompt_len; ++s) {
            float k_s[D], v_s[D];
            cache.load(rid, h, s, k_s, v_s);
            for (int d = 0; d < D; ++d)
                attn_h[d] += scores[s] * v_s[d];
        }

        // 写入 context 对应 head 的位置
        for (int d = 0; d < D; ++d)
            context[h * D + d] = attn_h[d];

        delete[] scores;
    }

    // ---- 第 3 步: 输出投影 context × Wo → last_hidden ----
    linear(context, Wo, last_hidden);
}

// ============================================================================
// decode_step
// ============================================================================
// 每生成一个 token 时调用:
//   (1) 对新 token 的 embedding 做 Q/K/V 投影，K/V 存入 cache
//   (2) Q 对所有 cached_len+1 个位置做 paged attention
//   (3) 拼接多头 × Wo → output hidden state

void decode_step(PagedKVCache &cache, int rid,
                 const float *token_emb, int cached_len,
                 float *output) {
    // ---- 第 1 步: 投影新 token，K/V 存入 cache ----
    float q_full[D_MODEL], k_full[D_MODEL], v_full[D_MODEL];
    linear(token_emb, Wq, q_full);
    linear(token_emb, Wk, k_full);
    linear(token_emb, Wv, v_full);

    float q_per_head[H][D];
    for (int h = 0; h < H; ++h) {
        float k_h[D], v_h[D];
        for (int d = 0; d < D; ++d) {
            q_per_head[h][d] = q_full[h * D + d];
            k_h[d] = k_full[h * D + d];
            v_h[d] = v_full[h * D + d];
        }
        // 存入 cache 的位置 cached_len（逻辑位置，通过 BlockTable 翻译到物理）
        cache.store(rid, h, cached_len, k_h, v_h);
    }

    // ---- 第 2 步: Paged attention — Q 对 0..cached_len 的所有位置 ----
    // 注意: seq_len = cached_len + 1，因为刚存入的新 token 也在 attention 范围内
    float context[D_MODEL];
    int seq_len = cached_len + 1;

    for (int h = 0; h < H; ++h) {
        // 计算 Q·K^T 分数
        float *scores = new float[seq_len];
        for (int s = 0; s < seq_len; ++s) {
            float k_s[D], v_s[D];
            cache.load(rid, h, s, k_s, v_s);
            float dot = 0;
            for (int d = 0; d < D; ++d)
                dot += q_per_head[h][d] * k_s[d];
            scores[s] = dot;
        }

        // softmax 归一化
        softmax_inplace(scores, seq_len);

        // 加权求和: Σ scores[s] × V[s]
        float attn_h[D] = {0};
        for (int s = 0; s < seq_len; ++s) {
            float k_s[D], v_s[D];
            cache.load(rid, h, s, k_s, v_s);
            for (int d = 0; d < D; ++d)
                attn_h[d] += scores[s] * v_s[d];
        }

        // 写入 context
        for (int d = 0; d < D; ++d)
            context[h * D + d] = attn_h[d];

        delete[] scores;
    }

    // ---- 第 3 步: 输出投影 context × Wo → output ----
    linear(context, Wo, output);
}

// ============================================================================
// generate — 自回归生成循环
// ============================================================================
// 完整的推理流程:
//   1. 分配 KV Cache block（预分配 prompt_len + max_new_tokens 个位置）
//   2. prefill_step → 得到 last_hidden
//   3. 循环 max_new_tokens 次:
//      a. last_hidden × lm_head → logits
//      b. argmax 贪婪采样 → next_token
//      c. 存到 generated[]
//      d. next_token embedding → decode_step → 更新 last_hidden
//   4. 释放 KV Cache

void generate(PagedKVCache &cache,
              const int *prompt, int prompt_len,
              const float *embedding,
              const float *lm_head,
              int max_new_tokens, int vocab_size,
              int *generated, int &generated_count) {
    // ---- 第 1 步: 分配 KV Cache ----
    // 预分配足够的 block 来容纳 prompt + 所有新 token
    int total_tokens = prompt_len + max_new_tokens;
    int num_blocks   = (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int rid = cache.allocate(num_blocks);
    if (rid < 0) {
        printf("ERROR: 无法分配 KV Cache\n");
        generated_count = 0;
        return;
    }

    // ---- 第 2 步: Prefill ----
    // 一次性处理所有 prompt token，缓存 K/V，得到最后一个位置的 hidden state
    float last_hidden[D_MODEL];
    prefill_step(cache, rid, prompt, prompt_len, embedding, last_hidden);

    // ---- 第 3 步: 自回归生成循环 ----
    generated_count = 0;
    int current_len = prompt_len;  // 当前序列总长度

    for (int step = 0; step < max_new_tokens; ++step) {
        // (a) hidden state → logits（通过 lm_head 投影）
        float *logits = new float[vocab_size];
        linear(last_hidden, lm_head, logits);

        // (b) 贪婪采样：取 logits 最大的 token
        int next_token = argmax(logits, vocab_size);
        generated[generated_count++] = next_token;
        delete[] logits;

        // (c) 取 next_token 的 embedding，做 decode
        const float *tok_emb = embedding + next_token * D_MODEL;
        decode_step(cache, rid, tok_emb, current_len, last_hidden);
        current_len++;
    }

    // ---- 第 4 步: 释放 KV Cache ----
    cache.free(rid);
}

// ============================================================================
// main — 测试完整流程
// ============================================================================
int main() {
    printf("========================================\n");
    printf("Phase 3 Day 5: 端到端推理\n");
    printf("========================================\n");

    init_weights();

    constexpr int VOCAB_SIZE = 16;   // 极小词表
    constexpr int PROMPT_LEN = 3;    // prompt 长度
    constexpr int MAX_NEW    = 4;    // 最多生成 4 个 token

    // ---- 构造 embedding 表 [vocab][D_MODEL] ----
    float *embedding = new float[VOCAB_SIZE * D_MODEL];
    srand(100);
    for (int i = 0; i < VOCAB_SIZE * D_MODEL; ++i)
        embedding[i] = (float)rand() / RAND_MAX * 0.5f - 0.25f;

    // ---- 构造 lm_head [D_MODEL][vocab] ----
    float *lm_head = new float[D_MODEL * VOCAB_SIZE];
    for (int i = 0; i < D_MODEL * VOCAB_SIZE; ++i)
        lm_head[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;

    // ---- prompt: [2, 7, 1] ----
    int prompt[PROMPT_LEN] = {2, 7, 1};

    printf("\nPrompt tokens: [");
    for (int i = 0; i < PROMPT_LEN; ++i)
        printf("%d%s", prompt[i], i < PROMPT_LEN - 1 ? ", " : "");
    printf("]\n");

    // ---- 初始化 KV Cache ----
    PagedKVCache cache;
    cache.init();

    // ---- 生成 ----
    int generated[MAX_NEW];
    int generated_count = 0;
    generate(cache, prompt, PROMPT_LEN, embedding, lm_head,
             MAX_NEW, VOCAB_SIZE, generated, generated_count);

    // ---- 输出结果 ----
    printf("\nGenerated tokens (%d): [", generated_count);
    for (int i = 0; i < generated_count; ++i)
        printf("%d%s", generated[i], i < generated_count - 1 ? ", " : "");
    printf("]\n");

    printf("\n========================================\n");
    printf("Test completed.\n");
    printf("========================================\n");

    delete[] embedding;
    delete[] lm_head;
    return 0;
}
