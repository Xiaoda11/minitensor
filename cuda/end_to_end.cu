/**
 * @file end_to_end.cu
 * @brief Phase 3 Day 5: 端到端推理 — Prefill + Decode + PagedAttention 集成
 *
 * 目标: 把 Day 2/3 的 PagedAttention 和 Phase 2 的 Prefill/Decode 流程
 *       焊在一起，跑一次完整的自回归生成。
 *
 * 你将实现:
 *   TODO 1: prefill_step  — 一次性处理所有 prompt tokens，缓存 K/V
 *   TODO 2: decode_step   — 每次生成 1 个 token，查 paged KV cache
 *   TODO 3: generate      — 串联 prefill + 若干次 decode 的自回归循环
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
constexpr int BLOCK_SIZE           = 16;   // 每个物理 block 的 slot 数
constexpr int MAX_LOGICAL_BLOCKS   = 8;    // 每个请求最多 8 个逻辑块
constexpr int TOTAL_PHYSICAL_BLOCKS = 16;  // 物理 block 总数
constexpr int MAX_REQUESTS         = 4;    // 最多同时请求数
constexpr int H                    = 2;    // head 数
constexpr int D                    = 4;    // head 维度（小值，方便手算验证）
constexpr int D_MODEL              = H * D;// 模型维度 = 8

// ============================================================================
// Part 1: BlockTable（精简版，Day 2 已掌握）
// ============================================================================
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
// Part 2: PagedKVCache（精简版，Day 2 已掌握）
// ============================================================================
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

    // 分配 block，返回 rid
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
            for (int i = 0; i < found; ++i) block_free[tables[rid].physical_blocks[i]] = true;
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

    // 存储一个 K/V 向量
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

    // 读取某个 pos 的 K（不需要 V 时用）
    const float* get_K_ptr(int rid, int head, int pos) const {
        int phys_blk, offset;
        tables[rid].translate(pos, &phys_blk, &offset);
        return K_pool + (phys_blk * H * BLOCK_SIZE * D + head * BLOCK_SIZE * D + offset * D);
    }
};

// ============================================================================
// Part 3: 微型模型权重（固定值，不用填）
// ============================================================================
// Wq/Wk/Wv: [D_MODEL][D_MODEL]   Wo: [D_MODEL][D_MODEL]

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

// 线性投影: out = x × W,  x: [D_MODEL], W: [D_MODEL][D_MODEL]
void linear(const float *x, const float *W, float *out) {
    memset(out, 0, D_MODEL * sizeof(float));
    for (int j = 0; j < D_MODEL; ++j)
        for (int i = 0; i < D_MODEL; ++i)
            out[j] += x[i] * W[i * D_MODEL + j];
}

// Softmax: 对长度为 len 的数组原地做 safe softmax
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

// Argmax: 返回最大值的索引（简易采样）
int argmax(const float *x, int len) {
    int best = 0;
    for (int i = 1; i < len; ++i) if (x[i] > x[best]) best = i;
    return best;
}

// ============================================================================
// TODO 1: prefill_step
// ============================================================================
//
// 输入:
//   cache      已分配好 block 的 PagedKVCache（rid 已返回）
//   rid        请求 ID
//   prompt     输入 token id 序列
//   prompt_len prompt 长度
//   embedding  token embedding 表 [vocab][D_MODEL]
//   last_hidden 输出: prefill 完成后最后一个位置的 hidden state [D_MODEL]
//
// 要做什么:
//   (1) 遍历所有 prompt token，对每个 token 做 Q/K/V 投影，K/V 存入 cache
//   (2) 用最后一个 token 的 Q，对所有位置的 K/V 做 paged attention
//   (3) 拼接多头 attention 输出，投影 context × Wo → last_hidden
//
// 为什么只输出最后一位? 自回归推理只需要下一个 token，prefill 的中间
// 位置输出不需要保留——这就是 "prefill → single last hidden" 的语义。
//
void prefill_step(PagedKVCache &cache, int rid,
                  const int *prompt, int prompt_len,
                  const float *embedding,
                  float *last_hidden) {
    // ================================================================
    // 第 1 步: 投影并存储所有 prompt token 的 K/V
    // ================================================================
    //
    // 提示:
    //   - 对每个 token position t = 0 .. prompt_len-1:
    //       取 embedding[prompt[t]] → x[D_MODEL]
    //       用 linear() 分别投影 x → q_full, k_full, v_full
    //       对每个 head h: 从 k_full/v_full 切出 head-h 的 D 维向量
    //       调用 cache.store(rid, h, t, k_h, v_h)
    //   - 保存最后一个 token (t == prompt_len-1) 的 Q，按 head 拆分
    //     存到 q_last[h][0..D-1]，供第 2 步使用
    //
    // ??? float q_last[H][D];  // 最后一个 token 的 Q，按 head 拆分
    // ???
    // ??? for (int t = 0; t < prompt_len; ++t) {
    // ???     // 取 embedding
    // ???     const float *x = embedding + prompt[t] * D_MODEL;
    // ???
    // ???     // 投影 Q, K, V
    // ???     float q_full[D_MODEL], k_full[D_MODEL], v_full[D_MODEL];
    // ???     linear(x, Wq, q_full);
    // ???     linear(x, Wk, k_full);
    // ???     linear(x, Wv, v_full);
    // ???
    // ???     // 拆成 head-dim 向量
    // ???     for (int h = 0; h < H; ++h) {
    // ???         // K/V 存入 cache
    // ???         float k_h[D], v_h[D];
    // ???         // 从 k_full/v_full 中切出 head h 的部分
    // ???         for (int d = 0; d < D; ++d) {
    // ???             k_h[d] = k_full[h * D + d];
    // ???             v_h[d] = v_full[h * D + d];
    // ???         }
    // ???         cache.store(rid, h, t, k_h, v_h);
    // ???
    // ???         // 如果是最后一个 token，保存 Q
    // ???         if (t == prompt_len - 1) {
    // ???             for (int d = 0; d < D; ++d)
    // ???                 q_last[h][d] = q_full[h * D + d];
    // ???         }
    // ???     }
    // ??? }

    // --- 你的代码 ---


    // --- 你的代码结束 ---

    // ================================================================
    // 第 2 步: 计算 last token 对所有位置的 paged attention
    // ================================================================
    //
    // 对每个 head h:
    //   1. 遍历 s = 0 .. prompt_len-1, 从 cache 读 K[s] → 算 attention score
    //   2. softmax → probs
    //   3. 再次遍历 s，probs[s] × V[s] 累加 → attn_out[D]
    //
    // ??? float context[D_MODEL];
    // ???
    // ??? for (int h = 0; h < H; ++h) {
    // ???     // ---- 计算 attention scores ----
    // ???     float *scores = new float[prompt_len];
    // ???     for (int s = 0; s < prompt_len; ++s) {
    // ???         float k_s[D], v_s[D];
    // ???         cache.load(rid, h, s, k_s, v_s);
    // ???
    // ???         // dot(q_last[h], k_s)
    // ???         float dot = 0;
    // ???         for (int d = 0; d < D; ++d)
    // ???             dot += q_last[h][d] * k_s[d];
    // ???         scores[s] = dot;
    // ???     }
    // ???
    // ???     // ---- softmax ----
    // ???     softmax_inplace(scores, prompt_len);
    // ???
    // ???     // ---- weighted sum over V ----
    // ???     float attn_h[D] = {0};
    // ???     for (int s = 0; s < prompt_len; ++s) {
    // ???         float k_s[D], v_s[D];
    // ???         cache.load(rid, h, s, k_s, v_s);
    // ???         for (int d = 0; d < D; ++d)
    // ???             attn_h[d] += scores[s] * v_s[d];
    // ???     }
    // ???
    // ???     // 写入 context 的对应 head 位置
    // ???     for (int d = 0; d < D; ++d)
    // ???         context[h * D + d] = attn_h[d];
    // ???
    // ???     delete[] scores;
    // ??? }

    // --- 你的代码 ---


    // --- 你的代码结束 ---

    // ================================================================
    // 第 3 步: 输出投影 context × Wo → last_hidden
    // ================================================================
    // ??? linear(context, Wo, last_hidden);

    // --- 你的代码 ---


    // --- 你的代码结束 ---
}

// ============================================================================
// TODO 2: decode_step
// ============================================================================
//
// 输入:
//   cache        PagedKVCache（已在 prefill 中存好了所有 K/V）
//   rid          请求 ID
//   token_emb    当前 token 的 embedding [D_MODEL]
//   cached_len   已缓存的 K/V 总长度（= prompt_len + 已生成的 token 数）
//   output       输出: 下一个 token 的 hidden state [D_MODEL]
//
// 要做什么:
//   (1) 对 token_emb 做 Q/K/V 投影，K/V 存入 cache 的 cached_len 位置
//   (2) Q 对所有 cached_len+1 个位置做 paged attention
//   (3) 拼接多头 → context × Wo → output
//
void decode_step(PagedKVCache &cache, int rid,
                 const float *token_emb, int cached_len,
                 float *output) {
    // ================================================================
    // 第 1 步: 投影 Q/K/V，并存储 K/V 到 cache
    // ================================================================
    //
    // 这和 prefill 的"存储"步骤类似，但只有一个 token
    //
    // ??? float q_full[D_MODEL], k_full[D_MODEL], v_full[D_MODEL];
    // ??? linear(token_emb, Wq, q_full);
    // ??? linear(token_emb, Wk, k_full);
    // ??? linear(token_emb, Wv, v_full);
    // ???
    // ??? float q_per_head[H][D];
    // ??? for (int h = 0; h < H; ++h) {
    // ???     float k_h[D], v_h[D];
    // ???     for (int d = 0; d < D; ++d) {
    // ???         q_per_head[h][d] = q_full[h * D + d];
    // ???         k_h[d] = k_full[h * D + d];
    // ???         v_h[d] = v_full[h * D + d];
    // ???     }
    // ???     // 把 K/V 存入 cache 的位置 cached_len
    // ???     cache.store(rid, h, cached_len, k_h, v_h);
    // ??? }

    // --- 你的代码 ---


    // --- 你的代码结束 ---

    // ================================================================
    // 第 2 步: Paged attention — Q 对 cached_len+1 个位置
    // ================================================================
    //
    // 注意: attention 范围是 0 .. cached_len（包括刚刚存入的新 token）
    //       因为新 token 也要 attend to 自己
    //
    // ??? float context[D_MODEL];
    // ??? int seq_len = cached_len + 1;  // 包括当前 token
    // ???
    // ??? for (int h = 0; h < H; ++h) {
    // ???     float *scores = new float[seq_len];
    // ???     for (int s = 0; s < seq_len; ++s) {
    // ???         float k_s[D], v_s[D];
    // ???         cache.load(rid, h, s, k_s, v_s);
    // ???         float dot = 0;
    // ???         for (int d = 0; d < D; ++d)
    // ???             dot += q_per_head[h][d] * k_s[d];
    // ???         scores[s] = dot;
    // ???     }
    // ???
    // ???     softmax_inplace(scores, seq_len);
    // ???
    // ???     float attn_h[D] = {0};
    // ???     for (int s = 0; s < seq_len; ++s) {
    // ???         float k_s[D], v_s[D];
    // ???         cache.load(rid, h, s, k_s, v_s);
    // ???         for (int d = 0; d < D; ++d)
    // ???             attn_h[d] += scores[s] * v_s[d];
    // ???     }
    // ???
    // ???     for (int d = 0; d < D; ++d) context[h * D + d] = attn_h[d];
    // ???     delete[] scores;
    // ??? }

    // --- 你的代码 ---


    // --- 你的代码结束 ---

    // ================================================================
    // 第 3 步: 输出投影
    // ================================================================
    // ??? linear(context, Wo, output);

    // --- 你的代码 ---


    // --- 你的代码结束 ---
}

// ============================================================================
// TODO 3: generate — 自回归生成循环
// ============================================================================
//
// 输入:
//   cache         PagedKVCache
//   prompt        token ids
//   prompt_len    prompt 长度
//   embedding     整个词表的 embedding [vocab][D_MODEL]
//   lm_head       logits 投影头 [D_MODEL][vocab]（用于从 hidden state → logits）
//   max_new_tokens 最多生成几个 token
//   vocab_size    词表大小
//   generated     输出: int 数组写入生成的 token ids
//   generated_count 输出: 实际生成了几个 token
//
// 流程:
//   1. 分配 KV Cache block
//   2. 调用 prefill_step → 得到 last_hidden
//   3. 循环 max_new_tokens 次:
//      a. last_hidden × lm_head → logits [vocab]
//      b. argmax/logits → next_token（贪婪采样）
//      c. 把 next_token 加入 generated 列表
//      d. next_token 的 embedding → decode_step → 更新 last_hidden
//      e. 如果 next_token 是 EOS（这里用 -1 表示没有 EOS，一直跑满）
//   4. 释放 KV Cache
//
void generate(PagedKVCache &cache,
              const int *prompt, int prompt_len,
              const float *embedding,
              const float *lm_head,
              int max_new_tokens, int vocab_size,
              int *generated, int &generated_count) {
    // ================================================================
    // 第 1 步: 分配 KV Cache
    // ================================================================
    //
    // 计算总共需要多少 token 位置: prompt_len + max_new_tokens
    // 然后换算成 block 数: (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE
    //
    // ??? int total_tokens = prompt_len + max_new_tokens;
    // ??? int num_blocks   = (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // ??? int rid = cache.allocate(num_blocks);
    // ??? if (rid < 0) {
    // ???     printf("ERROR: 无法分配 KV Cache\\n");
    // ???     generated_count = 0;
    // ???     return;
    // ??? }

    // --- 你的代码 ---


    // --- 你的代码结束 ---

    // ================================================================
    // 第 2 步: Prefill
    // ================================================================
    //
    // ??? float last_hidden[D_MODEL];
    // ??? prefill_step(cache, rid, prompt, prompt_len, embedding, last_hidden);

    // --- 你的代码 ---


    // --- 你的代码结束 ---

    // ================================================================
    // 第 3 步: 自回归循环
    // ================================================================
    //
    // generated_count = 0
    // current_len = prompt_len  （当前序列总长度 = prompt + 已生成）
    //
    // for step = 0 .. max_new_tokens-1:
    //   1. last_hidden × lm_head → logits
    //   2. argmax → next_token
    //   3. 保存 generated[generated_count++] = next_token
    //   4. 取 next_token 的 embedding
    //   5. decode_step → 更新 last_hidden
    //   6. current_len++
    //
    // ??? generated_count = 0;
    // ??? int current_len = prompt_len;
    // ???
    // ??? for (int step = 0; step < max_new_tokens; ++step) {
    // ???     // logits = last_hidden × lm_head
    // ???     float *logits = new float[vocab_size];
    // ???     linear(last_hidden, lm_head, logits);
    // ???
    // ???     int next_token = argmax(logits, vocab_size);
    // ???     generated[generated_count++] = next_token;
    // ???     delete[] logits;
    // ???
    // ???     // 取 embedding
    // ???     const float *tok_emb = embedding + next_token * D_MODEL;
    // ???
    // ???     // decode
    // ???     decode_step(cache, rid, tok_emb, current_len, last_hidden);
    // ???     current_len++;
    // ??? }

    // --- 你的代码 ---


    // --- 你的代码结束 ---

    // ================================================================
    // 第 4 步: 释放资源
    // ================================================================
    // ??? cache.free(rid);

    // --- 你的代码 ---


    // --- 你的代码结束 ---
}

// ============================================================================
// Part 4: 测试 — 跑一次完整的生成
// ============================================================================
int main() {
    printf("========================================\n");
    printf("Phase 3 Day 5: 端到端推理\n");
    printf("========================================\n");

    init_weights();

    constexpr int VOCAB_SIZE = 16;  // 极小词表
    constexpr int PROMPT_LEN = 3;
    constexpr int MAX_NEW    = 4;

    // ---- 构造 embedding 表 ----
    float *embedding = new float[VOCAB_SIZE * D_MODEL];
    srand(100);
    for (int i = 0; i < VOCAB_SIZE * D_MODEL; ++i)
        embedding[i] = (float)rand() / RAND_MAX * 0.5f - 0.25f;

    // ---- 构造 lm_head ----
    float *lm_head = new float[D_MODEL * VOCAB_SIZE];
    for (int i = 0; i < D_MODEL * VOCAB_SIZE; ++i)
        lm_head[i] = (float)rand() / RAND_MAX * 0.2f - 0.1f;

    // ---- 构造 prompt: [2, 7, 1] ----
    int prompt[PROMPT_LEN] = {2, 7, 1};

    printf("\nPrompt tokens: [");
    for (int i = 0; i < PROMPT_LEN; ++i) printf("%d%s", prompt[i], i < PROMPT_LEN-1 ? ", " : "");
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
        printf("%d%s", generated[i], i < generated_count-1 ? ", " : "");
    printf("]\n");

    printf("\n========================================\n");
    printf("Test completed.\n");
    printf("========================================\n");

    delete[] embedding;
    delete[] lm_head;
    return 0;
}
