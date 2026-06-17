/**
 * @file paged_attention_day3.cu
 * @brief Phase 3 Day 3: 分页 Attention 计算
 *
 * 目标: 用 Day 2 实现的 BlockTable + PagedKVCache，真正跑一次 attention。
 *
 * 核心问题:
 *   连续 KV Cache 里遍历 K[0], K[1], ... K[length-1] 很简单——
 *   指针就是连续的。分页 KV Cache 里 K 散落在不同物理 block 中，
 *   怎么遍历？
 *
 * 答案: 不通过"物理连续"遍历，而是通过"逻辑连续"遍历——
 *   对每个 pos，调用 translate() 找到物理位置。
 *
 * 编译:
 *   g++ -std=c++17 -O2 -x c++ paged_attention_day3.cu -o paged_attention_day3 && ./paged_attention_day3
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

// ============================================================================
// 常量 (与 Day 2 一致，但精简为单 head 以便手算验证)
// ============================================================================

constexpr int MAX_LOGICAL_BLOCKS = 8;
constexpr int TOTAL_PHYSICAL_BLOCKS = 8;
constexpr int BLOCK_SIZE = 4;          // 故意用小值，方便手算
constexpr int H = 1;                   // 单 head，简化
constexpr int D = 4;                   // D=4，所有向量可以手算验证

// ============================================================================
// Part 1: BlockTable (精简版，和 Day 2 一样但注释更少)
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

    void translate(int pos, int *out_phys_blk, int *out_offset) const {
        int logical_block = pos / BLOCK_SIZE;
        *out_offset = pos % BLOCK_SIZE;
        *out_phys_blk = physical_blocks[logical_block];
    }
};

// ============================================================================
// Part 2: MiniPagedCache (精简版 PagedKVCache，单 head)
// ============================================================================

class MiniPagedCache {
private:
    float *K_pool;   // [TOTAL * BLOCK_SIZE * D]
    float *V_pool;
    bool block_free[TOTAL_PHYSICAL_BLOCKS];
    BlockTable table;   // 我们只有一个请求

public:
    void init() {
        K_pool = new float[TOTAL_PHYSICAL_BLOCKS * BLOCK_SIZE * D]();
        V_pool = new float[TOTAL_PHYSICAL_BLOCKS * BLOCK_SIZE * D]();
        for (int i = 0; i < TOTAL_PHYSICAL_BLOCKS; ++i) block_free[i] = true;
        table.init();
    }

    ~MiniPagedCache() { delete[] K_pool; delete[] V_pool; }

    bool allocate(int num_blocks) {
        int found = 0;
        for (int pb = 0; pb < TOTAL_PHYSICAL_BLOCKS; ++pb) {
            if (block_free[pb]) {
                table.physical_blocks[found++] = pb;
                block_free[pb] = false;
                if (found == num_blocks) break;
            }
        }
        if (found < num_blocks) return false;
        table.num_blocks = num_blocks;
        table.active = true;
        return true;
    }

    void store(int pos, const float *k_vec, const float *v_vec) {
        int phys_blk, offset;
        table.translate(pos, &phys_blk, &offset);
        float *K_dst = K_pool + phys_blk * BLOCK_SIZE * D + offset * D;
        float *V_dst = V_pool + phys_blk * BLOCK_SIZE * D + offset * D;
        memcpy(K_dst, k_vec, D * sizeof(float));
        memcpy(V_dst, v_vec, D * sizeof(float));
    }

    void load(int pos, float *k_out, float *v_out) const {
        int phys_blk, offset;
        table.translate(pos, &phys_blk, &offset);
        const float *K_src = K_pool + phys_blk * BLOCK_SIZE * D + offset * D;
        const float *V_src = V_pool + phys_blk * BLOCK_SIZE * D + offset * D;
        memcpy(k_out, K_src, D * sizeof(float));
        memcpy(v_out, V_src, D * sizeof(float));
    }

    // 调试: 打印一个物理 block 的内容
    void print_block(int pb) const {
        printf("  Block %d: K = [", pb);
        for (int i = 0; i < BLOCK_SIZE * D; ++i)
            printf("%.0f%s", K_pool[pb * BLOCK_SIZE * D + i],
                   i == BLOCK_SIZE * D - 1 ? "" : " ");
        printf("]\n");
    }
};

// ============================================================================
// Part 3: 使用分页 cache 的数据，计算 Attention
// ============================================================================

/**
 * @brief 对分页 KV Cache 中的序列做单 query attention
 *
 * 输入:
 *   @param cache   已存储好 K/V 的分页 cache
 *   @param query   当前的 Q 向量 (长度 D)
 *   @param length  序列中已存储的 token 数
 *   @param output  输出: attention 加权后的向量 (长度 D)
 *
 * 这个函数模拟 decode 阶段: 新来了一个 token 的 Q，要和之前所有 token
 * 的 K/V 做 attention。区别只是 K/V 存在分页 cache 中，需要通过
 * block table 翻译才能找到。
 *
 * 连续版本 (回忆 Phase 2 decode):
 *   for (int pos = 0; pos < length; ++pos)
 *       float *k = K_cont + pos * D;   ← 地址连续，指针直接偏移
 *       score[pos] = dot(Q, k);
 *
 * 分页版本:
 *   for (int pos = 0; pos < length; ++pos)
 *       translate(pos, &phys_blk, &offset);  ← 要先翻译
 *       float *k = K_pool + phys_blk * B * D + offset * D;
 *       score[pos] = dot(Q, k);
 */
void paged_attention_decode(const MiniPagedCache &cache,
                            const float *query, int length,
                            float *output) {
    // ================================================================
    // TODO 1: 遍历所有 token 位置，计算 Q·K 分数
    // ================================================================
    //
    // 对 pos = 0 .. length-1:
    //   1. cache.load(pos, k_vec, v_vec)
    //   2. 计算 query 和 k_vec 的点积 → scores[pos]
    //
    // ??? float *scores = ???;
    // ??? for (int pos = 0; pos < length; ++pos) {
    // ???     float k_vec[D], v_vec[D];
    // ???     cache.load(pos, ???, ???);
    // ???     scores[pos] = ???;  // Q 和 K 的点积
    // ??? }

    // --- 你的代码 ---
    float *scores = new float[length];
    for (int pos = 0; pos < length; ++pos) {
        float k_vec[D], v_vec[D];
        cache.load(pos, k_vec, v_vec);
        float dot = 0;
        for (int d = 0; d < D; ++d) dot += query[d] * k_vec[d];
        scores[pos] = dot;
    }


    // --- 你的代码结束 ---

    // ================================================================
    // TODO 2: Softmax
    // ================================================================
    //
    // 对 scores[0..length-1] 做 safe softmax:
    //   1. 找 max_score
    //   2. exp(scores[i] - max_score)，求和
    //   3. 归一化 → probs[i]
    //
    // ??? float max_score = scores[0];
    // ??? for (...) if (scores[i] > max_score) max_score = scores[i];
    // ??? float *probs = ???;
    // ??? float exp_sum = 0;
    // ??? for (...) {
    // ???     probs[i] = ???;
    // ???     exp_sum += ???;
    // ??? }
    // ??? for (...) probs[i] /= exp_sum;

    // --- 你的代码 ---

    float max_score = scores[0];
    for (int i = 1; i < length; ++i)
         if (scores[i] > max_score) max_score = scores[i];
            float exp_sum = 0;
            float *probs = new float[length];
    for (int i = 0; i < length; ++i) {
        probs[i] = expf(scores[i] - max_score);
        exp_sum += probs[i];
    }
    for (int i = 0; i < length; ++i) probs[i] /= exp_sum;
    // --- 你的代码结束 ---

    // ================================================================
    // TODO 3: 加权求和 — output = sum(probs[pos] * V[pos])
    // ================================================================
    //
    // output = Σ probs[pos] × V[pos]
    // 每个 pos: load V, 乘权重, 累加到 output
    //
    // ??? memset(output, 0, D * sizeof(float));
    // ??? for (int pos = 0; pos < length; ++pos) {
    // ???     float k_vec[D], v_vec[D];
    // ???     cache.load(pos, ???, ???);
    // ???     for (int d = 0; d < D; ++d)
    // ???         output[d] += probs[pos] * v_vec[d];
    // ??? }

    // --- 你的代码 ---
    memset(output, 0, D * sizeof(float));
    for (int pos = 0; pos < length; ++pos) {
        float k_vec[D], v_vec[D];   
        cache.load(pos, k_vec, v_vec);
        for (int d = 0; d < D; ++d)
        output[d] += probs[pos] * v_vec[d];
    }


    // --- 你的代码结束 ---

    // 清理
    // ??? delete[] scores;
    // ??? delete[] probs;

    // --- 你的代码 ---
    delete[] scores;
    delete[] probs;
    // --- 你的代码结束 ---
}

// ============================================================================
// Part 4: 测试 — 小规模手算可验证
// ============================================================================

MiniPagedCache g_cache;

/**
 * @brief 测试: 3 个 token 的序列，手动填充 K/V，验证 attention 输出
 *
 * 设置:
 *   BLOCK_SIZE = 4, D = 4, H = 1
 *   序列长度 = 3 (3 个 token)
 *
 * 手动填充 K/V:
 *   pos=0: K=[1,0,0,0], V=[1,0,0,0]
 *   pos=1: K=[0,1,0,0], V=[0,1,0,0]
 *   pos=2: K=[0,0,1,0], V=[0,0,1,0]
 *   (故意设计成相互正交，方便手算验证)
 *
 * Query:
 *   Q = [1, 0, 1, 0]   (与 pos=0 和 pos=2 的 K 有关)
 *
 * 预期:
 *   Q·K[0] = 1*1 + 0*0 + 1*0 + 0*0 = 1
 *   Q·K[1] = 1*0 + 0*1 + 1*0 + 0*0 = 0
 *   Q·K[2] = 1*0 + 0*0 + 1*1 + 0*0 = 1
 *   softmax → 大致 [0.42, 0.16, 0.42]
 *   output ≈ 0.42*[1,0,0,0] + 0.16*[0,1,0,0] + 0.42*[0,0,1,0]
 *          = [0.42, 0.16, 0.42, 0]
 */
bool test_decode_attention() {
    printf("--- Test: Paged Attention (3 tokens) ---\n\n");

    g_cache.init();

    // Step 1: 分配 1 个 block (BLOCK_SIZE=4，够放 3 个 token)
    bool ok = g_cache.allocate(1);
    assert(ok);

    // Step 2: 手动填充 K/V
    // TODO 4: 填充 pos=0,1,2 的 K 和 V
    // 提示: 用 float k[D] = {...} 然后 g_cache.store(pos, k, v)
    //
    // ??? { float k0[4] = {1,0,0,0}, v0[4] = {1,0,0,0};
    // ???   g_cache.store(0, k0, v0); }
    // ??? { float k1[4] = {0,1,0,0}, v1[4] = {0,1,0,0};
    // ???   g_cache.store(1, k1, v1); }
    // ??? { float k2[4] = {0,0,1,0}, v2[4] = {0,0,1,0};
    // ???   g_cache.store(2, k2, v2); }

    // --- 你的代码 ---
    { float k0[4] = {1,0,0,0}, v0[4] = {1,0,0,0};
      g_cache.store(0, k0, v0); }
    { float k1[4] = {0,1,0,0}, v1[4] = {0,1,0,0};
      g_cache.store(1, k1, v1); }
    { float k2[4] = {0,0,1,0}, v2[4] = {0,0,1,0};
   g_cache.store(2, k2, v2); }

    // --- 你的代码结束 ---

    // Step 3: 打印物理 block 内容 (验证存储正确)
    printf("物理 block 0 的内容:\n");
    g_cache.print_block(0);
    // 预期: K = [1 0 0 0   0 1 0 0   0 0 1 0   0 0 0 0]
    //       (pos=0,1,2 占前 3 个槽，第 4 个槽是 0)

    // Step 4: 计算 attention
    float query[4] = {1, 0, 1, 0};
    float output[4];

    // TODO 5: 调用 paged_attention_decode
    // ??? paged_attention_decode(g_cache, query, 3, output);

    // --- 你的代码 ---

    paged_attention_decode(g_cache, query, 3, output);
    // --- 你的代码结束 ---

    printf("\nQuery    = [%.0f %.0f %.0f %.0f]\n",
           query[0], query[1], query[2], query[3]);
    printf("Output   = [%.3f %.3f %.3f %.3f]\n",
           output[0], output[1], output[2], output[3]);
    printf("Expected ≈ [0.422 0.155 0.422 0.000]\n");

    // 验证: output[0] 应该 > output[1] (Q 更关注 pos=0)
    //       output[2] 应该 > output[1] (Q 更关注 pos=2)
    assert(output[0] > output[1]);
    assert(output[2] > output[1]);

    printf("\n✓ Test passed\n");
    return true;
}

// ============================================================================
// main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Phase 3 Day 3: PagedAttention 计算\n");
    printf("========================================\n\n");

    bool ok = test_decode_attention();

    printf("\n========================================\n");
    printf("Result: %s\n", ok ? "All tests passed" : "FAILED");
    printf("========================================\n");

    return ok ? 0 : 1;
}
