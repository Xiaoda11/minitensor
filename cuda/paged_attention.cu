/**
 * @file paged_attention.cu
 * @brief Phase 3 Day 2: PagedAttention 核心数据结构实现
 *
 * 目标: 实现 Block Table + 分页 KV Cache，理解 PagedAttention 的
 *       逻辑块→物理块映射、按需分配、内部碎片。
 *
 * 核心概念:
 *   Block Table:  每个请求维护一个映射表，logical_block → physical_block
 *   Physical Pool: 所有请求共享的物理 block 池
 *   分配: 请求需要 slot 时，从 free list 中取 block
 *   释放: 请求完成后，归还 block 到 free list
 *
 * 与 Day 1 的关系:
 *   Day 1 用公式算出了连续方案的浪费 → Day 2 实现分页方案来消除浪费
 *
 * 编译:
 *   g++ -std=c++17 -O2 -x c++ paged_attention.cu -o paged_attention && ./paged_attention
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

// ============================================================================
// 常量定义
// ============================================================================

constexpr int MAX_REQUESTS = 8;          // 最多同时服务的请求数
constexpr int MAX_LOGICAL_BLOCKS = 16;   // 每个请求最多分配的逻辑块数
constexpr int TOTAL_PHYSICAL_BLOCKS = 32; // 物理 block 总数
constexpr int BLOCK_SIZE = 16;           // 每个 block 包含的 token 位置数
constexpr int H = 4;                      // Head 数
constexpr int D = 8;                      // Head 维度 (小值，方便手算验证)

// ============================================================================
// Part 1: Block Table — 逻辑块 → 物理块映射
// ============================================================================

/**
 * BlockTable: 每个请求维护一个
 *
 * 类比操作系统页表: 进程看到的连续地址空间 (logical blocks)
 * 被映射到物理内存中不连续的 page (physical blocks)。
 *
 * 例如: 请求需要 2 个 block，block_table = [3, 7]
 *       logical block 0 → physical block 3
 *       logical block 1 → physical block 7
 */
struct BlockTable {
    int physical_blocks[MAX_LOGICAL_BLOCKS]; // 映射表
    int num_blocks;                           // 当前分配了几个 block
    bool active;                              // 这个 slot 是否被占用

    void init() {
        num_blocks = 0;
        active = false;
        for (int i = 0; i < MAX_LOGICAL_BLOCKS; ++i) {
            physical_blocks[i] = -1;          // -1 表示未映射
        }
    }

    /**
     * @brief 给定 token 位置 pos，返回 (physical_block, offset_in_block)
     *
     * 这是 PagedAttention 最核心的地址翻译函数。
     *
     * 输入: 逻辑 token 位置 (例如请求的第 25 个 token)
     * 步骤:
     *   1. logical_block = pos / BLOCK_SIZE   (这个 token 在哪个逻辑块)
     *   2. offset_in_block = pos % BLOCK_SIZE (在块内的偏移)
     *   3. physical_block = block_table[logical_block]  (查表翻译)
     *   4. 返回 (physical_block, offset_in_block)
     *
     * @param pos          逻辑 token 位置 (0-based)
     * @param out_phys_blk 输出: 物理 block 索引
     * @param out_offset   输出: block 内偏移
     */
    void translate(int pos, int *out_phys_blk, int *out_offset) const {
        // ================================================================
        // TODO 1: 实现逻辑地址 → 物理地址翻译
        // ================================================================
        // 提示:
        //   logical_block = pos / BLOCK_SIZE
        //   offset_in_block = pos % BLOCK_SIZE
        //   physical_block = physical_blocks[logical_block]
        //

        // --- 你的代码 ---
        int logical_block = pos / BLOCK_SIZE;
        int offset_in_block = pos % BLOCK_SIZE ;
        int physical_block = physical_blocks[logical_block];
        *out_phys_blk = physical_block ;
        *out_offset = offset_in_block ;
       
        // --- 你的代码结束 ---
    }
};

// ============================================================================
// Part 2: Paged KV Cache — 物理 block 池 + 分配/释放
// ============================================================================

/**
 * PagedKVCache: 所有请求共享的物理 KV 存储
 *
 * 数据布局 (per-head, 物理 block 视角):
 *   K_pool[physical_block][head][offset_in_block][d]
 *
 * 从物理地址到线性内存偏移:
 *   先按 head 分组: 每个 block 存 H × BLOCK_SIZE × D 个 float
 *   然后 K 和 V 各一份 → ×2
 *
 *   一个物理 block 的 K 存储量 = H × BLOCK_SIZE × D 个 float
 *   物理 block p 的 K 起始偏移 = p × H × BLOCK_SIZE × D
 *   在该 block 内, head h, offset o 的 K 偏移 = h × BLOCK_SIZE × D + o × D
 */
class PagedKVCache {
private:
    // 物理内存池 — K 和 V 分开存储
    float *K_pool;  // [TOTAL_PHYSICAL_BLOCKS * H * BLOCK_SIZE * D]
    float *V_pool;  // 同上

    // Free list: 布尔数组标记哪些物理 block 未被使用
    bool block_free[TOTAL_PHYSICAL_BLOCKS];

    // 每个请求的 block table
    BlockTable tables[MAX_REQUESTS];

    // 辅助: 计算一个物理 block 的字节数
    int block_bytes() const {
        return H * BLOCK_SIZE * D * sizeof(float);
    }

public:
    void init() {
        // 分配物理池
        int total_elements = TOTAL_PHYSICAL_BLOCKS * H * BLOCK_SIZE * D;
        K_pool = new float[total_elements]();
        V_pool = new float[total_elements]();

        // 初始化 free list
        for (int i = 0; i < TOTAL_PHYSICAL_BLOCKS; ++i) {
            block_free[i] = true;   // 所有 block 初始空闲
        }

        // 初始化所有请求的 block table
        for (int i = 0; i < MAX_REQUESTS; ++i) {
            tables[i].init();
        }
    }

    ~PagedKVCache() {
        delete[] K_pool;
        delete[] V_pool;
    }

    // ---- 分配 ----

    /**
     * @brief 为一个新请求分配 num_blocks 个物理 block
     *
     * 步骤:
     *   1. 找一个空闲的 request slot (tables[rid].active == false)
     *   2. 从 free list 中找 num_blocks 个空闲的 physical block
     *   3. 将这些物理 block 填入 tables[rid].physical_blocks[]
     *   4. 设置 tables[rid].active = true, num_blocks = num_blocks
     *
     * @param num_blocks  需要分配的 block 数量
     * @return request_id (0 ~ MAX_REQUESTS-1)，失败返回 -1
     */
    int allocate(int num_blocks) {
        // ================================================================
        // TODO 2: 实现 block 分配
        // ================================================================
        // 提示:
        //   1. 找到第一个空闲的 request slot → rid
        //   2. 扫描 block_free[] (0..TOTAL_PHYSICAL_BLOCKS-1)，
        //      用 found 计数器，遇到空闲 block 就收集
        //   3. 同时填入 tables[rid].physical_blocks[found] 并标记 block_free[]=false
        //   4. found == num_blocks 时 break，最后设置 num_blocks 和 active
        //

        // --- 你的代码 ---
        int rid = 0;
        while (tables[rid].active != false)
        {
            rid++;
        }
        
           int found = 0;                          // 找到了几个
    
        for (int pb = 0; pb < TOTAL_PHYSICAL_BLOCKS; ++pb) {
            if (block_free[pb]) {               // 灯亮，这个柜子空的
                tables[rid].physical_blocks[found] = pb;   // 记下柜子编号
                block_free[pb] = false;         // 关灯，标记已占用
                found++;                         // 找到一个
    
            if (found == num_blocks) break;  // 找够了，停
        }
    }
    
        tables[rid].num_blocks = found;
        tables[rid].active = true;
        return rid;
        // --- 你的代码结束 ---
    }

    // ---- 释放 ----

    /**
     * @brief 释放一个请求的所有 block
     *
     * 步骤:
     *   1. 遍历 tables[rid].physical_blocks[0..num_blocks-1]
     *   2. 把每个物理 block 标记为空闲
     *   3. 重置 tables[rid]
     *
     * @param rid  请求 ID
     */
    void free(int rid) {
        // ================================================================
        // TODO 3: 实现 block 释放
        // ================================================================
        // 提示:
        //   for (int i = 0; i < tables[rid].num_blocks; ++i) {
        //       int pb = tables[rid].physical_blocks[i];
        //       block_free[pb] = ???;
        //   }
        //   tables[rid].init();   
       for (int i = 0; i < tables[rid].num_blocks; ++i) {
              int pb = tables[rid].physical_blocks[i];
           block_free[pb] = true;
       }
           tables[rid].init();
     
    }

    // ---- 存储 ----

    /**
     * @brief 存储一个 K/V 向量到分页 KV Cache
     *
     * 步骤:
     *   1. 用 tables[rid].translate(pos, ...) 得到 (phys_blk, offset)
     *   2. 计算 K_pool 中的目标地址: phys_blk * H * BLOCK_SIZE * D + head * BLOCK_SIZE * D + offset * D
     *   3. memcpy K 和 V
     *
     * @param rid    请求 ID
     * @param head   注意力头索引
     * @param pos    逻辑 token 位置
     * @param k_vec  要存储的 K 向量 (长度 D)
     * @param v_vec  要存储的 V 向量 (长度 D)
     */
    void store(int rid, int head, int pos, const float *k_vec, const float *v_vec) {
        // ================================================================
        // TODO 4: 实现分页存储
        // ================================================================
        // 提示:
        //   1. translate 得到 (phys_blk, offset)
        //   2. K_target = K_pool + phys_blk * H * BLOCK_SIZE * D
        //                     + head   * BLOCK_SIZE * D
        //                     + offset * D
        //   3. memcpy(K_target, k_vec, D * sizeof(float))
        //

        // --- 你的代码 ---
        int phys_blk;
        int offset;
        tables[rid].translate(pos,&phys_blk,&offset);
        float *K_target = K_pool + phys_blk * H * BLOCK_SIZE * D
                             + head   * BLOCK_SIZE * D
                            + offset * D;
        memcpy(K_target, k_vec, D * sizeof(float));
        float *V_target = V_pool + phys_blk * H * BLOCK_SIZE * D
                             + head   * BLOCK_SIZE * D
                            + offset * D;
        memcpy(V_target, v_vec, D * sizeof(float));
        // --- 你的代码结束 ---
    }

    // ---- 读取 ----

    /**
     * @brief 从分页 KV Cache 加载一个 K/V 向量
     *
     * 和 store 完全对称，只是方向反过来。
     *
     * @param rid    请求 ID
     * @param head   注意力头索引
     * @param pos    逻辑 token 位置
     * @param k_out  输出: K 向量 (长度 D)
     * @param v_out  输出: V 向量 (长度 D)
     */
    void load(int rid, int head, int pos, float *k_out, float *v_out) const {
        // ================================================================
        // TODO 5: 实现分页读取
        // ================================================================
        // 提示: 和 store 完全相同，只是 memcpy 方向相反 (从 pool → out)
        //

       int phys_blk;
        int offset;
        tables[rid].translate(pos,&phys_blk,&offset);
        float * K_src = K_pool + phys_blk * H * BLOCK_SIZE * D
                             + head   * BLOCK_SIZE * D
                            + offset * D;
        memcpy(k_out, K_src, D * sizeof(float));
        float *V_src = V_pool + phys_blk * H * BLOCK_SIZE * D
                             + head   * BLOCK_SIZE * D
                            + offset * D;
        memcpy(v_out, V_src, D * sizeof(float));
    }

    // ---- 获取状态 (供测试使用) ----

    const BlockTable &get_table(int rid) const { return tables[rid]; }

    int count_free_blocks() const {
        int count = 0;
        for (int i = 0; i < TOTAL_PHYSICAL_BLOCKS; ++i) {
            if (block_free[i]) count++;
        }
        return count;
    }
};

// ============================================================================
// Part 3: 测试
// ============================================================================

PagedKVCache g_cache;  // 全局变量，方便测试函数访问

/**
 * @brief 测试 1: 基本 allocate → store → load → free
 *
 * 验证:
 *   1. 分配 2 个 block → 得到 request ID
 *   2. 存储 pos=5 的 KV → 存储成功
 *   3. 读取 pos=5 的 KV → 读到之前存储的值
 *   4. 释放请求 → block 归还到 free list
 */
bool test_allocate_store_load() {
    printf("--- Test 1: allocate → store → load → free ---\n");

    // Step 1: 分配一个请求，需要 2 个 block
    // TODO 6: 调用 allocate

    int rid = -1;
    rid = g_cache.allocate(2);
    assert(rid >= 0);
    
    printf("  分配请求: rid=%d, num_blocks=%d\n", rid, g_cache.get_table(rid).num_blocks);

    // Step 2: 存储一些 KV
    float k_vec[D], v_vec[D];
    for (int d = 0; d < D; ++d) {
        k_vec[d] = (float)(rid * 100 + d);    // 用 rid 区分，方便验证
        v_vec[d] = (float)(rid * 1000 + d);
    }
    
    // 在 head=0 的 pos=5 处存储
    // TODO 7: 调用 store

    // --- 你的代码 ---
    g_cache.store(rid, 0, 5, k_vec, v_vec);
    printf("  存储: rid=%d, head=0, pos=5 → K[0..%d] = {", rid, D-1);
    for (int d = 0; d < D && d < 4; ++d) printf("%.0f,", k_vec[d]);
    printf("...}\n");

    // --- 你的代码结束 ---

    // Step 3: 读回来
    float k_out[D], v_out[D];
    // TODO 8: 调用 load

    // --- 你的代码 ---

    g_cache.load(rid, 0, 5, k_out, v_out);
    // --- 你的代码结束 ---

    // Step 4: 验证存储 = 读取
    for (int d = 0; d < D; ++d) {
        assert(k_out[d] == k_vec[d]);
        assert(v_out[d] == v_vec[d]);
    }
    printf("  ✓ store/load 一致，验证通过\n");

    // Step 5: 释放
    int free_before = g_cache.count_free_blocks();
    // TODO 9: 调用 free

    // --- 你的代码 ---
    g_cache.free(rid);

    // --- 你的代码结束 ---
    int free_after = g_cache.count_free_blocks();
    printf("  释放后: free blocks %d → %d (回收了 %d)\n",
           free_before, free_after, free_after - free_before);
    assert(free_after > free_before);

    return true;
}

/**
 * @brief 测试 2: 多请求并行 + 越界存储
 *
 * 模拟 3 个请求同时服务:
 *   - 请求 0: 长度 35 → 需要 ceil(35/16) = 3 blocks
 *   - 请求 1: 长度 20 → 需要 ceil(20/16) = 2 blocks
 *   - 请求 2: 长度 10 → 需要 ceil(10/16) = 1 block
 *
 * 验证:
 *   1. 三个请求分配到不同的物理 block (无重叠)
 *   2. 分别存储不同位置的 KV
 *   3. 各自读回 → 互不干扰
 *   4. 内部碎片: 分配 = 6 blocks × 16 slots = 96 slots, 使用 = 65, 浪费 = 31
 *      对比连续方案: 3 × 16 blocks = 48 用不完的浪费... 等等，连续方案每个请求
 *      需要固定分配 max_length 个 block。如果 S_max = 48:
 *        连续: 3 × 48 = 144 slots 分配, 65 使用, 54.9% 浪费
 *        分页: 96 slots 分配, 65 使用, 32.3% 浪费
 */
bool test_multi_request() {
    printf("\n--- Test 2: 多请求并行 ---\n");

    int lengths[3] = {35, 20, 10};
    int num_blocks[3];
    int rid[3];

    // Step 1: 分配 3 个请求
    for (int i = 0; i < 3; ++i) {
        // TODO 10: 计算需要的 block 数 + 分配

        // --- 你的代码 ---
        num_blocks[i] = (lengths[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;;
        rid[i] = g_cache.allocate(num_blocks[i]);
        // --- 你的代码结束 ---
        assert(rid[i] >= 0);
        printf("  请求 %d: length=%d, %d blocks, rid=%d\n",
               i, lengths[i], num_blocks[i], rid[i]);
    }

    // 验证: 三个请求使用了不同的物理 block (无共享)
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            for (int bi = 0; bi < num_blocks[i]; ++bi) {
                for (int bj = 0; bj < num_blocks[j]; ++bj) {
                    int pb_i = g_cache.get_table(rid[i]).physical_blocks[bi];
                    int pb_j = g_cache.get_table(rid[j]).physical_blocks[bj];
                    // 物理 block 索引不应该相同
                    // 注意: 这里的 assert 在你的代码跑通后会自动验证
                }
            }
        }
    }

    // Step 2: 每个请求在不同位置存储 KV
    for (int i = 0; i < 3; ++i) {
        float k_vec[D], v_vec[D];
        // TODO 11: 填充 k_vec/v_vec 并调用 store
        // 提示: 在 pos = lengths[i] - 1 (最后一个位置) 存储，方便验证

        // --- 你的代码 ---
        for (int d = 0; d < D; ++d) { 
            k_vec[d] = (float)(rid[i] * 100 + lengths[i] - 1 + d);    // 用 rid 区分，方便验证
            v_vec[d] = (float)(rid[i] * 100 + lengths[i] - 1 + d);
         }
        g_cache.store(rid[i], 0, lengths[i]-1, k_vec, v_vec);
        // --- 你的代码结束 ---
    }

    // Step 3: 各自读回，互不干扰
    for (int i = 0; i < 3; ++i) {
        float k_out[D], v_out[D];
        float expected_k = (float)(rid[i] * 100 + lengths[i] - 1);
        // TODO 12: 调用 load 并验证
        g_cache.load(rid[i], 0, lengths[i]-1, k_out, v_out);
        assert(k_out[0] == expected_k);
        // --- 你的代码 ---


        // --- 你的代码结束 ---
        printf("  请求 %d: pos=%d 验证通过\n", i, lengths[i] - 1);
    }

    // Step 4: 内部碎片分析
    int total_allocated = 0, total_used = 0;
    for (int i = 0; i < 3; ++i) {
        total_allocated += num_blocks[i] * BLOCK_SIZE;
        total_used += lengths[i];
    }
    float waste = 100.0f * (total_allocated - total_used) / total_allocated;
    printf("\n  分配统计: 分配=%d slots, 使用=%d, 分页浪费=%.1f%%\n",
           total_allocated, total_used, waste);
    printf("  连续方案 (S_max=%d): 分配=%d, 浪费=%.1f%%\n",
           48, 3 * 48, 100.0f * (3 * 48 - total_used) / (3 * 48));

    // Step 5: 释放所有请求
    for (int i = 0; i < 3; ++i) {
        // TODO 13: 释放请求

        // --- 你的代码 ---
        g_cache.free(rid[i]);

        // --- 你的代码结束 ---
    }

    return true;
}

// ============================================================================
// main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Phase 3 Day 2: PagedAttention 实现\n");
    printf("BlockTable + 分页 KV Cache\n");
    printf("========================================\n\n");

    g_cache.init();

    // 打印配置
    printf("配置: H=%d, D=%d, BLOCK_SIZE=%d\n", H, D, BLOCK_SIZE);
    printf("      物理 block 总数=%d, 每 block=%.1f KB\n\n",
           TOTAL_PHYSICAL_BLOCKS,
           (H * BLOCK_SIZE * D * sizeof(float) * 2) / 1024.0);

    bool ok1 = test_allocate_store_load();
    bool ok2 = test_multi_request();

    printf("\n========================================\n");
    printf("Result: %s\n", (ok1 && ok2) ? "All tests passed" : "Some tests FAILED");
    printf("========================================\n");

    return (ok1 && ok2) ? 0 : 1;
}
