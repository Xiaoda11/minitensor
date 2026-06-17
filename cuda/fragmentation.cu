/**
 * @file fragmentation.cu
 * @brief Phase 3 Week 1 Day 1: 连续 KV Cache 的碎片问题
 *
 * 目标: 用具体数字理解为什么连续预分配 KV Cache 会浪费大量显存，
 *       以及分页 (Paged) 方案如何解决。
 *
 * 场景模拟:
 *   一个推理引擎同时服务 N 个请求，每个请求的 prompt 长度不同。
 *   连续方案: 每个请求预分配 S_max = 2048 个 slot
 *   分页方案: 每个请求按需分配 block（每 block = 16 个 slot）
 *
 * 编译:
 *   g++ -std=c++17 -O2 -x c++ fragmentation.cu -o fragmentation && ./fragmentation
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <ctime>

// ============================================================================
// Part 0: 模拟数据 — N 个请求的真实长度
// ============================================================================

/**
 * @brief 生成模拟的请求长度分布
 *
 * 模拟真实场景: 大部分请求较短 (100-500 tokens)，
 * 少数长请求 (1000-2000 tokens)。
 * 但连续方案必须给每个请求都预留 S_max=2048。
 */
void generate_request_lengths(int *lengths, int N, int S_max) {
    srand(42);  // 固定种子，结果可复现
    for (int i = 0; i < N; ++i) {
        // 大部分短请求，少数长请求
        float r = (float)rand() / RAND_MAX;
        if (r < 0.6f) {
            lengths[i] = 50 + rand() % 200;        // 60%: 50-250
        } else if (r < 0.85f) {
            lengths[i] = 300 + rand() % 700;       // 25%: 300-1000
        } else {
            lengths[i] = 1000 + rand() % 1000;     // 15%: 1000-2000
        }
        if (lengths[i] > S_max) lengths[i] = S_max;
    }
}

// ============================================================================
// Part 1: 连续 KV Cache — 每个请求固定预分配 S_max
// ============================================================================

/**
 * @brief 分析连续 KV Cache 的内存浪费
 *
 * 参数:
 *   @param lengths   N 个请求的实际长度
 *   @param N         请求数
 *   @param S_max     每个请求的预分配 slot 数
 *   @param D         Head 维度 (每个 slot 占 D 个 float)
 *   @param L         Layer 数
 *   @param H         Head 数
 *   @param out_allocated  输出: 总分配的 slot 数 (float 数 = × 2 因为 K+V)
 *   @param out_used       输出: 实际用到的 slot 数
 *   @param out_waste_pct  输出: 浪费百分比
 */
void analyze_continuous(const int *lengths, int N, int S_max,
                        int D, int L, int H,
                        long long *out_allocated,
                        long long *out_used,
                        float *out_waste_pct) {
    // ================================================================
    // TODO 1: 计算连续方案的总分配量和实际使用量
    // ================================================================
    //
    // 提示:
    //   - 每个 layer × head 独立管理一段 cache
    //   - 连续方案: 每个请求在每个 (layer,head) 都分配 S_max 个 slot
    //   - 每个 slot = 1 个 K 向量 + 1 个 V 向量 = 2 × D 个 float
    //   - 总分配的 float 数 = ???
    //   - 实际使用的 float 数 = ???
    //

    // --- 你的代码 ---
    long long allocated_slots =N * L * H * S_max ;
    long long used_slots = 0 ;
    for (int i = 0; i < N; i++)
    {
       used_slots += L * H * lengths[i];
    }
       
    // --- 你的代码结束 ---

    *out_allocated = allocated_slots;
    *out_used = used_slots;
    *out_waste_pct = (allocated_slots > 0)
                     ? 100.0f * (allocated_slots - used_slots) / allocated_slots
                     : 0.0f;
}

// ============================================================================
// Part 2: 分页 KV Cache — 每个请求按需分配 block
// ============================================================================

/**
 * @brief 分析分页 KV Cache 的内存使用
 *
 * 参数:
 *   @param lengths     N 个请求的实际长度
 *   @param N           请求数
 *   @param S_max       单个请求最大长度 (block 总数受此限制)
 *   @param block_size  每个 block 包含的 slot 数 (如 16)
 *   @param D           Head 维度
 *   @param L           Layer 数
 *   @param H           Head 数
 *   @param out_allocated  输出: 总分配的 slot 数
 *   @param out_used       输出: 实际用到的 slot 数
 *   @param out_waste_pct  输出: 浪费百分比
 */
void analyze_paged(const int *lengths, int N, int S_max,
                   int block_size, int D, int L, int H,
                   long long *out_allocated,
                   long long *out_used,
                   float *out_waste_pct) {
    // ================================================================
    // TODO 2: 计算分页方案的分配量和浪费
    // ================================================================
    //
    // 分页方案: 每个请求按需分配 block
    //   - 请求 i 需要 ceil(lengths[i] / block_size) 个 block
    //   - 每个 block = block_size 个 slot
    //   - 最后一个 block 可能没装满 → 产生内部碎片 (internal fragmentation)
    //
    // 提示:
    //   - 先算每个请求需要几个 block
    //   - 分配量 = (分配的 block 数) × (每个 block 的 slot 数)
    //   - 使用量 = sum of lengths[i]
    //   - 浪费 = 分配 - 使用 = 每个请求最后 block 未用部分
    //

    // --- 你的代码 ---
    long long allocated_slots = 0;
    long long used_slots = 0;
    int *block_num = new int[N];
    for (size_t i = 0; i < N; i++)
    {
        block_num[i] = (lengths[i] + block_size - 1) / block_size;
        allocated_slots += L * H * block_num[i]*block_size;
        used_slots += L * H *lengths[i];
    }
    delete []block_num;
    // --- 你的代码结束 ---

    *out_allocated = allocated_slots;
    *out_used = used_slots;
    *out_waste_pct = (allocated_slots > 0)
                     ? 100.0f * (allocated_slots - used_slots) / allocated_slots
                     : 0.0f;
}

// ============================================================================
// Part 3: 可视化 — 小规模演示
// ============================================================================

/**
 * @brief 小规模可视化: 连续 vs 分页的内存布局对比
 *
 * 模拟 3 个请求，S_max=24，block_size=8，只有 1 个 layer×head。
 * 只显示 K 缓存 (V 同理)。
 *
 * 连续方案: 每个请求固定占 S_max=24 个 slot
 *
 *   请求 0 (真实长度 10): [##########xxxxxxxxxxxxxx]  24 slots, 14 浪费
 *   请求 1 (真实长度 18): [##################xxxxxx]  24 slots,  6 浪费
 *   请求 2 (真实长度 5):  [#####xxxxxxxxxxxxxxxxxxx]  24 slots, 19 浪费
 *                        ─────────────────────────
 *                        72 slots 分配, 33 使用, 54% 浪费
 *
 * TODO: 分页方案 (每个请求按需分配 block)
 *   画一下: 请求 0 需要 ceil(10/8)=2 blocks, 请求 1 需要 ??? blocks...
 *
 */
void visualize_paged_example() {
    printf("--- 可视化: 分页方案 (block_size=8) ---\n\n");

    // 3 个请求: length = {10, 18, 5}
    // 连续方案已在上面的 ASCII 展示
    // TODO: 你在这里画分页方案的 ASCII 图

    printf("  请求 0 (长度 10): need 2 blocks → 16 slots, waste 6\n");
    printf("  请求 1 (长度 18): need 3 blocks → 24 slots, waste 6\n");
    printf("  请求 2 (长度 5):  need 1 block  →  8 slots, waste 3\n");
    printf("\n  [48] slots 分配, [33] 使用, [31.25]%% 浪费\n");
}

// ============================================================================
// Part 4: 显存公式 (了解即可)
// ============================================================================

/**
 * @brief 推导 KV Cache 显存公式
 *
 * 连续方案: L × H × N × S_max × D × 2 × sizeof(float)
 * 分页方案: L × H × (total_blocks) × block_size × D × 2 × sizeof(float)
 *
 * 关键差异: N × S_max   vs   total_blocks × block_size
 *   - 连续: 即使请求只用 100 tokens，也占 S_max=2048 个位置
 *   - 分页: total_blocks = sum(ceil(length[i] / block_size))
 *     = 所有请求需要的 block 总和，按实际使用分配
 *
 * 举例: L=32, H=32, D=128, N=100, S_max=2048
 *   连续: 32 × 32 × 100 × 2048 × 128 × 2 × 4 = ~2.1 TB   ← 不可能！
 *
 * 这就是为什么真实推理系统必须用分页。
 */

// ============================================================================
// Part 5: 测试
// ============================================================================

bool test_basic() {
    printf("--- Test: N=100, S_max=2048, D=128, L=32, H=32 ---\n");

    int N = 100, S_max = 2048, D = 128, L = 32, H = 32;
    int *lengths = new int[N];
    generate_request_lengths(lengths, N, S_max);

    // 连续方案
    long long cont_alloc, cont_used; float cont_waste;
    // TODO 3: 调用 analyze_continuous 并打印结果
    analyze_continuous(lengths,N,S_max,D,L,H,&cont_alloc, &cont_used,&cont_waste);
    // 分页方案 (block_size=16)
    long long page_alloc, page_used; float page_waste;
    // TODO 4: 调用 analyze_paged 并打印结果
    int block_size = 16;
    analyze_paged(lengths,N,S_max,block_size,D,L,H,&page_alloc, &page_used,&page_waste);
    // 统计
    int total_len = 0, max_len = 0, min_len = S_max;
    for (int i = 0; i < N; ++i) {
        total_len += lengths[i];
        if (lengths[i] > max_len) max_len = lengths[i];
        if (lengths[i] < min_len) min_len = lengths[i];
    }
    printf("  请求统计: 平均=%d, 最小=%d, 最大=%d, 总=%d\n",
           total_len / N, min_len, max_len, total_len);
    printf("  连续方案: 分配=%lld slots, 使用=%lld, 浪费=%.1f%%\n",
           cont_alloc, cont_used, cont_waste);
    printf("  分页方案: 分配=%lld slots, 使用=%lld, 浪费=%.1f%%\n",
           page_alloc, page_used, page_waste);

    // 分页浪费应该远小于连续
    assert(page_waste < cont_waste);

    delete[] lengths;
    return true;
}

bool test_small() {
    printf("\n--- Test: N=10, S_max=100, block_size=16 ---\n");

    int N = 10, S_max = 100, D = 4, L = 2, H = 4;
    int *lengths = new int[N];
    generate_request_lengths(lengths, N, S_max);

    long long cont_alloc, cont_used; float cont_waste;
    // TODO 5: 调用 analyze_continuous
    analyze_continuous(lengths,N,S_max,D,L,H,&cont_alloc, &cont_used,&cont_waste);
    long long page_alloc, page_used; float page_waste;
    // TODO 6: 调用 analyze_paged (各种 block_size)
    analyze_paged(lengths,N,S_max,16,D,L,H,&page_alloc, &page_used,&page_waste);
    printf("  block_size=16: 连续浪费=%.1f%%, 分页浪费=%.1f%%\n",
           cont_waste, page_waste);

    // 用不同 block_size 对比
    analyze_paged(lengths, N, S_max, 8, D, L, H, &page_alloc, &page_used, &page_waste);
    printf("  block_size=8:  分页浪费=%.1f%%\n", page_waste);

    analyze_paged(lengths, N, S_max, 32, D, L, H, &page_alloc, &page_used, &page_waste);
    printf("  block_size=32: 分页浪费=%.1f%%\n", page_waste);

    delete[] lengths;
    return true;
}

// ============================================================================
// main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Phase 3 Day 1: KV Cache 碎片问题\n");
    printf("========================================\n\n");

    visualize_paged_example();

    bool ok1 = test_small();
    bool ok2 = test_basic();

    printf("\n========================================\n");
    printf("Result: %s\n", (ok1 && ok2) ? "All tests passed" : "Some tests FAILED");
    printf("========================================\n");

    return (ok1 && ok2) ? 0 : 1;
}
