/**
 * @file continuous_batching.cu
 * @brief Phase 3 Day 4: Continuous Batching 请求调度
 *
 * 目标: 理解推理引擎如何同时服务多个请求——请求来了就加进来，
 *       生成完了就移出去，不等待、不阻塞。
 *
 * 核心概念:
 *   连续方案: 等一批请求全部完成再开始下一批
 *             → GPU 空闲等慢请求，吞吐极低
 *
 *   Continuous Batching: 请求完成立即移出，新请求立即加入
 *             → GPU 始终满载，吞吐最高
 *
 * 编译:
 *   g++ -std=c++17 -O2 -x c++ continuous_batching.cu -o continuous_batching && ./continuous_batching
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// ============================================================================
// 常量
// ============================================================================

constexpr int MAX_REQUESTS = 8;       // 最多同时在线请求
constexpr int BLOCK_SIZE = 16;        // 每 block 16 个 slot
constexpr int MAX_BLOCKS = 32;        // 总共 32 个物理 block

// ============================================================================
// Part 1: 请求状态机
// ============================================================================

enum class ReqState {
    WAITING,    // 还没到达（或排队中）
    PREFILL,    // 正在处理 prompt（一次性）
    DECODE,     // 逐 token 生成中
    DONE        // 生成完毕
};

/**
 * 一个推理请求
 *
 * prompt_len:    prompt 的 token 数（比如 "写一首诗" = 4 tokens）
 * max_new_tokens: 最多生成多少个新 token
 * tokens_generated: 已经生成了几个
 * blocks_allocated: 占用了几个 KV Cache block
 * arrival_step:  在第几步到达系统
 */
struct Request {
    int id;
    int prompt_len;
    int max_new_tokens;
    int tokens_generated;
    int blocks_allocated;
    int arrival_step;
    ReqState state;

    void init(int _id, int _prompt_len, int _max_tokens, int _arrival) {
        id = _id;
        prompt_len = _prompt_len;
        max_new_tokens = _max_tokens;
        tokens_generated = 0;
        blocks_allocated = 0;
        arrival_step = _arrival;
        state = ReqState::WAITING;
    }
};

// ============================================================================
// Part 2: 简陋的 KV Cache 管理器（只跟踪 block 数量，不存实际数据）
// ============================================================================

class SimpleCacheMgr {
private:
    int free_blocks;
    int total_blocks;
public:
    void init(int total) {
        total_blocks = total;
        free_blocks = total;
    }

    // 分配 num 个 block，成功返回 true
    bool alloc(int num) {
        if (free_blocks < num) return false;
        free_blocks -= num;
        return true;
    }

    void free(int num) {
        free_blocks += num;
        if (free_blocks > total_blocks) free_blocks = total_blocks;
    }

    int available() const { return free_blocks; }
};

// ============================================================================
// Part 3: Continuous Batching 调度器
// ============================================================================

class Scheduler {
private:
    Request requests[MAX_REQUESTS];
    int num_requests;          // 总共注册的请求数（包括还没到达的）
    int active_count;          // 当前在 PREFILL 或 DECODE 状态的请求数
    SimpleCacheMgr cache;
    int current_step;

    // 辅助：计算请求需要的 block 数
    int blocks_needed(int total_tokens) const {
        return (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

public:
    void init() {
        num_requests = 0;
        active_count = 0;
        current_step = 0;
        cache.init(MAX_BLOCKS);
    }

    void add_request(int prompt_len, int max_tokens, int arrival_step) {
        requests[num_requests].init(num_requests, prompt_len, max_tokens, arrival_step);
        num_requests++;
    }

    /**
     * @brief 执行一个调度步
     *
     * 每一步做四件事:
     *   1. 检查是否有新请求到达 → 尝试分配 KV Cache, 进入 PREFILL
     *   2. PREFILL 请求完成 → 转入 DECODE
     *   3. DECODE 请求生成一个 token
     *   4. 完成的请求释放 KV Cache 并移出
     *
     * 打印每一步的状态变化。
     */
    void step() {
        printf("--- Step %d ---\n", current_step);

        // ================================================================
        // TODO 1: 新请求到达 → 加入 prefill
        // ================================================================
        //
        // 遍历所有请求，找到 state==WAITING 且 arrival_step==current_step 的
        // 尝试分配 KV Cache block:
        //   blocks = (prompt_len + BLOCK_SIZE - 1) / BLOCK_SIZE
        //   如果 cache.alloc(blocks) 成功 → state = PREFILL
        //   如果不够 → 打印 "waiting for memory"，留在 WAITING
        //

        // --- 你的代码 ---

         for (int i = 0; i < num_requests; ++i) {
             if (requests[i].state == ReqState::WAITING
                 && requests[i].arrival_step == current_step) {
                 int blocks = ((requests[i].prompt_len) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                 if (cache.alloc(blocks)) {
                     requests[i].blocks_allocated = blocks;
                     requests[i].state = ReqState::PREFILL;
                     active_count++;
                     printf("  [req %d] 到达, prompt=%d → 分配 %d blocks, PREFILL\n", requests[i].id,requests[i].prompt_len,blocks);
                 } else {
                     printf("  [req %d] 到达, 显存不足, 等待...\n", requests[i].id);
                 }
             }
         }


        // --- 你的代码结束 ---

        // ================================================================
        // TODO 2: Prefill → Decode 转换
        // ================================================================
        //
        // 遍历所有状态为 PREFILL 的请求，prefill 一步完成，转入 DECODE
        // （prefill 一次性处理所有 prompt tokens）
        //

        // --- 你的代码 ---
         for (int i = 0; i < num_requests; ++i) {
             if (requests[i].state == ReqState::PREFILL) {
                 requests[i].state = ReqState::DECODE;
                 printf("  [req %d] prefill 完成, 转入 DECODE\n", requests[i].id);
             }
         }


        // --- 你的代码结束 ---

        // ================================================================
        // TODO 3: Decode → 生成一个 token
        // ================================================================
        //
        // 遍历所有 DECODE 状态的请求，每个生成 1 个 token
        // tokens_generated++
        // 如果 tokens_generated == max_new_tokens → state = DONE
        //

        // --- 你的代码 ---
         for (int i = 0; i < num_requests; ++i) {
             if (requests[i].state == ReqState::DECODE) {
                 requests[i].tokens_generated++;
                 printf("  [req %d] DECODE: token %d/%d\n",
                        requests[i].id, requests[i].tokens_generated,
                        requests[i].max_new_tokens);
                 if (requests[i].tokens_generated == requests[i].max_new_tokens) {
                     requests[i].state = ReqState::DONE;
                     printf("  [req %d] 生成完毕!\n", requests[i].id);
                 }
             }
         }

        // --- 你的代码结束 ---

        // ================================================================
        // TODO 4: 完成请求 → 释放资源
        // ================================================================
        //
        // 遍历所有 DONE 状态的请求
        // 释放 KV Cache blocks, active_count--
        // 将状态设为 WAITING（可选，这里直接标记为 DONE 不再复用）
        //

        // --- 你的代码 ---
         for (int i = 0; i < num_requests; ++i) {
            if (requests[i].state == ReqState::DONE && requests[i].blocks_allocated > 0) {
                cache.free(requests[i].blocks_allocated);
                active_count--;
                printf("  [req %d] 释放 %d blocks, 剩余空闲=%d\n",
                       requests[i].id, requests[i].blocks_allocated,
                        cache.available());
                requests[i].blocks_allocated = 0;  // 防止重复释放
             }
         }

        // --- 你的代码结束 ---

        // 打印当前状态摘要
        printf("  活跃请求: %d, 空闲 block: %d/%d\n",
               active_count, cache.available(), MAX_BLOCKS);
        printf("\n");

        current_step++;
    }

    // 检查是否所有请求都完成了
    bool all_done() const {
        for (int i = 0; i < num_requests; ++i) {
            if (requests[i].state != ReqState::DONE
                && requests[i].state != ReqState::WAITING)
                return false;
        }
        // 还有没到达的请求也不算全完成
        for (int i = 0; i < num_requests; ++i) {
            if (requests[i].arrival_step >= current_step
                && requests[i].state != ReqState::DONE)
                return false;
        }
        return true;
    }

    int get_step() const { return current_step; }
};

// ============================================================================
// Part 4: 测试 — 模拟多请求并发
// ============================================================================

/**
 * 场景: 3 个请求，不同时刻到达，不同长度
 *
 *   请求 0: prompt=10, max_tokens=5,  step 0 到达
 *   请求 1: prompt=20, max_tokens=3,  step 1 到达 (延迟到达)
 *   请求 2: prompt=16, max_tokens=4,  step 3 到达 (更晚到达)
 *
 * 观察:
 *   - Step 0: 只有请求 0 在跑 (prefill → decode)
 *   - Step 1: 请求 1 加入，和请求 0 并行解码
 *   - Step 3: 请求 2 加入，此时请求 0/1 可能还在跑或已完成
 *   - 请求完成后立即释放 KV Cache，新请求可以复用
 */
void run_simulation() {
    Scheduler sched;
    sched.init();

    // Step 0: 请求 0 到达
    sched.add_request(10, 5, 0);
    // Step 1: 请求 1 到达
    sched.add_request(20, 3, 1);
    // Step 3: 请求 2 到达
    sched.add_request(16, 4, 3);

    // 跑 15 步，或直到全部完成
    for (int i = 0; i < 15; ++i) {
        sched.step();
        if (sched.all_done() && sched.get_step() > 5) {
            printf("所有请求处理完毕。\n");
            break;
        }
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Phase 3 Day 4: Continuous Batching\n");
    printf("========================================\n");
    printf("场景: 3 个请求, 不同时刻到达\n");
    printf("请求0: prompt=10, max_tokens=5, step=0 到达\n");
    printf("请求1: prompt=20, max_tokens=3, step=1 到达\n");
    printf("请求2: prompt=16, max_tokens=4, step=3 到达\n");
    printf("BLOCK_SIZE=%d, 物理 block 总数=%d\n\n", BLOCK_SIZE, MAX_BLOCKS);

    run_simulation();

    printf("========================================\n");
    printf("Simulation ended\n");
    printf("========================================\n");

    return 0;
}
