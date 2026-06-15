/**
 * @file phase2_main.cu
 * @brief Phase 2 Week 1 Day 2: KV Cache 数据结构测试
 *
 * 测试内容:
 *   Test 1: 单 head 基本写入/读取 — 验证 store_kv 和 get_*_ptr
 *   Test 2: 多 head 隔离 — 不同 head 的 K/V 不应互相覆盖
 *   Test 3: 多 layer 隔离 — 不同 layer 的 K/V 不应互相覆盖
 *   Test 4: 边界测试 — pos=0 和 pos=S_max-1 的正确性
 *
 * 编译 (纯 CPU 代码，不需要 nvcc):
 *   g++ -std=c++17 -O2 phase2_main.cu -o phase2_main && ./phase2_main
 */

#include "kv_cache.h"
#include <cmath>
#include <cassert>

// ============================================================================
// Test 1: 基本写入/读取（单 head, 单 layer）
// ============================================================================
bool test1_basic_store_retrieve() {
    printf("--- Test 1: Basic Store & Retrieve ---\n");

    int L = 1, H = 1, D = 4, S_max = 8;
    KVCache cache(L, H, D, S_max);

    // 写入 pos=0 的 K 和 V
    float k0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v0[] = {5.0f, 6.0f, 7.0f, 8.0f};
    cache.store_kv(0, 0, 0, k0, v0);

    // 写入 pos=3 的 K 和 V
    float k3[] = {9.0f, 10.0f, 11.0f, 12.0f};
    float v3[] = {13.0f, 14.0f, 15.0f, 16.0f};
    cache.store_kv(0, 0, 3, k3, v3);

    // 验证 pos=0
    bool ok = cache.verify_write_read(0, 0, 0, k0, v0);
    // 验证 pos=3
    ok = cache.verify_write_read(0, 0, 3, k3, v3) && ok;

    printf("  Test 1: %s\n\n", ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test 2: 多 head 隔离
// ============================================================================
bool test2_multi_head_isolation() {
    printf("--- Test 2: Multi-Head Isolation ---\n");

    int L = 1, H = 4, D = 4, S_max = 4;
    KVCache cache(L, H, D, S_max);

    // 每个 head 写入不同的数据
    float k[4][4] = {
        {1, 1, 1, 1},
        {2, 2, 2, 2},
        {3, 3, 3, 3},
        {4, 4, 4, 4}
    };
    float v[4][4] = {
        {10, 10, 10, 10},
        {20, 20, 20, 20},
        {30, 30, 30, 30},
        {40, 40, 40, 40}
    };

    // 每个 head 在 pos=0 写入
    for (int h = 0; h < H; ++h) {
        cache.store_kv(0, h, 0, k[h], v[h]);
    }

    // 验证: head 2 读回的是 {3,3,3,3} 而不是别人的数据
    bool ok = cache.verify_write_read(0, 2, 0, k[2], v[2]);

    // 额外验证: 改变 head 0 的数据不影响 head 1
    float new_k1[] = {99, 99, 99, 99};
    float new_v1[] = {88, 88, 88, 88};
    cache.store_kv(0, 1, 0, new_k1, new_v1);
    ok = cache.verify_write_read(0, 1, 0, new_k1, new_v1) && ok;
    // head 0 应该保持不变
    ok = cache.verify_write_read(0, 0, 0, k[0], v[0]) && ok;

    printf("  Test 2: %s\n\n", ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test 3: 多 layer 隔离
// ============================================================================
bool test3_multi_layer_isolation() {
    printf("--- Test 3: Multi-Layer Isolation ---\n");

    int L = 3, H = 2, D = 4, S_max = 4;
    KVCache cache(L, H, D, S_max);

    // 每层每个 head 写入 layer*100 + head*10 的模式
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            float k[4], v[4];
            float val = l * 100.0f + h * 10.0f;
            for (int d = 0; d < D; ++d) {
                k[d] = val + d;
                v[d] = val + d + 1000;
            }
            cache.store_kv(l, h, 0, k, v);
        }
    }

    // 验证 layer 2 head 1 的数据
    float expected_k[] = {210, 211, 212, 213};
    float expected_v[] = {1210, 1211, 1212, 1213};
    bool ok = cache.verify_write_read(2, 1, 0, expected_k, expected_v);

    // 验证 layer 0 head 0 没被覆盖
    float expected_k00[] = {0, 1, 2, 3};
    float expected_v00[] = {1000, 1001, 1002, 1003};
    ok = cache.verify_write_read(0, 0, 0, expected_k00, expected_v00) && ok;

    printf("  Test 3: %s\n\n", ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test 4: 边界测试 — pos=0 和 pos=S_max-1
// ============================================================================
bool test4_boundary_positions() {
    printf("--- Test 4: Boundary Positions ---\n");

    int L = 1, H = 1, D = 4, S_max = 8;
    KVCache cache(L, H, D, S_max);

    // 写入 pos=0
    float k0[] = {1, 2, 3, 4};
    float v0[] = {10, 20, 30, 40};
    cache.store_kv(0, 0, 0, k0, v0);

    // 写入 pos=7 (S_max - 1)
    float k7[] = {7, 8, 9, 10};
    float v7[] = {70, 80, 90, 100};
    cache.store_kv(0, 0, 7, k7, v7);

    // 验证 pos=0 没被 pos=7 影响
    bool ok = cache.verify_write_read(0, 0, 0, k0, v0);
    ok = cache.verify_write_read(0, 0, 7, k7, v7) && ok;

    printf("  Test 4: %s\n\n", ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test 5: advance + reset
// ============================================================================
bool test5_advance_reset() {
    printf("--- Test 5: Advance & Reset ---\n");

    int L = 1, H = 1, D = 4, S_max = 8;
    KVCache cache(L, H, D, S_max);

    assert(cache.current_len() == 0);

    cache.advance(5);
    assert(cache.current_len() == 5);

    cache.advance(1);
    assert(cache.current_len() == 6);

    cache.reset();
    assert(cache.current_len() == 0);

    printf("  Test 5: PASSED\n\n");
    return true;
}

// ============================================================================
// Test 6: 综合场景 — 模拟一次 prefill (填充 4 个 token, 2 layers, 3 heads)
// ============================================================================
bool test6_integration_prefill() {
    printf("--- Test 6: Integration (Simulated Prefill) ---\n");

    int L = 2, H = 3, D = 4, S_max = 8;
    KVCache cache(L, H, D, S_max);

    // 模拟 prefill: prompt 有 4 个 token
    int prompt_len = 4;

    // 为每个 (layer, head, pos) 写入 K 和 V
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            for (int pos = 0; pos < prompt_len; ++pos) {
                float k[4], v[4];
                // 编码: K = pos*100 + h*10 + d, V = K + 1000
                for (int d = 0; d < D; ++d) {
                    k[d] = pos * 100.0f + h * 10.0f + d;
                    v[d] = k[d] + 1000.0f;
                }
                cache.store_kv(l, h, pos, k, v);
            }
        }
    }

    cache.advance(prompt_len);
    assert(cache.current_len() == prompt_len);

    // 抽查几个位置
    bool ok = true;

    // Layer 0 Head 0 Pos 0: K = [0, 1, 2, 3]
    {
        float ek[] = {0, 1, 2, 3};
        float ev[] = {1000, 1001, 1002, 1003};
        ok = cache.verify_write_read(0, 0, 0, ek, ev) && ok;
    }

    // Layer 1 Head 2 Pos 3: K = 300 + 20 + d = [320, 321, 322, 323]
    {
        float ek[] = {320, 321, 322, 323};
        float ev[] = {1320, 1321, 1322, 1323};
        ok = cache.verify_write_read(1, 2, 3, ek, ev) && ok;
    }

    // Layer 0 Head 1 Pos 2: K = 200 + 10 + d = [210, 211, 212, 213]
    {
        float ek[] = {210, 211, 212, 213};
        float ev[] = {1210, 1211, 1212, 1213};
        ok = cache.verify_write_read(0, 1, 2, ek, ev) && ok;
    }

    printf("  Test 6: %s\n\n", ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("========================================\n");
    printf("Phase 2 Day 2: KV Cache Test Suite\n");
    printf("========================================\n\n");

    int passed = 0, total = 0;

    if (test1_basic_store_retrieve())   passed++; total++;
    if (test2_multi_head_isolation())   passed++; total++;
    if (test3_multi_layer_isolation())  passed++; total++;
    if (test4_boundary_positions())     passed++; total++;
    if (test5_advance_reset())          passed++; total++;
    if (test6_integration_prefill())    passed++; total++;

    printf("========================================\n");
    printf("Result: %d/%d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
