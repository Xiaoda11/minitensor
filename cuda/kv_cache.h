/**
 * @file kv_cache.h
 * @brief Phase 2 Week 1: KV Cache 数据结构（静态池）
 *
 * Learning objectives:
 *  - 理解 KV Cache 在内存中的布局
 *  - 掌握从 (layer, head, pos) 到 1D 偏移的指针计算
 *  - 预分配 max_seq_len，通过 current_len 管理"已使用"的空间
 *
 * 设计原理:
 *   K/V 只依赖自己 token 的 hidden state，与其他 token 无关。
 *   算过一次就缓存，decode 阶段复用，避免重复计算。
 *
 * 内存布局 (每个 layer 独立):
 *   K_cache[layer]  → [H][S_max][D]  扁平化为 1D 数组
 *   V_cache[layer]  → [H][S_max][D]  同上
 *
 *   K 总大小: L × H × S_max × D × sizeof(float)
 *   V 总大小: 同上
 *
 *   典型小模型: L=12, H=12, D=64, S_max=2048 → KV Cache ≈ 72 MB
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// 数据访问宏 — 从 (layer, head, pos) 算出在扁平的 1D 数组中的偏移
// ============================================================================
// 内存布局: [L][H][S_max][D]
//
//   第 0 层: head0 [0..D-1] [D..2D-1] ... head1 [...]
//   第 1 层: head0 [...]
//
// 一个 head 在一个 token 位置占据 D 个连续的 float。
// 同 head 内，不同 token 位置之间"不连续"（隔了 (H-1)*S_max*D + 其他 heads 的空间）。
//
// 三步定位 K[layer][head][pos][0]:
//   Step 1: 跳到第 layer 层的起点
//   Step 2: 在层内跳到第 head 个头的起点
//   Step 3: 在头内跳到第 pos 个 token 位置的起点
// ============================================================================

/**
 * @brief 计算 K[layer][head][pos][0] 在一维数组中的索引
 *
 * @param layer   层号 (0 .. L-1)
 * @param head    头号 (0 .. H-1)
 * @param pos     token 位置 (0 .. S_max-1)
 * @param H       heads per layer
 * @param D       head dimension
 * @param S_max   最大序列长度
 * @return        一维数组的起始索引
 */
inline int kv_offset(int layer, int head, int pos, int H, int D, int S_max) {
    // TODO: 填这行 — 用 layer, head, pos 算出正确偏移
    // 提示: 一层的大小 = H * S_max * D
    //       一个 head 在一层中的大小 = S_max * D
    //       一个 token 在一个 head 中的大小 = D
    return layer*H * S_max * D + head*S_max * D +  pos *D;
}

// ============================================================================
// KVCache 类
// ============================================================================

class KVCache {
public:
    /**
     * @brief 构造 KV Cache，预分配静态池
     *
     * @param L      Transformer 层数
     * @param H      每层 head 数
     * @param D      每个 head 的维度
     * @param S_max  最大序列长度（预分配）
     */
    KVCache(int L, int H, int D, int S_max)
        : num_layers_(L), num_heads_(H), head_dim_(D),
          max_seq_len_(S_max), current_len_(0)
    {
        size_t total = (size_t)L * H * S_max * D;
        K_data_ = new float[total]();
        V_data_ = new float[total]();
        printf("[KVCache] Allocated %.2f MB (L=%d H=%d D=%d S_max=%d)\n",
               (2.0f * total * sizeof(float)) / (1024 * 1024),
               L, H, D, S_max);
    }

    ~KVCache() {
        delete[] K_data_;
        delete[] V_data_;
    }

    // 禁止拷贝
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    // ---- 基本查询 ----

    int num_layers()   const { return num_layers_; }
    int num_heads()    const { return num_heads_; }
    int head_dim()     const { return head_dim_; }
    int max_seq_len()  const { return max_seq_len_; }
    int current_len()  const { return current_len_; }

    float* k_data()    { return K_data_; }
    float* v_data()    { return V_data_; }

    /**
     * @brief 重置缓存（开始新序列）
     * 只需要把 current_len 归零，旧数据会被新写入覆盖
     */
    void reset() {
        current_len_ = 0;
    }

    // ---- 核心操作 ----

    /**
     * @brief 存储一个 token 在某个 head 上的 K 和 V
     *
     * 调用场景:
     *   - Prefill: 对 prompt 的每个 token 调用
     *   - Decode:  对每个新生成的 token 调用
     *
     * @param layer  层号
     * @param head   头号
     * @param pos    token 位置（0-indexed，从序列开头算）
     * @param k      指向 K[layer][head][pos] 的 float 数组，长度 D
     * @param v      指向 V[layer][head][pos] 的 float 数组，长度 D
     */
    void store_kv(int layer, int head, int pos,
                  const float *k, const float *v) {
        // TODO: 填下面两行 — 用 kv_offset 算出起始索引
        int k_idx = kv_offset(layer, head, pos, num_heads_,  head_dim_ ,  max_seq_len_);
        int v_idx = kv_offset(layer, head, pos, num_heads_,  head_dim_ ,  max_seq_len_);

        // 复制 D 个元素
        memcpy(K_data_ + k_idx, k, head_dim_ * sizeof(float));
        memcpy(V_data_ + v_idx, v, head_dim_ * sizeof(float));
    }

    /**
     * @brief 获取某个位置的 K 指针（只读）
     */
    const float* get_k_ptr(int layer, int head, int pos) const {
        int idx = kv_offset(layer, head, pos, num_heads_, head_dim_, max_seq_len_);
        return K_data_ + idx;
    }

    /**
     * @brief 获取某个位置的 V 指针（只读）
     */
    const float* get_v_ptr(int layer, int head, int pos) const {
        int idx = kv_offset(layer, head, pos, num_heads_, head_dim_, max_seq_len_);
        return V_data_ + idx;
    }

    /**
     * @brief 推进序列长度（prefill 完成后或 decode 每步后调用）
     */
    void advance(int n) {
        current_len_ += n;
    }

    // ---- 调试 ----

    /**
     * @brief 校验: 写入后立即读回，检查 memcpy 是否损坏数据
     */
    bool verify_write_read(int layer, int head, int pos,
                           const float *expected_k, const float *expected_v) {
        const float *k_ptr = get_k_ptr(layer, head, pos);
        const float *v_ptr = get_v_ptr(layer, head, pos);
        for (int d = 0; d < head_dim_; ++d) {
            if (k_ptr[d] != expected_k[d] || v_ptr[d] != expected_v[d]) {
                printf("  Mismatch at layer=%d head=%d pos=%d dim=%d: "
                       "K(%f vs %f) V(%f vs %f)\n",
                       layer, head, pos, d,
                       k_ptr[d], expected_k[d], v_ptr[d], expected_v[d]);
                return false;
            }
        }
        return true;
    }

private:
    int num_layers_, num_heads_, head_dim_, max_seq_len_;
    int current_len_;
    float *K_data_;  // [L * H * S_max * D]
    float *V_data_;  // [L * H * S_max * D]
};
