/**
 * @file inference_benchmark.cpp
 * @brief Phase 4 Week 1: Prefill vs Decode 延迟对比
 *
 * 面试考点: prefill 是 compute-bound (O(S²·D))，decode 是 memory-bound (O(S·D))
 * 数据说明: 随着序列增长，prefill 延迟平方增长，decode 线性增长
 *
 * 编译: cmake .. && make minitensor_cpu_inference_benchmark && ./cpu/minitensor_cpu_inference_benchmark
 */

#include "../cuda/kv_cache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile float g_sink = 0.0f;

// ============================================================================
// CPU Attention 函数 (从 prefill.cu / decode.cu 提取)
// ============================================================================

// Prefill 版: Q[S×D] @ K[S×D]^T → O[S×D]
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

// Decode 版: Q[D] @ K[S×D]^T → O[D]
void decode_attention_cpu(const float *Q, const float *K, const float *V,
                          float *output, int S, int D) {
    float scale = 1.0f / sqrtf((float)D);
    float *scores = new float[S];
    for (int j = 0; j < S; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d)
            dot += Q[d] * K[j * D + d];
        scores[j] = dot * scale;
    }

    float max_val = scores[0];
    for (int j = 1; j < S; ++j)
        if (scores[j] > max_val) max_val = scores[j];

    float *probs = new float[S];
    float sum_exp = 0.0f;
    for (int j = 0; j < S; ++j) {
        probs[j] = expf(scores[j] - max_val);
        sum_exp += probs[j];
    }
    for (int j = 0; j < S; ++j)
        probs[j] /= sum_exp;

    for (int d = 0; d < D; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < S; ++j)
            sum += probs[j] * V[j * D + d];
        output[d] = sum;
    }

    delete[] scores;
    delete[] probs;
}

// ============================================================================
// Benchmark 工具
// ============================================================================

template <typename Fn>
double measure_us(Fn&& fn, int warmup_iters, int measure_iters) {
    for (int i = 0; i < warmup_iters; ++i)
        g_sink += fn();

    const auto start = Clock::now();
    for (int i = 0; i < measure_iters; ++i)
        g_sink += fn();
    const auto stop = Clock::now();

    return std::chrono::duration<double, std::micro>(stop - start).count()
           / static_cast<double>(measure_iters);
}

void fill_random(float *data, int n, float scale = 1.0f) {
    for (int i = 0; i < n; ++i)
        data[i] = scale * static_cast<float>((i % 1024) - 512) / 512.0f;
}

// ============================================================================
// Prefill Benchmark: 不同 S 下的完整 attention
// ============================================================================

struct PrefillResult {
    int S;
    int D;
    double latency_us;
    double gflops;
    bool is_prefill;
};

PrefillResult bench_prefill(int S, int D, int warmup, int iters) {
    float *Q = new float[S * D];
    float *K = new float[S * D];
    float *V = new float[S * D];
    float *O = new float[S * D];
    fill_random(Q, S * D);
    fill_random(K, S * D);
    fill_random(V, S * D);

    double us = measure_us([&]() {
        attention_cpu(Q, K, V, O, S, D);
        return O[0];
    }, warmup, iters);

    // FLOPs: Q@K^T = S*S*D*2, softmax ≈ S*S*5, P@V = S*D*S*2
    double total_flops = 2.0 * S * S * D   // Q @ K^T
                       + 5.0 * S * S       // softmax (exp + add + div)
                       + 2.0 * S * D * S;  // P @ V
    double gflops = total_flops / (us * 1e-6) / 1e9;

    delete[] Q; delete[] K; delete[] V; delete[] O;
    return {S, D, us, gflops, true};
}

// ============================================================================
// Decode Benchmark: 不同 cache size 下的单 token attention
// ============================================================================

struct DecodeResult {
    int cache_len;
    int D;
    double latency_us;
    double bandwidth_gbps;
    bool is_prefill;
};

DecodeResult bench_decode(int cache_len, int D, int warmup, int iters) {
    float *Q = new float[D];           // 1 token query
    float *K = new float[cache_len * D];
    float *V = new float[cache_len * D];
    float *O = new float[D];
    fill_random(Q, D);
    fill_random(K, cache_len * D);
    fill_random(V, cache_len * D);

    double us = measure_us([&]() {
        decode_attention_cpu(Q, K, V, O, cache_len, D);
        return O[0];
    }, warmup, iters);

    // 内存: Q[D] + K[cache_len*D] + V[cache_len*D] + O[D]
    size_t bytes = (2 + 2 * cache_len) * static_cast<size_t>(D) * sizeof(float);
    double gbps = static_cast<double>(bytes) / (us * 1e-6) / 1e9;

    delete[] Q; delete[] K; delete[] V; delete[] O;
    return {cache_len, D, us, gbps, false};
}

// ============================================================================
// 输出
// ============================================================================

void print_section(const char *title) {
    std::cout << "\n## " << title << "\n\n";
}

void print_prefill_markdown(const std::vector<PrefillResult>& results) {
    std::cout << "| Phase | Seq Len (S) | Head Dim (D) | Latency (us) | GFLOPS | Bottleneck |\n";
    std::cout << "| --- | ---: | ---: | ---: | ---: | --- |\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& r : results) {
        std::cout << "| Prefill | " << r.S
                  << " | " << r.D
                  << " | " << r.latency_us
                  << " | " << r.gflops
                  << " | O(S²·D) compute-bound (Q@K^T dominates) |\n";
    }
}

void print_decode_markdown(const std::vector<DecodeResult>& results) {
    std::cout << "| Phase | Cache Len (S) | Head Dim (D) | Latency (us) | Bandwidth (GB/s) | Bottleneck |\n";
    std::cout << "| --- | ---: | ---: | ---: | ---: | --- |\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& r : results) {
        std::cout << "| Decode | " << r.cache_len
                  << " | " << r.D
                  << " | " << r.latency_us
                  << " | " << r.bandwidth_gbps
                  << " | O(S·D) memory-bound (K/V cache load dominates) |\n";
    }
}

}  // namespace

int main() {
    const int D = 64;

    // -------- Prefill: S=128,256,512,1024 --------
    print_section("Prefill (Full Attention Q[S×D] @ K[S×D]^T)");
    std::vector<PrefillResult> prefill_results;
    for (int S : {128, 256, 512, 1024}) {
        int warmup = (S <= 256) ? 5 : 2;
        int iters  = (S <= 256) ? 20 : 5;
        prefill_results.push_back(bench_prefill(S, D, warmup, iters));
    }
    print_prefill_markdown(prefill_results);

    // -------- Decode: cache size=128,256,512,1024 --------
    print_section("Decode (Single Token Q[D] @ K_cache[S×D]^T)");
    std::vector<DecodeResult> decode_results;
    for (int S : {128, 256, 512, 1024}) {
        int iters = (S <= 512) ? 100 : 50;
        decode_results.push_back(bench_decode(S, D, 10, iters));
    }
    print_decode_markdown(decode_results);

    // -------- 交叉对比表 --------
    print_section("Prefill vs Decode 交叉对比");
    std::cout << "| Seq Len (S) | Prefill (us) | Decode (us) | Ratio (prefill/decode) |\n";
    std::cout << "| ---: | ---: | ---: | ---: |\n";
    for (size_t i = 0; i < prefill_results.size(); ++i) {
        double ratio = prefill_results[i].latency_us / decode_results[i].latency_us;
        std::cout << "| " << prefill_results[i].S
                  << " | " << prefill_results[i].latency_us
                  << " | " << decode_results[i].latency_us
                  << " | " << ratio << "× |\n";
    }

    return static_cast<int>(g_sink) == 123456789 ? 1 : 0;
}
