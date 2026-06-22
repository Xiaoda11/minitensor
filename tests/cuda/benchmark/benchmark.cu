#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../kernels.h"

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

struct BenchCase {
    std::string op;
    std::string shape;
    std::string config;       // block × grid, threads, smem
    size_t bytes;             // total bytes read+write
    double flops;             // floating-point ops (0 if not computed)
    int warmup_iters;
    int measure_iters;
    void (*launcher)();
};

// ---- Static device pointers (allocated once, freed at end) ----
float *d_a, *d_b, *d_c;                    // vector_add
float *d_A, *d_B, *d_C;                     // matmul
float *d_input, *d_output;                  // softmax + layernorm
float *d_Q, *d_K, *d_V, *d_O;              // attention

// ---- Problem sizes ----
constexpr int VEC_N = 16 * 1024 * 1024;   // 16M
constexpr int MAT_M = 1024, MAT_N = 1024, MAT_K = 1024;
constexpr int SOFTMAX_ROWS = 1024, SOFTMAX_COLS = 1024;
constexpr int LN_ROWS = 1024, LN_COLS = 1024;
constexpr float LN_EPS = 1e-5f;
constexpr int ATT_S = 128, ATT_D = 64;

double bandwidth_gbps(size_t bytes, float latency_ms) {
    if (latency_ms <= 0.0f) return 0.0;
    return static_cast<double>(bytes) / (static_cast<double>(latency_ms) * 1e-3) / 1e9;
}

double compute_gflops(double total_flops, float latency_ms) {
    if (latency_ms <= 0.0f) return 0.0;
    return total_flops / (static_cast<double>(latency_ms) * 1e-3) / 1e9;
}

float measure_ms(const BenchCase& bench) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < bench.warmup_iters; ++i) {
        bench.launcher();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench.measure_iters; ++i) {
        bench.launcher();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(bench.measure_iters);
}

// ===================================================================
// Launchers
// ===================================================================

void launch_vector_add() {
    constexpr int threads = 256;
    int blocks = (VEC_N + threads - 1) / threads;
    vector_add_kernel_grid_stride<<<blocks, threads>>>(d_a, d_b, d_c, VEC_N);
}

void launch_matmul_naive() {
    dim3 block(16, 16);
    dim3 grid((MAT_N + 15) / 16, (MAT_M + 15) / 16);
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, MAT_M, MAT_K, MAT_N);
}

void launch_matmul_tiled() {
    dim3 block(16, 16);
    dim3 grid((MAT_N + 15) / 16, (MAT_M + 15) / 16);
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, MAT_M, MAT_K, MAT_N);
}

void launch_softmax() {
    int num_warps = SOFTMAX_COLS / 32;
    int smem = num_warps * static_cast<int>(sizeof(float));
    softmax_warp_reduce_kernel<<<SOFTMAX_ROWS, SOFTMAX_COLS, smem>>>(
        d_input, d_output, SOFTMAX_ROWS, SOFTMAX_COLS);
}

void launch_layernorm() {
    int num_warps = LN_COLS / 32;
    int smem = num_warps * static_cast<int>(sizeof(WelfordState));
    layernorm_welford_kernel<<<LN_ROWS, LN_COLS, smem>>>(
        d_input, d_output, LN_ROWS, LN_COLS, LN_EPS);
}

void launch_attention() {
    int block_dim = 256;
    int smem = static_cast<int>((ATT_D + 64 * ATT_D + 64 + ATT_D) * sizeof(float));
    attention_fused_kernel_v2<<<ATT_S, block_dim, smem>>>(
        d_Q, d_K, d_V, d_O, ATT_S, ATT_D);
}

}  // namespace

int main() {
    // ---- Query device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("// GPU: %s, CC %d.%d, %zu MB VRAM, %d SMs\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / (1024 * 1024),
                prop.multiProcessorCount);

    // ---- Allocate device memory ----
    size_t vec_bytes = static_cast<size_t>(VEC_N) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, vec_bytes));

    size_t mat_bytes = static_cast<size_t>(MAT_M) * MAT_K * sizeof(float);
    size_t mat_out_bytes = static_cast<size_t>(MAT_M) * MAT_N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, mat_out_bytes));

    size_t sm_bytes = static_cast<size_t>(SOFTMAX_ROWS) * SOFTMAX_COLS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, sm_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sm_bytes));

    size_t att_qkv_bytes = static_cast<size_t>(ATT_S) * ATT_D * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_Q, att_qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, att_qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, att_qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, att_qkv_bytes));

    // ---- Benchmark suite ----
    // FLOPs formulas:
    //   vector_add: N adds = N FLOPs
    //   matmul: 2*M*K*N (1 mul + 1 add per inner product element)
    //   softmax: rows * (cols*3 for exp/sub/div + ~5*cols for max/sum/norm) ≈ rows*cols*8
    //   layernorm: rows * (cols*3 for sub/sqr/sum + ~cols*5 for mean/var/norm) ≈ rows*cols*8
    //   attention: S*S*D*2 (Q@K) + S*S*8 (softmax) + S*D*S*2 (P@V)

    const std::vector<BenchCase> benches{
        {"vector_add", "[16M]",
         "256 thr × 65536 blk",
         16ull * 1024 * 1024 * sizeof(float) * 3,
         16.0 * 1024 * 1024,  // 16M FLOPs
         10, 100, launch_vector_add},

        {"matmul_naive", "[1024×1024]×[1024×1024]",
         "16×16 thr × 64×64 blk",
         3ull * 1024 * 1024 * sizeof(float),
         2.0 * MAT_M * MAT_K * MAT_N,  // 2*M*K*N
         3, 20, launch_matmul_naive},

        {"matmul_tiled", "[1024×1024]×[1024×1024]",
         "16×16 thr × 64×64 blk, tile=16",
         3ull * 1024 * 1024 * sizeof(float),
         2.0 * MAT_M * MAT_K * MAT_N,
         3, 20, launch_matmul_tiled},

        {"softmax", "[1024×1024]",
         "1024 thr × 1024 blk, smem=1 warps×4B",
         2ull * 1024 * 1024 * sizeof(float),
         8.0 * SOFTMAX_ROWS * SOFTMAX_COLS,  // ~8 ops per element
         10, 100, launch_softmax},

        {"layernorm", "[1024×1024]",
         "1024 thr × 1024 blk, smem=1 warps×WelfordState",
         2ull * 1024 * 1024 * sizeof(float),
         8.0 * LN_ROWS * LN_COLS,
         10, 100, launch_layernorm},

        {"attention", "[B=1,H=1,S=128,D=64]",
         "256 thr × 128 blk, smem=~17KB",
         4ull * 128 * 64 * sizeof(float),
         2.0 * ATT_S * ATT_S * ATT_D + 8.0 * ATT_S * ATT_S + 2.0 * ATT_S * ATT_D * ATT_S,
         10, 100, launch_attention},
    };

    // ---- Determine bottlenecks ----
    auto bottleneck = [](const BenchCase& b, double gbps, double gflops) -> std::string {
        if (b.op == "vector_add")
            return "memory bandwidth (3 arrays, 0 compute reuse)";

        if (b.op == "matmul_naive")
            return "global memory traffic (no tiling, O(MNK) reads, only ~12 GB/s HBM on RTX 2060)";

        if (b.op == "matmul_tiled")
            return "shared memory reuse + occupancy (tile=16, 2 warps/SM on 2060 → low occupancy)";

        if (b.op == "softmax")
            return "memory bandwidth (row reductions, warp-level parallelism helps but each row only 1 warp)";

        if (b.op == "layernorm")
            return "memory bandwidth (row reductions, Welford online mean/variance, 2-pass fuse)";

        if (b.op == "attention")
            return "QK reduction + softmax (fused kernel reduces HBM traffic but S×S compute dominates)";

        return "unknown";
    };

    std::puts("");
    std::puts("| Op | Shape | Config | Latency (us) | Bandwidth (GB/s) | GFLOPS | Bottleneck |");
    std::puts("| --- | --- | --- | ---: | ---: | ---: | --- |");

    for (const auto& bench : benches) {
        const float ms = measure_ms(bench);
        const double gbps = bandwidth_gbps(bench.bytes, ms);
        const double gflops = compute_gflops(bench.flops, ms);
        if (bench.flops > 0) {
            std::printf("| %s | %s | %s | %.2f | %.2f | %.2f | %s |\n",
                        bench.op.c_str(), bench.shape.c_str(), bench.config.c_str(),
                        ms * 1000.0f, gbps, gflops, bottleneck(bench, gbps, gflops).c_str());
        } else {
            std::printf("| %s | %s | %s | %.2f | %.2f | — | %s |\n",
                        bench.op.c_str(), bench.shape.c_str(), bench.config.c_str(),
                        ms * 1000.0f, gbps, bottleneck(bench, gbps, gflops).c_str());
        }
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return 0;
}
