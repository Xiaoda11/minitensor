#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
    size_t bytes;
    int warmup_iters;
    int measure_iters;
    void (*launcher)();
};

double bandwidth_gbps(size_t bytes, float latency_ms) {
    if (latency_ms <= 0.0f) return 0.0;
    return static_cast<double>(bytes) / (static_cast<double>(latency_ms) * 1e-3) / 1e9;
}

float measure_ms(const BenchCase& bench) {
    cudaEvent_t start;
    cudaEvent_t stop;
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

void launch_vector_add() {
    // TODO: allocate inputs once, call vector_add_kernel or grid-stride variant.
}

void launch_matmul_naive() {
    // TODO: call matmul_naive_kernel with M=N=K=1024.
}

void launch_matmul_tiled() {
    // TODO: call matmul_tiled_kernel or matmul_tiled_unroll_kernel.
}

void launch_softmax() {
    // TODO: call softmax_naive_kernel / warp reduction variant on [1024, 1024].
}

void launch_layernorm() {
    // TODO: call layernorm_naive_kernel / warp reduction / Welford variant.
}

void launch_attention() {
    // TODO: call attention_fused_kernel or v2 with batch=1, seq_len=128, head_dim=64.
}

}  // namespace

int main() {
    const std::vector<BenchCase> benches{
        {"vector_add", "[16M]", 16ull * 1024 * 1024 * sizeof(float) * 3, 10, 100, launch_vector_add},
        {"matmul_naive", "[1024x1024]x[1024x1024]", 3ull * 1024 * 1024 * sizeof(float), 3, 20, launch_matmul_naive},
        {"matmul_tiled", "[1024x1024]x[1024x1024]", 3ull * 1024 * 1024 * sizeof(float), 3, 20, launch_matmul_tiled},
        {"softmax", "[1024x1024]", 2ull * 1024 * 1024 * sizeof(float), 10, 100, launch_softmax},
        {"layernorm", "[1024x1024]", 2ull * 1024 * 1024 * sizeof(float), 10, 100, launch_layernorm},
        {"attention", "[B=1,H=1,S=128,D=64]", 4ull * 128 * 64 * sizeof(float), 10, 100, launch_attention},
    };

    std::puts("| Device | Op | Shape | Latency (us) | Bandwidth (GB/s) | Bottleneck |");
    std::puts("| --- | --- | --- | ---: | ---: | --- |");
    for (const auto& bench : benches) {
        const float ms = measure_ms(bench);
        const double gbps = bandwidth_gbps(bench.bytes, ms);
        std::printf("| CUDA | %s | %s | %.2f | %.2f | TODO |\n",
                    bench.op.c_str(), bench.shape.c_str(), ms * 1000.0f, gbps);
    }

    return 0;
}
