#include "tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile float g_sink = 0.0f;

struct BenchResult {
    std::string op;
    std::string shape;
    double latency_us;
    double bandwidth_gbps;
    std::string bottleneck;
};

std::string shape_to_string(const std::vector<int>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += std::to_string(shape[i]);
        if (i + 1 < shape.size()) out += "x";
    }
    out += "]";
    return out;
}

template <typename Fn>
double measure_us(Fn&& fn, int warmup_iters, int measure_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        g_sink += fn();
    }

    const auto start = Clock::now();
    for (int i = 0; i < measure_iters; ++i) {
        g_sink += fn();
    }
    const auto stop = Clock::now();

    const auto total_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    return total_us / static_cast<double>(measure_iters);
}

double bandwidth_gbps(size_t bytes, double latency_us) {
    if (latency_us <= 0.0) return 0.0;
    return static_cast<double>(bytes) / (latency_us * 1e-6) / 1e9;
}

Tensor<float> make_tensor(const std::vector<int>& shape, float scale = 1.0f) {
    Tensor<float> tensor(shape);
    for (int i = 0; i < tensor.size(); ++i) {
        tensor.data()[i] = scale * static_cast<float>((i % 1024) - 512) / 512.0f;
    }
    return tensor;
}

BenchResult bench_add() {
    const std::vector<int> shape{1024, 1024};
    auto a = make_tensor(shape, 0.5f);
    auto b = make_tensor(shape, 0.25f);
    const int iters = 100;
    const double us = measure_us([&]() {
        auto c = a + b;
        return c.data()[0];
    }, 10, iters);
    const size_t bytes = static_cast<size_t>(a.size()) * sizeof(float) * 3;
    return {"add", shape_to_string(shape), us, bandwidth_gbps(bytes, us), "memory bandwidth"};
}

BenchResult bench_mul() {
    const std::vector<int> shape{1024, 1024};
    auto a = make_tensor(shape, 0.5f);
    auto b = make_tensor(shape, 0.25f);
    const int iters = 100;
    const double us = measure_us([&]() {
        auto c = a * b;
        return c.data()[0];
    }, 10, iters);
    const size_t bytes = static_cast<size_t>(a.size()) * sizeof(float) * 3;
    return {"mul", shape_to_string(shape), us, bandwidth_gbps(bytes, us), "memory bandwidth"};
}

BenchResult bench_matmul() {
    const int m = 256;
    const int k = 256;
    const int n = 256;
    auto a = make_tensor({m, k}, 0.01f);
    auto b = make_tensor({k, n}, 0.02f);
    const int iters = 5;
    const double us = measure_us([&]() {
        auto c = matmul(a, b);
        return c.data()[0];
    }, 1, iters);
    const size_t bytes = static_cast<size_t>(m * k + k * n + m * n) * sizeof(float);
    return {"matmul", "[256x256]x[256x256]", us, bandwidth_gbps(bytes, us), "compute / cache locality"};
}

BenchResult bench_matmul_blocked() {
    const int m = 256;
    const int k = 256;
    const int n = 256;
    auto a = make_tensor({m, k}, 0.01f);
    auto b = make_tensor({k, n}, 0.02f);
    const int iters = 5;
    const double us = measure_us([&]() {
        auto c = matmul_blocked(a, b, 32);
        return c.data()[0];
    }, 1, iters);
    const size_t bytes = static_cast<size_t>(m * k + k * n + m * n) * sizeof(float);
    return {"matmul_blocked", "[256x256]x[256x256], tile=32", us, bandwidth_gbps(bytes, us), "compute / cache reuse"};
}

BenchResult bench_softmax() {
    const std::vector<int> shape{1024, 1024};
    auto input = make_tensor(shape, 0.01f);
    const int iters = 20;
    const double us = measure_us([&]() {
        auto output = softmax(input);
        return output.data()[0];
    }, 3, iters);
    const size_t bytes = static_cast<size_t>(input.size()) * sizeof(float) * 2;
    return {"softmax", shape_to_string(shape), us, bandwidth_gbps(bytes, us), "exp latency + row reductions"};
}

BenchResult bench_layernorm() {
    const std::vector<int> shape{1024, 1024};
    auto input = make_tensor(shape, 0.01f);
    Tensor<float> weight({shape[1]});
    Tensor<float> bias({shape[1]});
    weight.fill(1.0f);
    bias.fill(0.0f);
    const int iters = 20;
    const double us = measure_us([&]() {
        auto output = layernorm(input, weight, bias, 1e-5f);
        return output.data()[0];
    }, 3, iters);
    const size_t bytes = static_cast<size_t>(input.size()) * sizeof(float) * 2
                         + static_cast<size_t>(shape[1]) * sizeof(float) * 2;
    return {"layernorm", shape_to_string(shape), us, bandwidth_gbps(bytes, us), "row reductions + sqrt"};
}

BenchResult bench_transpose() {
    const std::vector<int> shape{256, 512};
    auto input = make_tensor(shape, 0.01f);
    const int iters = 50;
    const double us = measure_us([&]() {
        auto output = input.transpose(0, 1);
        return output.data()[0];
    }, 5, iters);
    const size_t bytes = static_cast<size_t>(input.size()) * sizeof(float) * 2;
    return {"transpose", "[256x512] -> [512x256]", us, bandwidth_gbps(bytes, us), "strided writes"};
}

BenchResult bench_reshape() {
    const std::vector<int> shape{1024, 1024};
    auto input = make_tensor(shape, 0.01f);
    const int iters = 100;
    const double us = measure_us([&]() {
        auto output = input.reshape({512, 2048});
        return output.data()[0];
    }, 10, iters);
    const size_t bytes = static_cast<size_t>(input.size()) * sizeof(float) * 2;
    return {"reshape", "[1024x1024] -> [512x2048]", us, bandwidth_gbps(bytes, us), "full copy"};
}

void print_markdown(const std::vector<BenchResult>& results) {
    std::cout << "| Device | Op | Shape | Latency (us) | Bandwidth (GB/s) | Bottleneck |" << '\n';
    std::cout << "| --- | --- | --- | ---: | ---: | --- |" << '\n';
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& result : results) {
        std::cout << "| CPU | " << result.op
                  << " | " << result.shape
                  << " | " << result.latency_us
                  << " | " << result.bandwidth_gbps
                  << " | " << result.bottleneck
                  << " |" << '\n';
    }
}

}  // namespace

int main() {
    const std::vector<BenchResult> results{
        bench_add(),
        bench_mul(),
        bench_matmul(),
        bench_matmul_blocked(),
        bench_softmax(),
        bench_layernorm(),
        bench_transpose(),
        bench_reshape(),
    };

    print_markdown(results);
    return static_cast<int>(g_sink) == 123456789 ? 1 : 0;
}
