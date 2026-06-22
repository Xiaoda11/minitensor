/**
 * @file vector_add.cu
 * @brief CUDA Week 1 — Vector Addition (GPU "Hello World")
 *
 * Learning objectives:
 *  - Grid / Block / Thread hierarchy
 *  - Global memory allocation (cudaMalloc / cudaFree)
 *  - Host ↔ Device data transfer (cudaMemcpy)
 *  - Kernel launch syntax <<<grid, block>>>
 *  - Thread indexing: blockIdx, blockDim, threadIdx
 *  - CUDA error checking macros
 *  - CUDA event timing (cudaEvent)
 *
 * Usage:
 *   nvcc -O2 vector_add.cu -o vector_add && ./vector_add
 *
 * Reference:
 *   NVIDIA CUDA C++ Programming Guide §2.1–§2.3
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

// ============================================================================
// 1. Error checking macros
// ============================================================================

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " — " << cudaGetErrorString(err_) << std::endl;       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// 2. CPU reference implementation
// ============================================================================

/// @brief Standard CPU vector addition: C[i] = A[i] + B[i]
void vector_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// 3. GPU Kernel — each thread computes ONE element
// ============================================================================

/**
 * @brief GPU kernel: 每个线程计算一个元素的加法
 *
 * Thread indexing formula:
 *   global_id = blockIdx.x * blockDim.x + threadIdx.x
 *
 * Grid layout example (n = 1,000,000, threads_per_block = 256):
 *   blocks = ceil(n / 256) = 1,000,000 / 256 = 3907 blocks
 *
 *   Block 0:     Threads 0..255  → global IDs 0..255
 *   Block 1:     Threads 0..255  → global IDs 256..511
 *   Block 3906:  Threads 0..255  → global IDs 999936..1000191
 *                (threads beyond n are guarded by the `if (tid < n)` check)
 */
__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止越界：最后一个 block 可能有线程超出 n
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// ============================================================================
// 4. Optimized kernel variant — multiple elements per thread (grid-stride loop)
// ============================================================================

/**
 * @brief Grid-stride loop pattern: 每个线程处理多个元素
 *
 * Motivation: 当 n >> grid_size 时, 每个线程处理 n/grid_size 个元素,
 * 减少 kernel launch overhead 和 block 调度开销.
 *
 * This is the idiomatic CUDA pattern for element-wise kernels.
 */
__global__ void vector_add_kernel_grid_stride(const float *a, const float *b,
                                               float *c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// 5. Test harness
// ============================================================================

#ifndef KERNEL_EXPORT
int main(int argc, char **argv) {
    // --- Parse arguments ---
    int n = (argc > 1) ? std::atoi(argv[1]) : 1 << 24;       // default: 16M elements
    int threads_per_block = (argc > 2) ? std::atoi(argv[2]) : 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    size_t bytes = n * sizeof(float);

    std::cout << "========================================" << std::endl;
    std::cout << "CUDA Week 1: Vector Addition" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Elements:          " << n << " (" << (bytes >> 20) << " MB)" << std::endl;
    std::cout << "Threads per block: " << threads_per_block << std::endl;
    std::cout << "Blocks:            " << num_blocks << std::endl;
    std::cout << "Grid-stride loops: " << (n + threads_per_block * num_blocks - 1)
              / (threads_per_block * num_blocks) << " iterations/thread\n" << std::endl;

    // --- Device info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Max Threads/Block:  " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Grid Dim:       (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "  SMs:                " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Shared Mem:     " << (prop.sharedMemPerBlock >> 10) << " KB/block\n" << std::endl;

    // --- Allocate host memory ---
    std::vector<float> h_a(n), h_b(n), h_c_cpu(n), h_c_gpu(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i) * 0.01f;
        h_b[i] = static_cast<float>(i) * 0.02f;
    }

    // --- Allocate device memory ---
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // --- Copy input from host to device ---
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // --- CPU baseline (warm-up) ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vector_add_cpu(h_a.data(), h_b.data(), h_c_cpu.data(), n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --- GPU: Basic kernel (1 element per thread) ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_basic_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_basic_ms, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // --- GPU: Grid-stride kernel ---
    CUDA_CHECK(cudaEventRecord(start));
    vector_add_kernel_grid_stride<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_stride_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_stride_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // --- Verify correctness ---
    bool pass = true;
    for (int i = 0; i < n && pass; ++i) {
        if (std::abs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_c_cpu[i]
                      << ", GPU=" << h_c_gpu[i] << std::endl;
            pass = false;
        }
    }

    // --- Report ---
    std::cout << "========== Results ==========" << std::endl;
    std::cout << "CPU time:           " << cpu_ms << " ms" << std::endl;
    std::cout << "GPU (basic kernel): " << gpu_basic_ms << " ms"
              << "  (" << (cpu_ms / gpu_basic_ms) << "x speedup)" << std::endl;
    std::cout << "GPU (grid-stride):  " << gpu_stride_ms << " ms"
              << "  (" << (cpu_ms / gpu_stride_ms) << "x speedup)" << std::endl;
    std::cout << "Verification:       " << (pass ? "PASSED ✓" : "FAILED ✗") << std::endl;

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return pass ? 0 : 1;
}
#endif // KERNEL_EXPORT