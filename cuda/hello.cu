/**
 * @file hello.cu
 * @brief 极简 CUDA Hello World — 验证 nvcc + GPU 环境可用
 *
 * Usage:
 *   nvcc -ccbin g++-12 hello.cu -o hello && ./hello
 *
 * 期望输出：
 *   Hello from CPU!
 *   GPU: NVIDIA GeForce RTX 2060 (CC 7.5)
 *   GPU says:
 *     Hello from GPU thread 0!
 *     ...
 *     Hello from GPU thread 7!
 *   Done!
 */

#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel 必须在调用前声明/定义
__global__ void hello() {
    printf("  Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name
              << " (CC " << prop.major << "." << prop.minor << ")" << std::endl;

    int n_threads = 8;
    int n_blocks = 1;

    std::cout << "GPU says:" << std::endl;
    hello<<<n_blocks, n_threads>>>();

    cudaDeviceSynchronize();

    std::cout << "Done!" << std::endl;
    return 0;
}