/**
 * @file hello.cu
 * @brief 极简 CUDA Hello World — 验证 nvcc + GPU 环境可用
 *
 * Usage:
 *   nvcc hello.cu -o hello && ./hello
 *
 * 期望输出：
 *   Hello from CPU!
 *   Hello from GPU thread 0!
 *   Hello from GPU thread 1!
 *   Hello from GPU thread 2!
 *   ...
 *   Hello from GPU thread 7!
 *   Done!
 */

#include <cuda_runtime.h>
#include <iostream>

int main() {
    // 1. CPU 问候
    std::cout << "Hello from CPU!" << std::endl;

    // 2. 查 GPU 信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name
              << " (CC " << prop.major << "." << prop.minor << ")" << std::endl;

    // 3. GPU 问候 — 打印每个线程的 ID
    //    用 printf 直接从 kernel 输出到终端
    int n_threads = 8;
    int n_blocks = 1;

    // 注意：kernel 里用 printf 需要启用 cudaDeviceSynchronize
    // 而且要 #include <stdio.h>    ← kernel 里用 C 的头文件
    std::cout << "GPU says:" << std::endl;

    // 启动 kernel
    hello<<<n_blocks, n_threads>>>();

    // 必须同步，否则 kernel 里的 printf 可能来不及输出程序就退出了
    cudaDeviceSynchronize();

    std::cout << "Done!" << std::endl;
    return 0;
}

/// @brief GPU kernel：每个线程打印自己的 ID
__global__ void hello() {
    printf("  Hello from GPU thread %d!\n", threadIdx.x);
}