/**
 * @file hello.cu
 * @brief 极简 CUDA Hello World — 验证 nvcc + GPU 环境可用
 *
 * Usage:
 *   nvcc -Wno-deprecated-gpu-targets hello.cu -o hello && ./hello
 */

#include <cuda_runtime.h>
#include <cstdio>

__global__ void hello() {
    printf("  Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);

    printf("GPU says:\n");
    hello<<<1, 8>>>();
    cudaDeviceSynchronize();
    printf("Done!\n");
    return 0;
}