add_executable(cuda_kernel_benchmark benchmark.cu)

target_compile_options(cuda_kernel_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
