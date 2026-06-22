# Define KERNEL_EXPORT so kernel source files skip their main()
add_compile_definitions(KERNEL_EXPORT)

add_executable(cuda_kernel_benchmark
    benchmark.cu
    ${CMAKE_SOURCE_DIR}/../../../cuda/vector_add.cu
    ${CMAKE_SOURCE_DIR}/../../../cuda/matmul_naive.cu
    ${CMAKE_SOURCE_DIR}/../../../cuda/matmul_tiled.cu
    ${CMAKE_SOURCE_DIR}/../../../cuda/softmax.cu
    ${CMAKE_SOURCE_DIR}/../../../cuda/layernorm.cu
    ${CMAKE_SOURCE_DIR}/../../../cuda/attention.cu
)

target_include_directories(cuda_kernel_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_options(cuda_kernel_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
