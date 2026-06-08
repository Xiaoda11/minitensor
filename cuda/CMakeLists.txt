cmake_minimum_required(VERSION 3.20)
project(minitensor_cuda LANGUAGES CXX CUDA)

# ============================================
# CUDA Phase 1: Kernel Rewrite (minitensor v0.4)
# ============================================
#
# Build:
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
#   make -j$(nproc)
#
# Or compile a single file directly:
#   nvcc -O2 vector_add.cu -o vector_add

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Detect compute capability if not set ----
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures (e.g. 75 for Turing, 80 for Ampere, 86 for Ampere GA10x, 89 for Ada)")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, defaulting to 75 (Turing)")
endif()

# ---- Compiler flags ----
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# ---- Week 1: Vector Addition ----
add_executable(vector_add vector_add.cu)
target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# ---- Week 2: Simple Matmul ----
# add_executable(matmul_naive matmul_naive.cu)
# add_executable(matmul_tiled matmul_tiled.cu)

# ---- Week 3: Softmax / LayerNorm ----
# add_executable(softmax_cuda softmax.cu)
# add_executable(layernorm_cuda layernorm.cu)

# ---- Week 4: Performance analysis ----
# add_executable(matmul_bench matmul_bench.cu)