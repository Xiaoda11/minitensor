cmake_minimum_required(VERSION 3.16)
project(minitensor_cpu LANGUAGES CXX)

# ============================================
# MiniTensor CPU — Header-only tensor library
# ============================================
#
# Build:
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#
# Run tests:
#   ./minitensor

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Test executable (all 47 tests, Days 1-15) ----
add_library(libtensor INTERFACE)
target_include_directories(libtensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(minitensor main.cpp)
target_link_libraries(minitensor PRIVATE libtensor)

# ---- Optional: benchmark ----
add_executable(minitensor_cpu_benchmark benchmark.cpp)
target_link_libraries(minitensor_cpu_benchmark PRIVATE libtensor)
target_compile_options(minitensor_cpu_benchmark PRIVATE -O3)
