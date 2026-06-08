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
add_executable(minitensor main.cpp)

# ---- Optional: benchmark ----
# add_executable(benchmark benchmark.cpp)