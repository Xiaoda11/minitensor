cmake_minimum_required(VERSION 3.10)
project(MiniTensor)

# 设置 C++ 标准 (复习 C++17 特性)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 编译选项：Debug 模式方便使用 GDB 调试内存
set(CMAKE_BUILD_TYPE Debug)

add_executable(minitensor main.cpp tensor.cpp)
