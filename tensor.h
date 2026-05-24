#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>

class Tensor {
public:
    // 1. 构造函数：分配内存
    Tensor(int size);
    
    // 2. 析构函数：释放内存 (核心复习点！)
    ~Tensor();

    // 3. 拷贝构造函数 (深拷贝 vs 浅拷贝 复习点)
    Tensor(const Tensor& other);

    // 4. 拷贝赋值运算符 (Rule of Three 最后一块)
    Tensor& operator=(const Tensor& other);
    //5. +=运算符
    Tensor& operator+=(const Tensor& other);
    // 接口：获取数据指针 (模拟与 CUDA 交互的入口)
    float* data();
    const float* data() const;
    int size() const;

    // 运算符重载：实现 A + B
    Tensor operator+(const Tensor& other) const;

    // 辅助函数：填充数据
    void fill(float value);

    // 打印张量信息 (方便调试)
    void print_info(const std::string& name) const;

private:
    float* data_ptr_; // 模拟显存/内存指针
    int size_;
};

#endif
