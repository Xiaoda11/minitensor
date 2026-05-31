#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <memory>
#include<vector>
#include <cmath>
template<typename T> 
class Tensor {
public:
    // 1. 构造函数：分配内存
    Tensor(std::vector<int> shape_);
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
    T* data();
    const T* data() const;
    int size() const;

    // 运算符重载：实现 A + B
    Tensor operator+(const Tensor& other) const;
    //逐元素向乘
    Tensor operator*(const Tensor& other) const;
    // 辅助函数：填充数据
    void fill(T value);

    // 打印张量信息 (方便调试)
    void print_info(const std::string& name) const;

    // 右值引用
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    const std::vector<int>& shape() const;
    const std::vector<int>& stride() const;

    // 声明友元函数 (类外定义)
    template<typename U>
    friend Tensor<U> matmul(const Tensor<U>& a, const Tensor<U>& b);

    template<typename U>
    friend Tensor<U> softmax(const Tensor<U>& input);

    template<typename U>
    friend Tensor<U> layernorm(const Tensor<U>& input,
                                const Tensor<U>& weight,
                                const Tensor<U>& bias,
                                U eps);

private:
    std::unique_ptr<T[]> data_ptr_; // 模拟显存/内存指针
   std::vector<int> shape_;   // 每维大小
    std::vector<int> stride_;  // 每维 stride
    int numel_;                // 元素总数 (= product of shape)
};
// 1. 构造函数：分配内存
template<typename T>
Tensor<T>::Tensor(int size) : Tensor(std::vector<int>{size}) {}
template<typename T>
Tensor<T>::Tensor(std::vector<int> shape) : numel_(1),  data_ptr_(nullptr) {
    std::cout << "[Constructor] Allocating typename Ts." << std::endl;
    shape_ = shape ;
    for (int i = 0; i < shape_.size(); i++)
    {
        if (shape_[i] <= 0)
        {
            throw std::invalid_argument("...");
        }
        numel_ *= shape_[i];
    }
    int temp = numel_;
    stride_.resize(shape.size());
    for (int j = 0; j < shape_.size(); j++)
    {
        stride_[j] = temp = temp/shape_[j];
      
    }
    
     data_ptr_  = std::make_unique<T[]>(numel_);
}

// 2. 析构函数：释放内存（智能指针自动实现）
template<typename T>
Tensor<T>::~Tensor() = default; 

// 3. 拷贝构造函数 (实现深拷贝)
template<typename T>
Tensor<T>::Tensor(const Tensor& other) : 
numel_(other.numel_), data_ptr_(nullptr),shape_(other.shape_),stride_(other.stride_) {
    std::cout << "[Copy Constructor] Deep copying " << numel_ << " Ts." << std::endl;
     data_ptr_ = std::make_unique<T[]>(numel_);
    for ( int i = 0; i < numel_; i++)
    {
        data_ptr_[i] = other.data_ptr_[i];
    }
}
// 右值引用：移动构造函数
template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : numel_(other.numel_), data_ptr_(std::move(other.data_ptr_)),
    shape_(std::move(other.shape_)),stride_(std::move(other.stride_)) {
    other.numel_ = 0;
    std::cout << "[Move Constructor] Resource stolen from temporary." << std::endl;
}

// 获取数据指针
template<typename T>
T* Tensor<T>::data() {
    return data_ptr_.get();
}
template<typename T>
const T* Tensor<T>::data() const {
    return data_ptr_.get();
}
template<typename T>
int Tensor<T>::size() const {
    return numel_;
}
template<typename T>
const std::vector<int>& Tensor<T>::shape() const { return shape_; }
template<typename T>
const std::vector<int>& Tensor<T>::stride() const { return stride_; }
// 填充数据
template<typename T>
void Tensor<T>::fill(T value) {
    if (!data_ptr_) return; 
    for (int i = 0; i < numel_; ++i) {
        data_ptr_[i] = value;
    }
}
template<typename T>
void Tensor<T>::print_info(const std::string& name) const {
    std::cout << name << " [Size: " << numel_ << "]: ";
    if (!data_ptr_) {
        std::cout << "(Empty/Null)" << std::endl;
        return;
    }
    for (int i = 0; i < numel_; ++i) {
        std::cout << data_ptr_[i] << " ";
    }
    std::cout << std::endl;
}

// 运算符重载：Tensor A + Tensor B
template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    if (this->numel_ != other.numel_) {
        throw std::invalid_argument("Tensor sizes must match for addition.");
    }
    
    // 1. 创建一个新 Tensor (结果)
    Tensor result(this->shape_);
    
    // TODO: 2. 执行加法逻辑
    for (int i = 0; i < numel_; ++i) {
            result.data_ptr_[i] = this-> data_ptr_[i] + other. data_ptr_[i];
        }
    
    return result; 
    }
// 运算符重载：Tensor A = Tensor B
template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
        // 1. 自赋值检查
        if (this == &other)
        {
            std::cout << "slef given data" << std::endl;
        return *this;
        }
        
        // 2. 释放旧内存
        // 3. 重新分配 + 拷贝（跟拷贝构造函数一样）
        this->numel_ = other.numel_;
        this-> data_ptr_= std::make_unique<T[]>(other.numel_);
         for ( int i = 0; i < numel_; i++)
    {
         data_ptr_[i] =   other. data_ptr_[i];
    }
        stride_ = other.stride_;
        shape_ = other.shape_;
    // 4. 返回 *thisi
        return *this;
    }
    // 运算符重载：Tensor A += Tensor B
template<typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor& other){
        if (numel_ != other.numel_)
        {
          throw std::invalid_argument("Tensor sizes must match for addition.");
        }
        for (int i = 0; i < numel_; i++)
        {
            data_ptr_[i]+=other.data_ptr_[i];
        }
        return *this;
    }
    // 右值引用 移动赋值运算符
template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this == &other)
    {
        return *this;/* code */
    }
    this->data_ptr_ = std::move(other.data_ptr_);
    this->numel_ = other.numel_;
    other.data_ptr_= nullptr;
    other.numel_= 0;
    stride_ = std::move(other.stride_);
    shape_ = std::move(other.shape_);
    return *this;
    }
// 运算符重载：逐元素相乘
template<typename T>
Tensor<T> Tensor<T>::operator* (const Tensor& other) const {
    if (this->numel_ != other.numel_) {
        throw std::invalid_argument("Tensor sizes must match for addition.");
    }
    
    // 1. 创建一个新 Tensor (结果)
    Tensor result(this->shape_);
    
    // TODO: 2. 执行加法逻辑
    for (int i = 0; i < numel_; ++i) {
            result.data_ptr_[i] = this-> data_ptr_[i] * other. data_ptr_[i];
        }
    
    return result; 
    }
// 矩阵乘法: A(M,K) × B(K,N) = C(M,N)  (类外定义)
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b){
    // 1. 形状检查：两个都必须是 2D
    if (a.shape_.size() != 2 || b.shape_.size() != 2) {
        throw std::invalid_argument("matmul: both tensors must be 2D");
    }
    // 2. 维度匹配检查：A 的列数 == B 的行数
    if (a.shape_[1] != b.shape_[0]) {
        throw std::invalid_argument("matmul: A.columns must equal B.rows");
    }

    int M = a.shape_[0];  // A 的行数 = C 的行数
    int K = a.shape_[1];  // A 的列数 = B 的行数 (公共维度)
    int N = b.shape_[1];  // B 的列数 = C 的列数

    // 3. 创建结果张量 C(M, N)
    Tensor<T> c({M, N});

    // 4. 三重循环计算
    for (int i = 0; i < M; i++) {          // 遍历 C 的每一行
        for (int j = 0; j < N; j++) {      // 遍历 C 的每一列
            T sum = 0;
            for (int k = 0; k < K; k++) {  // 沿公共维度求和
                // row-major 索引: [i][j] = i * stride[0] + j * stride[1]
                sum += a.data_ptr_[i * a.stride_[0] + k] *
                       b.data_ptr_[k * b.stride_[0] + j];
            }
            c.data_ptr_[i * c.stride_[0] + j] = sum;
        }
    }
    return c;
}
// softmax: 沿最后一维做 softmax (数值稳定版本)
template<typename T>
Tensor<T> softmax(const Tensor<T>& input) {
    if (input.shape().size() != 2) {
        throw std::invalid_argument("softmax: input must be 2D");
    }

    Tensor<T> output(input.shape());
    int rows = input.shape()[0];
    int cols = input.shape()[1];

    for (int i = 0; i < rows; i++) {
        // 1. 找最大值 (数值稳定性: 防止 exp 溢出)
        T max_val = input.data()[i * input.stride()[0] + 0];
        for (int j = 1; j < cols; j++) {
            T val = input.data()[i * input.stride()[0] + j];
            if (val > max_val) max_val = val;
        }

        // 2. 算 exp(x - max) 之和
        T sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += std::exp(input.data()[i * input.stride()[0] + j] - max_val);
        }

        // 3. 归一化: 每个元素 / sum
        for (int j = 0; j < cols; j++) {
            output.data()[i * output.stride()[0] + j] =
                std::exp(input.data()[i * input.stride()[0] + j] - max_val) / sum;
        }
    }

    return output;
}

// layernorm: 沿最后一维做 Layer Normalization
// input: [N, D], weight: [D], bias: [D]
// output[i][j] = weight[j] * (input[i][j] - mean_i) / sqrt(var_i + eps) + bias[j]
template<typename T>
Tensor<T> layernorm(const Tensor<T>& input,
                    const Tensor<T>& weight,
                    const Tensor<T>& bias,
                    T eps = static_cast<T>(1e-5)) {
    if (input.shape().size() != 2) {
        throw std::invalid_argument("layernorm: input must be 2D");
    }

    int last_dim = input.shape()[1];
    if (weight.size() != last_dim || bias.size() != last_dim) {
        throw std::invalid_argument("layernorm: weight/bias size must match last dim");
    }

    Tensor<T> output(input.shape());
    int rows = input.shape()[0];

    for (int i = 0; i < rows; i++) {
        // 1. 算均值
        T mean = 0;
        for (int j = 0; j < last_dim; j++) {
            mean += input.data()[i * input.stride()[0] + j];
        }
        mean /= static_cast<T>(last_dim);

        // 2. 算方差
        T var = 0;
        for (int j = 0; j < last_dim; j++) {
            T diff = input.data()[i * input.stride()[0] + j] - mean;
            var += diff * diff;
        }
        var /= static_cast<T>(last_dim);

        // 3. 归一化 + 仿射变换 (gamma * normed + beta)
        for (int j = 0; j < last_dim; j++) {
            int idx = i * input.stride()[0] + j;
            output.data()[idx] = weight.data()[j] *
                (input.data()[idx] - mean) / std::sqrt(var + eps) + bias.data()[j];
        }
    }

    return output;
}

#endif
