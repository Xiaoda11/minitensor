#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
/**
 * @brief MiniTensor: 一个简易的 N 维张量类
 * 主要用于学习：内存管理 (Rule of 5), 广播机制基础, 以及基本算子实现。
 */
template <typename T>
class Tensor {
public:
    // ================= 1. 生命周期管理 (Rule of 5) =================

    /// @brief 构造函数：分配内存
    explicit Tensor(std::vector<int> shape);

    /// @brief 便捷构造函数：针对 1D 张量
    explicit Tensor(int size);

    ~Tensor() = default; // unique_ptr 自动管理内存释放

    Tensor(const Tensor& other);            // 拷贝构造
    Tensor(Tensor&& other) noexcept;        // 移动构造

    Tensor& operator=(const Tensor& other); // 拷贝赋值
    Tensor& operator=(Tensor&& other) noexcept; // 移动赋值

    // ================= 2. 数据访问 =================

    T* data();
    const T* data() const;
    int size() const { return numel_; }
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& stride() const { return stride_; }

    // ================= 3. 逐元素运算 =================

    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // Hadamard product (逐元素乘)
    Tensor& operator+=(const Tensor& other);

    /// @brief 填充标量值
    void fill(T value);

    /// @brief 打印张量信息 (调试用)
    void print_info(const std::string& name = "Tensor") const;

    // ================= 4. 友元声明 (外部算子) =================

    template <typename U>
    friend Tensor<U> matmul(const Tensor<U>& a, const Tensor<U>& b);

    template <typename U>
    friend Tensor<U> softmax(const Tensor<U>& input);

    template <typename U>
    friend Tensor<U> layernorm(const Tensor<U>& input,
                               const Tensor<U>& weight,
                               const Tensor<U>& bias,
                               U eps);
    template <typename U>
    friend Tensor<U> matmul_blocked(const Tensor<U>& a, const Tensor<U>& b, int tile_size);
private:
    std::unique_ptr<T[]> data_ptr_; ///< 原始数据指针 (模拟显存/内存)
    std::vector<int> shape_;        ///< 维度信息 (如 {2, 3, 4})
    std::vector<int> stride_;       ///< 行优先 (Row-Major) 步长
    int numel_;                     ///< 元素总数
};

// =============================================================================
// 实现部分 (Header-only 模板类)
// =============================================================================

// --- 1. 生命周期 ---

template <typename T>
Tensor<T>::Tensor(int size) : Tensor(std::vector<int>{size}) {}

template <typename T>
Tensor<T>::Tensor(std::vector<int> shape)
    : shape_(std::move(shape)), numel_(1), data_ptr_(nullptr) {
    
    if (shape_.empty()) throw std::invalid_argument("Shape 不能为空");

    // 1. 计算元素总数
    for (int dim : shape_) {
        if (dim <= 0) throw std::invalid_argument("维度大小必须为正数");
        numel_ *= dim;
    }

    // 2. 计算步长 (Row-Major 布局)
    // 逻辑: Stride[i] = product(shape[i+1:])
    // 例: Shape [2, 3, 4] -> Strides [12, 4, 1]
    stride_.resize(shape_.size());
    int stride_val = numel_;
    for (size_t i = 0; i < shape_.size(); ++i) {
        stride_val /= shape_[i];
        stride_[i] = stride_val;
    }

    // 3. 分配内存
    data_ptr_ = std::make_unique<T[]>(numel_);
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : numel_(other.numel_), data_ptr_(nullptr),
      shape_(other.shape_), stride_(other.stride_) {
    
    if (numel_ > 0) {
        data_ptr_ = std::make_unique<T[]>(numel_);
        // 深拷贝数据
        for (int i = 0; i < numel_; ++i) data_ptr_[i] = other.data_ptr_[i];
    }
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : numel_(other.numel_), data_ptr_(std::move(other.data_ptr_)),
      shape_(std::move(other.shape_)), stride_(std::move(other.stride_)) {
    other.numel_ = 0; // 资源被移走后，源对象变为空
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this == &other) return *this; // 自赋值检查

    // 重新分配并拷贝
    numel_ = other.numel_;
    shape_ = other.shape_;
    stride_ = other.stride_;
    data_ptr_ = nullptr; // 清理旧数据
    
    if (numel_ > 0) {
        data_ptr_ = std::make_unique<T[]>(numel_);
        for (int i = 0; i < numel_; ++i) data_ptr_[i] = other.data_ptr_[i];
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    numel_ = other.numel_;
    shape_ = std::move(other.shape_);
    stride_ = std::move(other.stride_);
    data_ptr_ = std::move(other.data_ptr_);

    other.numel_ = 0; // 源对象状态重置
    return *this;
}

// --- 2. 访问器 ---

template <typename T>
T* Tensor<T>::data() { return data_ptr_.get(); }

template <typename T>
const T* Tensor<T>::data() const { return data_ptr_.get(); }

// --- 3. 运算符 ---

template <typename T>
void Tensor<T>::fill(T value) {
    if (!data_ptr_) return;
    for (int i = 0; i < numel_; ++i) data_ptr_[i] = value;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    if (numel_ != other.numel_) throw std::invalid_argument("加法：形状不匹配");
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) result.data_ptr_[i] = data_ptr_[i] + other.data_ptr_[i];
    return result;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor& other) {
    if (numel_ != other.numel_) throw std::invalid_argument("+=：形状不匹配");
    for (int i = 0; i < numel_; ++i) data_ptr_[i] += other.data_ptr_[i];
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor& other) const {
    if (numel_ != other.numel_) throw std::invalid_argument("乘法：形状不匹配");
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) result.data_ptr_[i] = data_ptr_[i] * other.data_ptr_[i];
    return result;
}

template <typename T>
void Tensor<T>::print_info(const std::string& name) const {
    std::cout << name << " Shape(" << numel_ << "): [";
    if (!data_ptr_) {
        std::cout << "Empty";
    } else {
        for (int i = 0; i < numel_; ++i) {
            std::cout << data_ptr_[i];
            if (i < numel_ - 1) std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// =============================================================================
// 全局算子函数
// =============================================================================

/// @brief 矩阵乘法 (仅支持 2D)
///        计算 C = A @ B，维度要求: A[M,K] x B[K,N] -> C[M,N]
template <typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw std::invalid_argument("matmul: 输入必须是 2D 张量");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul: 内部维度必须匹配");

    int M = a.shape()[0];
    int K = a.shape()[1]; // 公共维度
    int N = b.shape()[1];

    Tensor<T> c({M, N});

    // 标准 O(N^3) 实现
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                // 行优先访问模式: row_offset + index
                sum += a.data()[i * a.stride()[0] + k] *
                       b.data()[k * b.stride()[0] + j];
            }
            c.data()[i * c.stride()[0] + j] = sum;
        }
    }
    return c;
}

/// @brief Softmax 函数 (数值稳定版本)
///        沿最后一维计算 exp(x - max(x)) / sum(exp(x - max(x)))
///        当前支持 2D 输入 [Batch, Features]
template <typename T>
Tensor<T> softmax(const Tensor<T>& input) {
    if (input.shape().size() != 2) throw std::invalid_argument("softmax: 输入必须是 2D");

    int num_rows = input.shape()[0];
    int cols = input.shape()[1];
    int row_stride = input.stride()[0];
    Tensor<T> output(input.shape());

    const T* in_data = input.data();
    T* out_data = output.data();

    for (int i = 0; i < num_rows; ++i) {
        int offset = i * row_stride; // 当前行的起始偏移量

        // 1. 寻找最大值 (防止 exp 溢出)
        T max_val = in_data[offset];
        for (int j = 1; j < cols; ++j) {
            if (in_data[offset + j] > max_val) max_val = in_data[offset + j];
        }

        // 2. 计算 exp(x - max) 并求和
        // 优化: 计算完 exp 后直接存入 output，避免重复计算
        T sum_exp = 0;
        for (int j = 0; j < cols; ++j) {
            T e = std::exp(in_data[offset + j] - max_val);
            out_data[offset + j] = e; 
            sum_exp += e;
        }

        // 3. 归一化
        T inv_sum = static_cast<T>(1.0) / sum_exp;
        for (int j = 0; j < cols; ++j) {
            out_data[offset + j] *= inv_sum;
        }
    }
    return output;
}

/// @brief Layer Normalization (仅支持 2D)
///        沿最后一维进行归一化: y = (x - mean) / sqrt(var + eps) * weight + bias
template <typename T>
Tensor<T> layernorm(const Tensor<T>& input,
                    const Tensor<T>& weight,
                    const Tensor<T>& bias,
                    T eps = static_cast<T>(1e-5f)) {
    if (input.shape().size() != 2) throw std::invalid_argument("layernorm: 输入必须是 2D");

    int num_rows = input.shape()[0];
    int last_dim = input.shape()[1];
    if (weight.size() != last_dim || bias.size() != last_dim)
        throw std::invalid_argument("layernorm: Weight/Bias 尺寸不匹配");

    int row_stride = input.stride()[0];
    Tensor<T> output(input.shape());

    const T* in_data = input.data();
    T* out_data = output.data();
    const T* w_data = weight.data();
    const T* b_data = bias.data();

    for (int i = 0; i < num_rows; ++i) {
        int offset = i * row_stride;

        // 1. 计算均值
        T sum = 0;
        for (int j = 0; j < last_dim; ++j) sum += in_data[offset + j];
        T mean = sum / static_cast<T>(last_dim);

        // 2. 计算方差
        T var_sum = 0;
        for (int j = 0; j < last_dim; ++j) {
            T diff = in_data[offset + j] - mean;
            var_sum += diff * diff;
        }
        T var = var_sum / static_cast<T>(last_dim);

        // 3. 归一化 + 仿射变换
        // 优化: 使用倒数乘法 inv_std 替代除法，速度更快
        T inv_std = static_cast<T>(1.0) / std::sqrt(var + eps);
        
        for (int j = 0; j < last_dim; ++j) {
            out_data[offset + j] = w_data[j] * (in_data[offset + j] - mean) * inv_std + b_data[j];
        }
    }
    return output;
}
/// @brief 矩阵乘法 (分块版本，缓存友好)
///        计算 C = A @ B，维度要求: A[M,K] x B[K,N] -> C[M,N]
template <typename T>
Tensor<T> matmul_blocked(const Tensor<T>& a, const Tensor<T>& b, int tile_size = 32) {
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw std::invalid_argument("matmul_blocked: 输入必须是 2D 张量");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul_blocked: 内部维度必须匹配");

    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[1];

    Tensor<T> c({M, N});
    c.fill(static_cast<T>(0));  // 必须清零，因为内层是 += 累加

    // 提取指针和步长，避免内层循环重复调用函数
    const T* a_data = a.data();
    const T* b_data = b.data();
    T* c_data = c.data();
    int a_stride = a.stride()[0];
    int b_stride = b.stride()[0];
    int c_stride = c.stride()[0];

    // 6 层循环：外层分块 + 内层 i-k-j 顺序
    for (int ii = 0; ii < M; ii += tile_size) {
        for (int jj = 0; jj < N; jj += tile_size) {
            for (int kk = 0; kk < K; kk += tile_size) {
                for (int i = ii; i < std::min(ii + tile_size, M); ++i) {
                    for (int k = kk; k < std::min(kk + tile_size, K); ++k) {
                        T a_val = a_data[i * a_stride + k];  // 提为循环不变量
                        for (int j = jj; j < std::min(jj + tile_size, N); ++j) {
                            c_data[i * c_stride + j] += a_val * b_data[k * b_stride + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}

#endif // TENSOR_H
