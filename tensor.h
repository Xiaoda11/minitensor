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

 // =================  形状转换 =================
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;

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

// --- 3. Broadcasting 工具函数 ---

/// @brief 判断两个 shape 是否可以广播
/// 规则：从后往前逐维比较，满足以下之一即兼容：
///   (1) 两维相等
///   (2) 其中一维为 1
///   (3) 其中一维不存在（短的那个视为 1）
inline bool can_broadcast(const std::vector<int>& a, const std::vector<int>& b) {
    int na = a.size();
    int nb = b.size();
    int n = std::max(na, nb);
    for (int i = 0; i < n; ++i) {
        int dim_a = (i < na) ? a[na - 1 - i] : 1;  // 从后往前取，缺失视为 1
        int dim_b = (i < nb) ? b[nb - 1 - i] : 1;
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true; 
}

/// @brief 计算广播后的结果 shape
inline std::vector<int> broadcast_shape(const std::vector<int>& a, const std::vector<int>& b) {
    int na = a.size();
    int nb = b.size();
    int n = std::max(na, nb);
    std::vector<int> result(n);
    for (int i = 0; i < n; ++i) {
        int dim_a = (i < na) ? a[na - 1 - i] : 1;
        int dim_b = (i < nb) ? b[nb - 1 - i] : 1;
        result[n - 1 - i] = std::max(dim_a, dim_b);
    }
    return result;
}

// --- 4. 运算符 ---

template <typename T>
void Tensor<T>::fill(T value) {
    if (!data_ptr_) return;
    for (int i = 0; i < numel_; ++i) data_ptr_[i] = value;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    // 1. 检查是否可以广播
    if (!can_broadcast(shape_, other.shape_)) {
        throw std::invalid_argument("加法：形状无法广播");
    }
    
    // 2. 计算广播后的结果 shape
    auto out_shape = broadcast_shape(shape_, other.shape_);
    Tensor result(out_shape);
    
    // 3. 判断是否需要走广播路径
    bool need_broadcast = (shape_ != other.shape_);
    
    if (need_broadcast) {
        // 广播路径：用 ND 索引映射
        int ndim = out_shape.size();
        
        struct DimInfo {
            int out_dim;
            int a_stride;
            int b_stride;
        };
        std::vector<DimInfo> dims(ndim);
        for (int d = 0; d < ndim; ++d) {
            dims[d].out_dim = out_shape[d];
            int a_ndim = shape_.size();
            int a_idx = d - (ndim - a_ndim);
            dims[d].a_stride = (a_idx >= 0 && shape_[a_idx] != 1) ? stride_[a_idx] : 0;
            
            int b_ndim = other.shape_.size();
            int b_idx = d - (ndim - b_ndim);
            dims[d].b_stride = (b_idx >= 0 && other.shape_[b_idx] != 1) ? other.stride_[b_idx] : 0;
        }
        
        for (int i = 0; i < result.numel_; ++i) {
            // 线性索引 → 多维坐标：从后往前取模
            int remainder = i;
            int a_offset = 0;
            int b_offset = 0;
            for (int d = ndim - 1; d >= 0; --d) {
                int coord = remainder % dims[d].out_dim;
                remainder /= dims[d].out_dim;
                a_offset += coord * dims[d].a_stride;
                b_offset += coord * dims[d].b_stride;
            }
            result.data_ptr_[i] = data_ptr_[a_offset] + other.data_ptr_[b_offset];
        }
    } else {
        // 快速路径：shape 完全相同
        for (int i = 0; i < numel_; ++i) {
            result.data_ptr_[i] = data_ptr_[i] + other.data_ptr_[i];
        }
    }
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
    // 1. 检查是否可以广播
    if (!can_broadcast(shape_, other.shape_)) {
        throw std::invalid_argument("乘法：形状无法广播");
    }
    
    // 2. 计算广播后的结果 shape
    auto out_shape = broadcast_shape(shape_, other.shape_);
    Tensor result(out_shape);
    
    // 3. 判断是否需要走广播路径
    bool need_broadcast = (shape_ != other.shape_);
    
    if (need_broadcast) {
        // 广播路径：用 ND 索引映射
        int ndim = out_shape.size();
        
        // 预计算每个维度的信息，避免内层循环重复调用
        struct DimInfo {
            int out_dim;    // 输出维度大小
            int a_stride;   // a 在该维的 stride（如果该维为 1 则 stride 视为 0）
            int b_stride;   // b 在该维的 stride（如果该维为 1 则 stride 视为 0）
        };
        std::vector<DimInfo> dims(ndim);
        for (int d = 0; d < ndim; ++d) {
            dims[d].out_dim = out_shape[d];
            // 如果 a 在该维度存在且不为 1，用真实 stride；否则 stride=0（索引不变）
            int a_ndim = shape_.size();
            int a_idx = d - (ndim - a_ndim);  // a 的维度对齐到输出维度
            dims[d].a_stride = (a_idx >= 0 && shape_[a_idx] != 1) ? stride_[a_idx] : 0;
            
            int b_ndim = other.shape_.size();
            int b_idx = d - (ndim - b_ndim);
            dims[d].b_stride = (b_idx >= 0 && other.shape_[b_idx] != 1) ? other.stride_[b_idx] : 0;
        }
        
        // 主循环：线性遍历输出，ND 索引映射
        for (int i = 0; i < result.numel_; ++i) {
            // 线性索引 → 多维坐标：从后往前取模
            int remainder = i;
            int a_offset = 0;
            int b_offset = 0;
            for (int d = ndim - 1; d >= 0; --d) {
                int coord = remainder % dims[d].out_dim;
                remainder /= dims[d].out_dim;
                a_offset += coord * dims[d].a_stride;
                b_offset += coord * dims[d].b_stride;
            }
            result.data_ptr_[i] = data_ptr_[a_offset] * other.data_ptr_[b_offset];
        }
    } else {
        // 快速路径：shape 完全相同，直接逐元素乘
        for (int i = 0; i < numel_; ++i) {
            result.data_ptr_[i] = data_ptr_[i] * other.data_ptr_[i];
        }
    }
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
template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int>& new_shape) const {
    // 1. 验证元素总数不变
    int new_numel = 1;
    for (int dim : new_shape) {
        if (dim <= 0) throw std::invalid_argument("reshape: 维度大小必须为正数");
        new_numel *= dim;
    }
    if (new_numel != numel_) {
        throw std::invalid_argument(
            "reshape: 元素总数不匹配 — 当前 " + std::to_string(numel_) +
            ", 目标 " + std::to_string(new_numel));
    }

    // 2. 创建新张量（按 new_shape 分配内存和 stride）
    Tensor<T> result(new_shape);

    // 3. 拷贝数据 — reshape 不改变数据顺序，只是 reinterpret
    //    因为我们始终维护 contiguous 内存，所以直接逐元素拷贝即可
    for (int i = 0; i < numel_; ++i) {
        result.data_ptr_[i] = data_ptr_[i];
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::transpose(int dim0, int dim1) const {
    int ndim = shape_.size();
    // 1. 支持负索引: -1 表示最后一个维度
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    // 2. 边界检查
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::invalid_argument("transpose: 维度超出范围");
    }

    // 3. 构建新的 shape 和 stride — 交换 dim0 和 dim1
    std::vector<int> new_shape = shape_;
    std::vector<int> new_stride = stride_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_stride[dim0], new_stride[dim1]);

    // 4. 创建结果张量 — 使用 new_stride 构造
    //    但 Tensor 构造函数会自己算 stride，所以我们需要一个不同的方式:
    //    创建一个 contiguous 的新张量，然后通过 ND 索引映射填充转置后的数据
    Tensor<T> result(new_shape);

    // 5. ND 索引映射: 遍历原张量的每个线性索引
    //    找到其多维坐标，交换 dim0/dim1 坐标后，计算在新张量中的位置
    for (int i = 0; i < numel_; ++i) {
        // 线性索引 → 多维坐标
        int remainder = i;
        std::vector<int> coords(ndim);
        for (int d = ndim - 1; d >= 0; --d) {
            coords[d] = remainder % shape_[d];
            remainder /= shape_[d];
        }

        // 交换 dim0 和 dim1 的坐标
        std::swap(coords[dim0], coords[dim1]);

        // 多维坐标 → 新张量的线性索引
        int new_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            new_idx += coords[d] * result.stride()[d];
        }

        result.data_ptr_[new_idx] = data_ptr_[i];
    }

    return result;
}

// =============================================================================
// v0.3 — Computation Graph (Day 14)
// =============================================================================
// 计算图：将算子组织成 DAG，按拓扑序依次执行
// 映射到 llama.cpp 的 ggml_build_forward + ggml_graph_compute

/// @brief 计算图节点的操作类型
enum class GraphOp {
    Input,      // 叶子节点：用户输入的张量
    Matmul,     // 矩阵乘法
    Add,        // 逐元素加法
    Mul,        // 逐元素乘法
    Transpose,  // 转置
    Softmax,    // softmax
};

inline const char* graph_op_name(GraphOp op) {
    switch (op) {
        case GraphOp::Input:    return "Input";
        case GraphOp::Matmul:   return "Matmul";
        case GraphOp::Add:      return "Add";
        case GraphOp::Mul:      return "Mul";
        case GraphOp::Transpose: return "Transpose";
        case GraphOp::Softmax:  return "Softmax";
    }
    return "Unknown";
}

/// @brief 计算图节点
struct GraphNode {
    int id;                     ///< 节点 ID (在图中的索引)
    GraphOp op;                 ///< 操作类型
    std::vector<int> input_ids; ///< 前置节点 ID 列表
    std::vector<int> out_shape; ///< 输出张量的 shape
    std::vector<float> output;  ///< 计算结果（执行后填充）
    bool computed = false;      ///< 是否已计算

    GraphNode(int id, GraphOp op) : id(id), op(op) {}
};

/// @brief 轻量级张量引用 — 前向声明
class Graph;

/// @brief 轻量级张量引用 — 指向计算图中的某个节点
/// 不持有数据，只持有图的弱引用和节点 ID
/// 通过它可以链式调用算子: g.matmul(a, b).softmax()
class TensorRef {
public:
    TensorRef() : graph_(nullptr), node_id_(-1) {}

    /// @brief 打印这个节点的形状
    void print_info(const std::string& name = "TensorRef") const;

    const std::vector<int>& shape() const;
    const std::vector<float>& data() const;

    int node_id() const { return node_id_; }

private:
    friend class Graph;
    Graph* graph_;
    int node_id_;

    TensorRef(Graph* g, int nid) : graph_(g), node_id_(nid) {}
};

/// @brief 前向计算图 (DAG)
/// 用法：
///   Graph g;
///   auto a = g.tensor({2, 3});      // 创建叶子节点
///   auto b = g.tensor({3, 4});
///   auto c = g.matmul(a, b);        // 构建计算关系
///   auto d = g.softmax(c);
///   g.compute();                     // 按拓扑序执行整个图
///   d.print_info();                  // 查看结果
class Graph {
public:
    Graph() = default;

    /// @brief 创建叶子节点 (输入张量)
    TensorRef tensor(std::vector<int> shape) {
        int id = static_cast<int>(nodes_.size());
        nodes_.emplace_back(id, GraphOp::Input);
        nodes_.back().out_shape = std::move(shape);
        return TensorRef(this, id);
    }

    /// @brief 设置叶子节点的数据
    void set_data(TensorRef ref, const std::vector<float>& data) {
        auto& node = nodes_[ref.node_id_];
        node.output = data;
        node.computed = true;
    }

    // ========== 算子构建 ==========

    TensorRef matmul(TensorRef a, TensorRef b) {
        if (a.graph_ != this || b.graph_ != this)
            throw std::invalid_argument("matmul: TensorRef 不属于同一个 Graph");
        int M = node(a).out_shape[0];
        int K = node(a).out_shape[1];
        int N = node(b).out_shape[1];
        if (node(a).out_shape[1] != node(b).out_shape[0])
            throw std::invalid_argument("matmul: 维度不匹配");
        int id = add_node(GraphOp::Matmul, {a.node_id_, b.node_id_}, {M, N});
        return TensorRef(this, id);
    }

    TensorRef add(TensorRef a, TensorRef b) {
        if (a.graph_ != this || b.graph_ != this)
            throw std::invalid_argument("add: TensorRef 不属于同一个 Graph");
        if (!can_broadcast(node(a).out_shape, node(b).out_shape))
            throw std::invalid_argument("add: 形状无法广播");
        auto out = broadcast_shape(node(a).out_shape, node(b).out_shape);
        int id = add_node(GraphOp::Add, {a.node_id_, b.node_id_}, out);
        return TensorRef(this, id);
    }

    TensorRef mul(TensorRef a, TensorRef b) {
        if (a.graph_ != this || b.graph_ != this)
            throw std::invalid_argument("mul: TensorRef 不属于同一个 Graph");
        if (!can_broadcast(node(a).out_shape, node(b).out_shape))
            throw std::invalid_argument("mul: 形状无法广播");
        auto out = broadcast_shape(node(a).out_shape, node(b).out_shape);
        int id = add_node(GraphOp::Mul, {a.node_id_, b.node_id_}, out);
        return TensorRef(this, id);
    }

    TensorRef transpose(TensorRef a, int dim0, int dim1) {
        if (a.graph_ != this)
            throw std::invalid_argument("transpose: TensorRef 不属于同一个 Graph");
        auto& shape = node(a).out_shape;
        int ndim = static_cast<int>(shape.size());
        if (dim0 < 0) dim0 += ndim;
        if (dim1 < 0) dim1 += ndim;
        if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
            throw std::invalid_argument("transpose: 维度超出范围");
        auto new_shape = shape;
        std::swap(new_shape[dim0], new_shape[dim1]);
        int id = add_node(GraphOp::Transpose, {a.node_id_}, new_shape);
        // 存储转置的维度信息
        nodes_[id].input_ids.push_back(dim0);
        nodes_[id].input_ids.push_back(dim1);
        return TensorRef(this, id);
    }

    TensorRef softmax(TensorRef a) {
        if (a.graph_ != this)
            throw std::invalid_argument("softmax: TensorRef 不属于同一个 Graph");
        int id = add_node(GraphOp::Softmax, {a.node_id_}, node(a).out_shape);
        return TensorRef(this, id);
    }

    // ========== 执行引擎 ==========

    /// @brief 按拓扑序执行整个图
    void compute() {
        auto order = topo_sort();
        for (int nid : order) {
            auto& n = nodes_[nid];
            if (n.op == GraphOp::Input) continue; // 叶子节点已有数据
            execute_node(n);
            n.computed = true;
        }
    }

    /// @brief 拓扑排序 (Kahn 算法)
    /// 返回节点 ID 列表，保证每个节点的输入都在它之前
    std::vector<int> topo_sort() const {
        int n = static_cast<int>(nodes_.size());
        std::vector<int> in_degree(n, 0);
        std::vector<std::vector<int>> adj(n);

        // 构建邻接表和入度
        for (int i = 0; i < n; ++i) {
            for (int dep : actual_deps(nodes_[i])) {
                adj[dep].push_back(i);
                in_degree[i]++;
            }
        }

        std::vector<int> order;
        std::vector<int> queue;
        for (int i = 0; i < n; ++i) {
            if (in_degree[i] == 0) queue.push_back(i);
        }

        while (!queue.empty()) {
            int u = queue.back();
            queue.pop_back();
            order.push_back(u);
            for (int v : adj[u]) {
                in_degree[v]--;
                if (in_degree[v] == 0) queue.push_back(v);
            }
        }

        if (static_cast<int>(order.size()) != n)
            throw std::runtime_error("计算图包含环路，无法执行拓扑排序");
        return order;
    }

    int num_nodes() const { return static_cast<int>(nodes_.size()); }

    /// @brief 打印图结构 (调试用)
    void print_graph() const {
        std::cout << "=== Computation Graph (" << nodes_.size() << " nodes) ===" << std::endl;
        for (auto& n : nodes_) {
            std::cout << "  [" << n.id << "] " << graph_op_name(n.op)
                      << " -> shape " << shape_str(n.out_shape)
                      << " | deps: [";
            for (size_t i = 0; i < actual_deps(n).size(); ++i) {
                std::cout << actual_deps(n)[i] << (i + 1 < actual_deps(n).size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "==========================================" << std::endl;
    }

    const GraphNode& get_node(int id) const { return nodes_[id]; }

private:
    std::vector<GraphNode> nodes_;

    int add_node(GraphOp op, const std::vector<int>& deps, std::vector<int> out_shape) {
        int id = static_cast<int>(nodes_.size());
        nodes_.emplace_back(id, op);
        nodes_.back().input_ids = deps;
        nodes_.back().out_shape = std::move(out_shape);
        return id;
    }

    const GraphNode& node(TensorRef ref) const { return nodes_[ref.node_id_]; }
    GraphNode& node(TensorRef ref) { return nodes_[ref.node_id_]; }

    /// @brief 获取节点的实际依赖（排除转置节点的维度参数）
    static std::vector<int> actual_deps(const GraphNode& n) {
        if (n.op == GraphOp::Transpose) {
            return {n.input_ids[0]};  // 后两个元素是 dim0, dim1，不是依赖
        }
        return n.input_ids;
    }

    /// @brief 执行单个节点的计算
    void execute_node(GraphNode& n) {
        // 收集输入数据
        auto deps = actual_deps(n);
        if (deps.empty()) return;

        // 获取第一个输入的数据
        auto& input0 = nodes_[deps[0]];
        if (!input0.computed) throw std::runtime_error("依赖节点尚未计算");

        const float* in0 = input0.output.data();
        int numel0 = static_cast<int>(input0.output.size());

        switch (n.op) {
        case GraphOp::Matmul: {
            auto& input1 = nodes_[deps[1]];
            const float* in1 = input1.output.data();
            int M = n.out_shape[0];
            int K = input0.out_shape[1];
            int N = n.out_shape[1];
            int n_out = M * N;
            n.output.resize(n_out);
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0;
                    for (int k = 0; k < K; ++k) {
                        sum += in0[i * K + k] * in1[k * N + j];
                    }
                    n.output[i * N + j] = sum;
                }
            }
            break;
        }
        case GraphOp::Add: {
            auto& input1 = nodes_[deps[1]];
            const float* in1 = input1.output.data();
            int out_n = 1;
            for (int d : n.out_shape) out_n *= d;
            n.output.resize(out_n);
            // 复用 Tensor 的广播加法逻辑
            Tensor<float> ta(input0.out_shape);
            std::memcpy(ta.data(), in0, numel0 * sizeof(float));
            Tensor<float> tb(input1.out_shape);
            std::memcpy(tb.data(), in1, input1.output.size() * sizeof(float));
            auto tc = ta + tb;
            n.output.resize(tc.size());
            std::memcpy(n.output.data(), tc.data(), tc.size() * sizeof(float));
            break;
        }
        case GraphOp::Mul: {
            auto& input1 = nodes_[deps[1]];
            const float* in1 = input1.output.data();
            int out_n = 1;
            for (int d : n.out_shape) out_n *= d;
            n.output.resize(out_n);
            Tensor<float> ta(input0.out_shape);
            std::memcpy(ta.data(), in0, numel0 * sizeof(float));
            Tensor<float> tb(input1.out_shape);
            std::memcpy(tb.data(), in1, input1.output.size() * sizeof(float));
            auto tc = ta * tb;
            n.output.resize(tc.size());
            std::memcpy(n.output.data(), tc.data(), tc.size() * sizeof(float));
            break;
        }
        case GraphOp::Transpose: {
            int ndim = static_cast<int>(input0.out_shape.size());
            int dim0 = n.input_ids[1];
            int dim1 = n.input_ids[2];
            int out_n = 1;
            for (int d : n.out_shape) out_n *= d;
            n.output.resize(out_n);
            // ND 索引映射转置
            std::vector<int> coords(ndim);
            int remainder;
            for (int i = 0; i < out_n; ++i) {
                remainder = i;
                for (int d = ndim - 1; d >= 0; --d) {
                    coords[d] = remainder % n.out_shape[d];
                    remainder /= n.out_shape[d];
                }
                std::swap(coords[dim0], coords[dim1]);
                int src_idx = 0;
                for (int d = 0; d < ndim; ++d) {
                    // 计算源张量的线性索引
                    int src_stride = 1;
                    for (int dd = ndim - 1; dd > d; --dd) src_stride *= input0.out_shape[dd];
                    src_idx += coords[d] * src_stride;
                }
                n.output[i] = in0[src_idx];
            }
            break;
        }
        case GraphOp::Softmax: {
            int out_n = 1;
            for (int d : n.out_shape) out_n *= d;
            n.output.resize(out_n);
            Tensor<float> ta(input0.out_shape);
            std::memcpy(ta.data(), in0, numel0 * sizeof(float));
            auto tb = ::softmax(ta);
            n.output.resize(tb.size());
            std::memcpy(n.output.data(), tb.data(), tb.size() * sizeof(float));
            break;
        }
        default:
            throw std::runtime_error(std::string("未实现的算子: ") + graph_op_name(n.op));
        }
    }

    static std::string shape_str(const std::vector<int>& shape) {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i + 1 < shape.size()) s += ", ";
        }
        s += "]";
        return s;
    }
};

// =============================================================================
// TensorRef 方法实现 — 必须在 Graph 完整定义之后
// =============================================================================

inline void TensorRef::print_info(const std::string& name) const {
    if (!graph_ || node_id_ < 0) {
        std::cout << name << ": <invalid>" << std::endl;
        return;
    }
    const auto& n = graph_->get_node(node_id_);
    std::cout << name << " Shape(" << n.out_shape.size() << "): [";
    for (size_t i = 0; i < n.out_shape.size(); ++i) {
        std::cout << n.out_shape[i] << (i + 1 < n.out_shape.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;
    if (n.computed) {
        std::cout << "  data: [";
        for (size_t i = 0; i < n.output.size(); ++i) {
            std::cout << n.output[i];
            if (i + 1 < n.output.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

inline const std::vector<int>& TensorRef::shape() const {
    if (!graph_ || node_id_ < 0) throw std::runtime_error("TensorRef: invalid");
    return graph_->get_node(node_id_).out_shape;
}

inline const std::vector<float>& TensorRef::data() const {
    if (!graph_ || node_id_ < 0) throw std::runtime_error("TensorRef: invalid");
    return graph_->get_node(node_id_).output;
}

#endif // TENSOR_H
