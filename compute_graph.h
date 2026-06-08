#ifndef COMPUTE_GRAPH_H
#define COMPUTE_GRAPH_H

#include "tensor.h"
#include <stdexcept>
#include <string>

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

    // ========== 执行辅助：广播逐元素运算 ==========

    /// @brief 执行广播加法或乘法，直接在 raw buffer 上运算（零拷贝）
    /// @tparam MulOp  true=乘法, false=加法
    template <bool MulOp>
    void broadcast_exec(const GraphNode& in0, const GraphNode& in1, GraphNode& out) {
        const float* a = in0.output.data();
        const float* b = in1.output.data();
        int ndim_out = static_cast<int>(out.out_shape.size());
        int ndim_a  = static_cast<int>(in0.out_shape.size());
        int ndim_b  = static_cast<int>(in1.out_shape.size());

        // 预计算 a/b 每个输出维度的 stride（广播维度 stride=0 代表复用同一元素）
        std::vector<int> stride_a(ndim_out, 0), stride_b(ndim_out, 0);
        int stride = 1;
        for (int d = ndim_a - 1; d >= 0; --d) {
            int od = d + (ndim_out - ndim_a);
            stride_a[od] = (in0.out_shape[d] != 1) ? stride : 0;
            stride *= in0.out_shape[d];
        }
        stride = 1;
        for (int d = ndim_b - 1; d >= 0; --d) {
            int od = d + (ndim_out - ndim_b);
            stride_b[od] = (in1.out_shape[d] != 1) ? stride : 0;
            stride *= in1.out_shape[d];
        }

        // 预计算输出各维 stride（用于分解线性索引）
        std::vector<int> out_stride(ndim_out);
        stride = 1;
        for (int d = ndim_out - 1; d >= 0; --d) {
            out_stride[d] = stride;
            stride *= out.out_shape[d];
        }

        int total = 1;
        for (int d : out.out_shape) total *= d;
        out.output.resize(total);

        for (int i = 0; i < total; ++i) {
            int remainder = i;
            int off_a = 0, off_b = 0;
            for (int d = 0; d < ndim_out; ++d) {
                int coord = remainder / out_stride[d];
                remainder %= out_stride[d];
                off_a += coord * stride_a[d];
                off_b += coord * stride_b[d];
            }
            if constexpr (MulOp)
                out.output[i] = a[off_a] * b[off_b];
            else
                out.output[i] = a[off_a] + b[off_b];
        }
    }

    /// @brief 快速路径：相同 shape 无广播的逐元素运算
    template <bool MulOp>
    void no_broadcast_exec(const float* a, const float* b, float* out, int n) {
        if constexpr (MulOp)
            for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
        else
            for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }

    /// @brief 执行单个节点的计算
    void execute_node(GraphNode& n) {
        auto deps = actual_deps(n);
        if (deps.empty()) return;

        auto& input0 = nodes_[deps[0]];
        if (!input0.computed) throw std::runtime_error("依赖节点尚未计算");
        const float* in0 = input0.output.data();

        switch (n.op) {
        case GraphOp::Matmul: {
            auto& input1 = nodes_[deps[1]];
            const float* in1 = input1.output.data();
            int M = n.out_shape[0];
            int K = input0.out_shape[1];
            int N = n.out_shape[1];
            n.output.resize(M * N);
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0;
                    for (int k = 0; k < K; ++k)
                        sum += in0[i * K + k] * in1[k * N + j];
                    n.output[i * N + j] = sum;
                }
            }
            break;
        }
        case GraphOp::Add: {
            auto& input1 = nodes_[deps[1]];
            if (input0.out_shape == input1.out_shape) {
                // 快速路径：无广播
                int n_el = static_cast<int>(input0.output.size());
                n.output.resize(n_el);
                no_broadcast_exec<false>(in0, input1.output.data(), n.output.data(), n_el);
            } else {
                broadcast_exec<false>(input0, input1, n);
            }
            break;
        }
        case GraphOp::Mul: {
            auto& input1 = nodes_[deps[1]];
            if (input0.out_shape == input1.out_shape) {
                int n_el = static_cast<int>(input0.output.size());
                n.output.resize(n_el);
                no_broadcast_exec<true>(in0, input1.output.data(), n.output.data(), n_el);
            } else {
                broadcast_exec<true>(input0, input1, n);
            }
            break;
        }
        case GraphOp::Transpose: {
            int ndim = static_cast<int>(input0.out_shape.size());
            int dim0 = n.input_ids[1];
            int dim1 = n.input_ids[2];
            int total = 1;
            for (int d : n.out_shape) total *= d;
            n.output.resize(total);

            // 预计算源张量 stride（消除内层 O(ndim²) 重复乘法）
            std::vector<int> src_stride(ndim);
            int stride = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                src_stride[d] = stride;
                stride *= input0.out_shape[d];
            }

            // 预计算输出 stride
            std::vector<int> out_stride(ndim);
            stride = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                out_stride[d] = stride;
                stride *= n.out_shape[d];
            }

            for (int i = 0; i < total; ++i) {
                int remainder = i;
                int src_idx = 0;
                for (int d = 0; d < ndim; ++d) {
                    int coord = remainder / out_stride[d];
                    remainder %= out_stride[d];
                    // 对换 dim0 和 dim1 对应的坐标
                    int orig_d = (d == dim0) ? dim1 : (d == dim1) ? dim0 : d;
                    src_idx += coord * src_stride[orig_d];
                }
                n.output[i] = in0[src_idx];
            }
            break;
        }
        case GraphOp::Softmax: {
            int cols = n.out_shape.back();       // 最后一维 = 特征维度
            int rows = 1;
            for (size_t d = 0; d + 1 < n.out_shape.size(); ++d) rows *= n.out_shape[d];
            int total = rows * cols;
            n.output.resize(total);

            for (int r = 0; r < rows; ++r) {
                int offset = r * cols;
                // 1) 找最大值（防 overflow）
                float max_val = in0[offset];
                for (int c = 1; c < cols; ++c)
                    if (in0[offset + c] > max_val) max_val = in0[offset + c];
                // 2) exp 并求和
                float sum_exp = 0;
                for (int c = 0; c < cols; ++c) {
                    float e = std::exp(in0[offset + c] - max_val);
                    n.output[offset + c] = e;
                    sum_exp += e;
                }
                // 3) 归一化
                float inv = 1.0f / sum_exp;
                for (int c = 0; c < cols; ++c)
                    n.output[offset + c] *= inv;
            }
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

#endif // COMPUTE_GRAPH_H
