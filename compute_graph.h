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

#endif // COMPUTE_GRAPH_H
