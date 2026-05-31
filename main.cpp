#include "tensor.h"
#include <cassert>

int main() {
    std::cout << "=== MiniTensor<float>: Day 1 Test (Rule of Three & Basics) ===" << std::endl;

    // 1. 测试构造与内存分配
    Tensor<float> t1(5);
    t1.fill(1.0f);
    t1.print_info("T1");

    // 2. 测试拷贝构造 (深拷贝验证)
    Tensor<float> t2 = t1; 
    t2.fill(2.0f);
    
    std::cout << "--- Modifying T2 ---" << std::endl;
    t1.print_info("T1"); // T1 应该还是 1.0
    t2.print_info("T2"); // T2 应该是 2.0

    // 3. 测试运算符重载
    Tensor<float> t3 = t1 + t2;
    t3.print_info("T3 (T1 + T2)");

    std::cout << "=== Day 1 Test Complete ===\n" << std::endl;

    // ==========================================
    // Day 2 Tests
    // ==========================================

    // 4. 测试 operator= (赋值运算符)
    std::cout << "--- Day 2 Test: operator= (Assignment) ---" << std::endl;
    {
        Tensor<float> a(3);
        a.fill(10.0f);
        
        Tensor<float> b(5);  // b 已经有数据了，size 不同
        b.fill(20.0f);
        
        b = a;  // 把 a 赋给 b (释放 b 原来的内存，分配新的，拷贝)
        b.print_info("B (after B = A)");
        
        a.fill(99.0f);  // 修改 a
        b.print_info("B (should be unchanged)");  // b 应该是深拷贝，不受影响
    }

    // 5. 测试 operator+= (就地加法 + 链式调用)
    std::cout << "\n--- Day 2 Test: operator+= (In-place Add) ---" << std::endl;
    {
        Tensor<float> x(3); x.fill(1.0f);
        Tensor<float> y(3); y.fill(2.0f);
        Tensor<float> z(3); z.fill(3.0f);
        
        // 测试 +=
        x += y;
        x.print_info("X (after X += Y)");  // 应该是 3, 3, 3
        
        // 测试链式: (X += Y) += Z
        // 注意 X 已经是 3 了，所以 (3+2)+3 = 8
        Tensor<float> p(3); p.fill(2.0f);
        Tensor<float> q(3); q.fill(3.0f);
        (p += q) += q;
        p.print_info("P (after chain (P+=Q)+=Q)");
    }

    // 6. 测试自赋值
    std::cout << "\n--- Day 2 Test: Self-Assignment ---" << std::endl;
    {
        Tensor<float> self(4);
        self.fill(7.7f);
        self = self;  // 应该安全通过，不崩溃
        self.print_info("Self (after self = self)");
    }

    // 7. 测试 const 正确性
    std::cout << "\n--- Day 2 Test: Const Correctness ---" << std::endl;
    {
        const Tensor<float> c(3);
        // c.fill(1.0f);  // ← 编译会报错，因为 const 对象不能调用非 const 方法
        // const 对象只能调用 const 方法：
        std::cout << "Const Tensor<float> size: " << c.size() << std::endl;
    }

    std::cout << "\n=== All Day 2 Tests Passed ===" << std::endl;

    // ==========================================
    // Day 3 Tests: Move Semantics
    // ==========================================

    // 8. 测试移动构造函数 — 从临时对象构造
    std::cout << "\n--- Day 3 Test: Move Constructor ---" << std::endl;
    {
        Tensor<float> source(4);
        source.fill(5.5f);
        source.print_info("Source (before move)");
        
        Tensor<float> dest(std::move(source));  // 显式 move
        dest.print_info("Dest (after move)");
        source.print_info("Source (after move, should be empty)");
        // 注意：source 被 move 后 data_ptr_ 是 nullptr，print_info 应该显示 (Empty/Null)
    }

    // 9. 测试移动赋值运算符
    std::cout << "\n--- Day 3 Test: Move Assignment ---" << std::endl;
    {
        Tensor<float> a(3); a.fill(1.0f);
        Tensor<float> b(6); b.fill(2.0f);
        
        b = std::move(a);  // 把 a 的资源偷给 b
        b.print_info("B (after move assignment)");
        a.print_info("A (after move, should be empty)");
    }

    // 10. 验证 operator+ 返回时走的是移动而不是拷贝
    std::cout << "\n--- Day 3 Test: operator+ should use move, not copy ---" << std::endl;
    {
        Tensor<float> x(3); x.fill(1.0f);
        Tensor<float> y(3); y.fill(2.0f);
        
        std::cout << "-- Calling x + y --" << std::endl;
        Tensor<float> z = x + y;
        std::cout << "-- Result received --" << std::endl;
        z.print_info("Z (x + y)");
        // 理想情况下，return result 时应该走 Move Constructor 而不是 Copy Constructor
        // （取决于编译器优化，但我们的移动实现要准备好）
    }

    std::cout << "\n=== All Day 3 Tests Passed ===" << std::endl;

    // ==========================================
    // Day 6 Tests: Shape & Stride
    // ==========================================

    // 11. 测试多维张量构造 + shape/stride 访问
    std::cout << "\n--- Day 6 Test: Multi-Dimensional Shape & Stride ---" << std::endl;
    {
        Tensor<float> t({2, 3, 4});

        // 验证 shape
        auto s = t.shape();
        std::cout << "Shape: [" << s[0] << ", " << s[1] << ", " << s[2] << "]" << std::endl;
        assert(s[0] == 2 && s[1] == 3 && s[2] == 4);

        // 验证 stride (row-major): {12, 4, 1}
        auto st = t.stride();
        std::cout << "Stride: [" << st[0] << ", " << st[1] << ", " << st[2] << "]" << std::endl;
        assert(st[0] == 12 && st[1] == 4 && st[2] == 1);

        // 验证 size = product of shape
        assert(t.size() == 24);

        // 验证数据可写可读
        t.fill(3.14f);
        assert(t.data()[0] == 3.14f);
        assert(t.data()[23] == 3.14f);
    }

    // 12. 测试 2D 张量 stride
    std::cout << "\n--- Day 6 Test: 2D Stride ---" << std::endl;
    {
        Tensor<float> t({5, 7});
        auto st = t.stride();
        std::cout << "Stride: [" << st[0] << ", " << st[1] << "]" << std::endl;
        assert(st[0] == 7 && st[1] == 1);
        assert(t.size() == 35);
    }

    // 13. 测试 1D 兼容构造 (int)
    std::cout << "\n--- Day 6 Test: 1D Backward Compat ---" << std::endl;
    {
        Tensor<float> t(8);
        auto s = t.shape();
        auto st = t.stride();
        std::cout << "Shape: [" << s[0] << "], Stride: [" << st[0] << "]" << std::endl;
        assert(s.size() == 1 && s[0] == 8);
        assert(st.size() == 1 && st[0] == 1);
    }

    // 14. 测试多维拷贝构造保留 shape/stride
    std::cout << "\n--- Day 6 Test: Copy Preserves Shape/Stride ---" << std::endl;
    {
        Tensor<float> a({3, 4});
        a.fill(5.0f);
        Tensor<float> b = a;
        assert(b.shape()[0] == 3 && b.shape()[1] == 4);
        assert(b.stride()[0] == 4 && b.stride()[1] == 1);
        assert(b.data()[0] == 5.0f);
        assert(b.data()[11] == 5.0f);
    }

    // 15. 测试多维移动构造保留 shape/stride
    std::cout << "\n--- Day 6 Test: Move Preserves Shape/Stride ---" << std::endl;
    {
        Tensor<float> a({2, 3, 5});
        a.fill(9.0f);
        Tensor<float> b(std::move(a));
        assert(b.shape()[0] == 2 && b.shape()[1] == 3 && b.shape()[2] == 5);
        assert(b.stride()[0] == 15 && b.stride()[1] == 5 && b.stride()[2] == 1);
        assert(a.size() == 0);  // moved-out
    }

    std::cout << "\n=== All Day 6 Tests Passed ===" << std::endl;

    // ==========================================
    // Day 8 Tests: matmul
    // ==========================================

    std::cout << "\n--- Day 8 Test: Matrix Multiplication (matmul) ---" << std::endl;

    // 16. 测试 2x3 × 3x2 = 2x2
    {
        Tensor<float> a({2, 3});
        Tensor<float> b({3, 2});

        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        float a_data[] = {1, 2, 3, 4, 5, 6};
        for (int i = 0; i < 6; i++) a.data()[i] = a_data[i];

        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]
        float b_data[] = {7, 8, 9, 10, 11, 12};
        for (int i = 0; i < 6; i++) b.data()[i] = b_data[i];

        // C = [[58, 64],
        //      [139, 154]]
        Tensor<float> c = matmul(a, b);

        c.print_info("C = A × B (2x3 × 3x2)");
        assert(c.shape()[0] == 2 && c.shape()[1] == 2);
        assert(c.data()[0] == 58);
        assert(c.data()[1] == 64);
        assert(c.data()[2] == 139);
        assert(c.data()[3] == 154);
        std::cout << "Test 16 PASSED" << std::endl;
    }

    // 17. 测试方阵 2x2 × 2x2
    {
        Tensor<float> a({2, 2});
        Tensor<float> b({2, 2});

        a.data()[0] = 1; a.data()[1] = 2;
        a.data()[2] = 3; a.data()[3] = 4;

        b.data()[0] = 5; b.data()[1] = 6;
        b.data()[2] = 7; b.data()[3] = 8;

        // C = [[19, 22],
        //      [43, 50]]
        Tensor<float> c = matmul(a, b);

        c.print_info("C = A × B (2x2)");
        assert(c.data()[0] == 19);
        assert(c.data()[1] == 22);
        assert(c.data()[2] == 43);
        assert(c.data()[3] == 50);
        std::cout << "Test 17 PASSED" << std::endl;
    }

    // 18. 测试维度不匹配应该抛异常
    {
        Tensor<float> a({2, 3});
        Tensor<float> b({4, 2});  // 3 != 4

        try {
            Tensor<float> c = matmul(a, b);
            std::cout << "Test 18 FAILED: should have thrown" << std::endl;
            return 1;
        } catch (const std::invalid_argument& e) {
            std::cout << "Test 18 PASSED: caught expected error" << std::endl;
        }
    }

    std::cout << "\n=== All Day 8 Tests Passed ===" << std::endl;

    // ==========================================
    // Day 9 Tests: Softmax & LayerNorm
    // ==========================================

    // 19. 测试 Softmax — 基础 2D
    std::cout << "\n--- Day 9 Test: Softmax (2D) ---" << std::endl;
    {
        Tensor<float> input({2, 3});
        input.data()[0] = 1.0f; input.data()[1] = 2.0f; input.data()[2] = 3.0f;  // row 0
        input.data()[3] = 4.0f; input.data()[4] = 5.0f; input.data()[5] = 6.0f;  // row 1

        Tensor<float> out = softmax(input);
        out.print_info("Softmax output");

        // 验证每行 sum ≈ 1.0
        float row0_sum = out.data()[0] + out.data()[1] + out.data()[2];
        float row1_sum = out.data()[3] + out.data()[4] + out.data()[5];
        std::cout << "Row 0 sum: " << row0_sum << ", Row 1 sum: " << row1_sum << std::endl;
        assert(std::abs(row0_sum - 1.0f) < 1e-6f);
        assert(std::abs(row1_sum - 1.0f) < 1e-6f);

        // 手动验证: softmax([1,2,3]) = [exp(1-3), exp(2-3), exp(3-3)] / sum
        // = [exp(-2), exp(-1), exp(0)] / (exp(-2) + exp(-1) + 1)
        float e0 = std::exp(-2.0f), e1 = std::exp(-1.0f), e2 = 1.0f;
        float denom = e0 + e1 + e2;
        assert(std::abs(out.data()[0] - e0 / denom) < 1e-6f);
        assert(std::abs(out.data()[2] - e2 / denom) < 1e-6f);

        std::cout << "Test 19 PASSED" << std::endl;
    }

    // 20. 测试 Softmax — 数值稳定性 (大值输入)
    std::cout << "\n--- Day 9 Test: Softmax Numerical Stability ---" << std::endl;
    {
        Tensor<float> input({1, 3});
        input.data()[0] = 100.0f; input.data()[1] = 101.0f; input.data()[2] = 102.0f;

        Tensor<float> out = softmax(input);
        out.print_info("Softmax with large values");

        float row_sum = out.data()[0] + out.data()[1] + out.data()[2];
        assert(std::abs(row_sum - 1.0f) < 1e-5f);
        // 不应该产生 inf/nan
        assert(!std::isnan(out.data()[0]));
        assert(!std::isinf(out.data()[0]));
        std::cout << "Test 20 PASSED" << std::endl;
    }

    // 21. 测试 LayerNorm — 基础
    std::cout << "\n--- Day 9 Test: LayerNorm (2D) ---" << std::endl;
    {
        Tensor<float> input({2, 4});
        input.data()[0] = 1.0f; input.data()[1] = 2.0f; input.data()[2] = 3.0f; input.data()[3] = 4.0f;
        input.data()[4] = 5.0f; input.data()[5] = 6.0f; input.data()[6] = 7.0f; input.data()[7] = 8.0f;

        Tensor<float> weight({4}); weight.fill(1.0f);  // gamma = 1
        Tensor<float> bias({4});   bias.fill(0.0f);    // beta = 0

        Tensor<float> out = layernorm(input, weight, bias);
        out.print_info("LayerNorm output (gamma=1, beta=0)");

        // 验证每行均值 ≈ 0
        float row0_mean = (out.data()[0] + out.data()[1] + out.data()[2] + out.data()[3]) / 4.0f;
        float row1_mean = (out.data()[4] + out.data()[5] + out.data()[6] + out.data()[7]) / 4.0f;
        std::cout << "Row 0 mean: " << row0_mean << ", Row 1 mean: " << row1_mean << std::endl;
        assert(std::abs(row0_mean) < 1e-5f);
        assert(std::abs(row1_mean) < 1e-5f);

        std::cout << "Test 21 PASSED" << std::endl;
    }

    // 22. 测试 LayerNorm — 仿射变换 (gamma != 1, beta != 0)
    std::cout << "\n--- Day 9 Test: LayerNorm with gamma & beta ---" << std::endl;
    {
        Tensor<float> input({1, 3});
        input.data()[0] = 1.0f; input.data()[1] = 2.0f; input.data()[2] = 3.0f;

        Tensor<float> weight({3});
        weight.data()[0] = 2.0f; weight.data()[1] = 3.0f; weight.data()[2] = 1.0f;
        Tensor<float> bias({3});
        bias.data()[0] = 0.5f; bias.data()[1] = -0.5f; bias.data()[2] = 1.0f;

        Tensor<float> out = layernorm(input, weight, bias);
        out.print_info("LayerNorm output (custom gamma, beta)");

        std::cout << "Test 22 PASSED" << std::endl;
    }

    // 23. 测试 LayerNorm — 尺寸不匹配应该抛异常
    std::cout << "\n--- Day 9 Test: LayerNorm shape mismatch ---" << std::endl;
    {
        Tensor<float> input({2, 4});
        input.fill(1.0f);
        Tensor<float> weight({3}); weight.fill(1.0f);  // 不匹配! 应该是 4
        Tensor<float> bias({4}); bias.fill(0.0f);

        try {
            layernorm(input, weight, bias);
            std::cout << "Test 23 FAILED: should have thrown" << std::endl;
            return 1;
        } catch (const std::invalid_argument& e) {
            std::cout << "Test 23 PASSED: caught expected error: " << e.what() << std::endl;
        }
    }

    std::cout << "\n=== All Day 9 Tests Passed ===" << std::endl;

    // ==========================================
    // Day 10 Tests: Simple Inference Demo
    // ==========================================

    std::cout << "\n=== Day 10: Simple Inference Demo ===" << std::endl;

    // 24. 简单分类器: Input -> Linear -> Softmax -> Class Probabilities
    std::cout << "\n--- Day 10 Test: Simple Classifier (Linear + Softmax) ---" << std::endl;
    {
        // 场景: 3 分类问题，输入 4 维特征
        // 输入: [batch=1, features=4]
        Tensor<float> input({1, 4});
        input.data()[0] = 0.5f; input.data()[1] = -1.0f;
        input.data()[2] = 2.0f; input.data()[3] = 0.3f;

        // 权重: [4, 3] (4 输入特征, 3 输出类别)
        Tensor<float> weight({4, 3});
        weight.data()[0] =  0.1f; weight.data()[1] = -0.2f; weight.data()[2] =  0.3f;
        weight.data()[3] =  0.4f; weight.data()[4] =  0.5f; weight.data()[5] = -0.1f;
        weight.data()[6] = -0.3f; weight.data()[7] =  0.2f; weight.data()[8] =  0.6f;
        weight.data()[9] =  0.1f; weight.data()[10] = -0.4f; weight.data()[11] = 0.2f;

        // bias: [3]
        Tensor<float> bias({3});
        bias.data()[0] = 0.1f; bias.data()[1] = 0.2f; bias.data()[2] = -0.1f;

        // logits = input @ weight + bias
        // 用 matmul 做 input(1x4) × weight(4x3) = logits(1x3)
        Tensor<float> logits = matmul(input, weight);

        // 手动加 bias (逐行加)
        for (int j = 0; j < 3; j++) {
            logits.data()[j] += bias.data()[j];
        }
        logits.print_info("Logits (pre-softmax)");

        // softmax -> 概率分布
        Tensor<float> probs = softmax(logits);
        probs.print_info("Class Probabilities (post-softmax)");

        // 验证 sum = 1
        float prob_sum = probs.data()[0] + probs.data()[1] + probs.data()[2];
        std::cout << "Probability sum: " << prob_sum << std::endl;
        assert(std::abs(prob_sum - 1.0f) < 1e-5f);

        // 找 argmax
        int predicted = 0;
        if (probs.data()[1] > probs.data()[predicted]) predicted = 1;
        if (probs.data()[2] > probs.data()[predicted]) predicted = 2;
        std::cout << "Predicted class: " << predicted << std::endl;

        std::cout << "Test 24 PASSED" << std::endl;
    }

    // 25. Mini Transformer Block: LayerNorm -> QKV Projection -> Attention -> Output
    std::cout << "\n--- Day 10 Test: Mini Transformer Block ---" << std::endl;
    {
        // 模拟: batch=1, seq_len=3, hidden=4
        Tensor<float> x({3, 4});
        // 填入一些模拟的 token embeddings
        float x_data[] = {
            0.1f, -0.2f,  0.3f,  0.1f,   // token 0
           -0.1f,  0.4f, -0.3f,  0.2f,   // token 1
            0.2f,  0.1f,  0.1f, -0.4f,   // token 2
        };
        for (int i = 0; i < 12; i++) x.data()[i] = x_data[i];
        x.print_info("Input embeddings (3 tokens, hidden=4)");

        // Step 1: LayerNorm
        Tensor<float> gamma({4}); gamma.fill(1.0f);
        Tensor<float> beta({4});  beta.fill(0.0f);
        Tensor<float> x_norm = layernorm(x, gamma, beta);
        x_norm.print_info("After LayerNorm (per-token normalized)");

        // Step 2: QKV projection (简化: 用 matmul)
        // hidden=4, 每个头 dim=2, 这里简化为 Q = x_norm @ Wq
        Tensor<float> Wq({4, 2});
        Wq.data()[0] = 0.5f; Wq.data()[1] = -0.3f;
        Wq.data()[2] = 0.2f; Wq.data()[3] = 0.6f;
        Wq.data()[4] = -0.1f; Wq.data()[5] = 0.4f;
        Wq.data()[6] = 0.3f; Wq.data()[7] = -0.2f;

        // Q = x_norm(3x4) @ Wq(4x2) = Q(3x2)
        Tensor<float> Q = matmul(x_norm, Wq);
        Q.print_info("Query (Q)");

        // K = Q (简化, self-attention 中 K=Q)
        Tensor<float> K = Q;  // 移动语义

        // V = x_norm @ Wv (类似)
        Tensor<float> Wv({4, 2});
        Wv.data()[0] = 0.3f; Wv.data()[1] = 0.1f;
        Wv.data()[2] = -0.2f; Wv.data()[3] = 0.5f;
        Wv.data()[4] = 0.4f; Wv.data()[5] = -0.1f;
        Wv.data()[6] = 0.1f; Wv.data()[7] = 0.3f;
        Tensor<float> V = matmul(x_norm, Wv);
        V.print_info("Value (V)");

        // Step 3: Attention scores = Q @ K^T (简化)
        // K^T = (3x2)^T = (2x3), 所以需要手动构造
        Tensor<float> Kt({2, 3});
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                Kt.data()[j * 3 + i] = K.data()[i * K.stride()[0] + j];

        // scores = Q(3x2) @ Kt(2x3) = (3x3)
        Tensor<float> scores = matmul(Q, Kt);
        scores.print_info("Attention scores (Q @ K^T)");

        // Step 4: softmax on scores (按行)
        Tensor<float> attention = softmax(scores);
        attention.print_info("Attention weights (softmax)");

        // 验证每行 sum = 1
        for (int i = 0; i < 3; i++) {
            float row_sum = 0;
            for (int j = 0; j < 3; j++) row_sum += attention.data()[i * attention.stride()[0] + j];
            assert(std::abs(row_sum - 1.0f) < 1e-5f);
        }

        // Step 5: Output = attention @ V
        Tensor<float> output = matmul(attention, V);
        output.print_info("Attention output");

        std::cout << "Test 25 PASSED" << std::endl;
    }

    std::cout << "\n=== All Day 10 Tests Passed ===" << std::endl;
    std::cout << "\n=== MiniTensor v0.2 Complete! ===" << std::endl;

    return 0;
}
