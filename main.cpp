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

    return 0;
}
