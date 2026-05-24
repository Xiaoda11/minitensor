#include "tensor.h"
#include <cassert>

int main() {
    std::cout << "=== MiniTensor: Day 1 Test (Rule of Three & Basics) ===" << std::endl;

    // 1. 测试构造与内存分配
    Tensor t1(5);
    t1.fill(1.0f);
    t1.print_info("T1");

    // 2. 测试拷贝构造 (深拷贝验证)
    Tensor t2 = t1; 
    t2.fill(2.0f);
    
    std::cout << "--- Modifying T2 ---" << std::endl;
    t1.print_info("T1"); // T1 应该还是 1.0
    t2.print_info("T2"); // T2 应该是 2.0

    // 3. 测试运算符重载
    Tensor t3 = t1 + t2;
    t3.print_info("T3 (T1 + T2)");

    std::cout << "=== Day 1 Test Complete ===\n" << std::endl;

    // ==========================================
    // Day 2 Tests
    // ==========================================

    // 4. 测试 operator= (赋值运算符)
    std::cout << "--- Day 2 Test: operator= (Assignment) ---" << std::endl;
    {
        Tensor a(3);
        a.fill(10.0f);
        
        Tensor b(5);  // b 已经有数据了，size 不同
        b.fill(20.0f);
        
        b = a;  // 把 a 赋给 b (释放 b 原来的内存，分配新的，拷贝)
        b.print_info("B (after B = A)");
        
        a.fill(99.0f);  // 修改 a
        b.print_info("B (should be unchanged)");  // b 应该是深拷贝，不受影响
    }

    // 5. 测试 operator+= (就地加法 + 链式调用)
    std::cout << "\n--- Day 2 Test: operator+= (In-place Add) ---" << std::endl;
    {
        Tensor x(3); x.fill(1.0f);
        Tensor y(3); y.fill(2.0f);
        Tensor z(3); z.fill(3.0f);
        
        // 测试 +=
        x += y;
        x.print_info("X (after X += Y)");  // 应该是 3, 3, 3
        
        // 测试链式: (X += Y) += Z
        // 注意 X 已经是 3 了，所以 (3+2)+3 = 8
        Tensor p(3); p.fill(2.0f);
        Tensor q(3); q.fill(3.0f);
        (p += q) += q;
        p.print_info("P (after chain (P+=Q)+=Q)");
    }

    // 6. 测试自赋值
    std::cout << "\n--- Day 2 Test: Self-Assignment ---" << std::endl;
    {
        Tensor self(4);
        self.fill(7.7f);
        self = self;  // 应该安全通过，不崩溃
        self.print_info("Self (after self = self)");
    }

    // 7. 测试 const 正确性
    std::cout << "\n--- Day 2 Test: Const Correctness ---" << std::endl;
    {
        const Tensor c(3);
        // c.fill(1.0f);  // ← 编译会报错，因为 const 对象不能调用非 const 方法
        // const 对象只能调用 const 方法：
        std::cout << "Const tensor size: " << c.size() << std::endl;
    }

    std::cout << "\n=== All Day 2 Tests Passed ===" << std::endl;
    return 0;
}
