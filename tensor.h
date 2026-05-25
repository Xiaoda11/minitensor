#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cstring>
#include <stdexcept>
template<typename T> 
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
    T* data();
    const T* data() const;
    int size() const;

    // 运算符重载：实现 A + B
    Tensor operator+(const Tensor& other) const;

    // 辅助函数：填充数据
    void fill(T value);

    // 打印张量信息 (方便调试)
    void print_info(const std::string& name) const;

    // 右值引用
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    

private:
    T* data_ptr_; // 模拟显存/内存指针
    int size_;
};

// 1. 构造函数：分配内存
template<typename T>
Tensor<T>::Tensor(int size) : size_(size), data_ptr_(nullptr) {
    std::cout << "[Constructor] Allocating " << size_ << " typename Ts." << std::endl;
    
    // TODO: 使用 new 分配内存
    data_ptr_ = new T[size_];
}

// 2. 析构函数：释放内存
template<typename T>
Tensor<T>::~Tensor() {
    std::cout << "[Destructor] Freeing memory." << std::endl;
    
    // TODO: 使用 delete[] 释放内存
    delete[] data_ptr_;
}

// 3. 拷贝构造函数 (实现深拷贝)
template<typename T>
Tensor<T>::Tensor(const Tensor& other) : size_(other.size_), data_ptr_(nullptr) {
    std::cout << "[Copy Constructor] Deep copying " << size_ << " Ts." << std::endl;
    
    
    data_ptr_ = new T[size_];
    std::memcpy(data_ptr_, other.data_ptr_, size_ * sizeof(T));
}
// 右值引用：移动构造函数
template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : size_(other.size_), data_ptr_(other.data_ptr_) {
    other.size_ = 0;
    other.data_ptr_ = nullptr;
    std::cout << "[Move Constructor] Resource stolen from temporary." << std::endl;
}

// 获取数据指针
template<typename T>
T* Tensor<T>::data() {
    return data_ptr_;
}
template<typename T>
const T* Tensor<T>::data() const {
    return data_ptr_;
}
template<typename T>
int Tensor<T>::size() const {
    return size_;
}

// 填充数据
template<typename T>
void Tensor<T>::fill(T value) {
    if (!data_ptr_) return; 
    for (int i = 0; i < size_; ++i) {
        data_ptr_[i] = value;
    }
}
template<typename T>
void Tensor<T>::print_info(const std::string& name) const {
    std::cout << name << " [Size: " << size_ << "]: ";
    if (!data_ptr_) {
        std::cout << "(Empty/Null)" << std::endl;
        return;
    }
    for (int i = 0; i < size_; ++i) {
        std::cout << data_ptr_[i] << " ";
    }
    std::cout << std::endl;
}

// 运算符重载：Tensor A + Tensor B
template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    if (this->size_ != other.size_) {
        throw std::invalid_argument("Tensor sizes must match for addition.");
    }
    
    // 1. 创建一个新 Tensor (结果)
    Tensor result(this->size_);
    
    // TODO: 2. 执行加法逻辑
    for (int i = 0; i < size_; ++i) {
            result.data_ptr_[i] = this->data_ptr_[i] + other.data_ptr_[i];
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
        delete[] this->data_ptr_;
        // 3. 重新分配 + 拷贝（跟拷贝构造函数一样）
        this->size_ = other.size_;
        this->data_ptr_= new T [other.size_];
        std::memcpy(this->data_ptr_, other.data_ptr_, size_ * sizeof(T));
        // 4. 返回 *this
        return *this;
    }
    // 运算符重载：Tensor A += Tensor B
template<typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor& other){
        if (size_ != other.size_)
        {
          throw std::invalid_argument("Tensor sizes must match for addition.");
        }
        for (int i = 0; i < size_; i++)
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
    delete []this->data_ptr_;
    this->data_ptr_ = other.data_ptr_;
    this->size_ = other.size_;
    other.data_ptr_= nullptr;
    other.size_= 0;
    return *this;
    }
#endif
