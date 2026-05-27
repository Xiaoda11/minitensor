#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <memory>
#include<vector>

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

    // 辅助函数：填充数据
    void fill(T value);

    // 打印张量信息 (方便调试)
    void print_info(const std::string& name) const;

    // 右值引用
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    const std::vector<int>& shape() const;
    const std::vector<int>& stride() const;
    

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
#endif
