#include "tensor.h"
#include <cstring>
#include <stdexcept>

// === 你的作业：补全下面的实现 ===

// 1. 构造函数：分配内存
Tensor::Tensor(int size) : size_(size), data_ptr_(nullptr) {
    std::cout << "[Constructor] Allocating " << size_ << " floats." << std::endl;
    
    // TODO: 使用 new 分配内存
    data_ptr_ = new float[size_];
}

// 2. 析构函数：释放内存
Tensor::~Tensor() {
    std::cout << "[Destructor] Freeing memory." << std::endl;
    
    // TODO: 使用 delete[] 释放内存
    delete[] data_ptr_;
}

// 3. 拷贝构造函数 (实现深拷贝)
Tensor::Tensor(const Tensor& other) : size_(other.size_), data_ptr_(nullptr) {
    std::cout << "[Copy Constructor] Deep copying " << size_ << " floats." << std::endl;
    
    
    data_ptr_ = new float[size_];
    std::memcpy(data_ptr_, other.data_ptr_, size_ * sizeof(float));
}

// 获取数据指针
float* Tensor::data() {
    return data_ptr_;
}

const float* Tensor::data() const {
    return data_ptr_;
}

int Tensor::size() const {
    return size_;
}

// 填充数据
void Tensor::fill(float value) {
    if (!data_ptr_) return; 
    for (int i = 0; i < size_; ++i) {
        data_ptr_[i] = value;
    }
}

void Tensor::print_info(const std::string& name) const {
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
Tensor Tensor::operator+(const Tensor& other) const {
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
Tensor& Tensor::operator=(const Tensor& other) {
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
        this->data_ptr_= new float [other.size_];
        std::memcpy(this->data_ptr_, other.data_ptr_, size_ * sizeof(float));
        // 4. 返回 *this
        return *this;
    }
    // 运算符重载：Tensor A += Tensor B
    Tensor& Tensor::operator+=(const Tensor& other){
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
