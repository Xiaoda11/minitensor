#ifndef MEMORY_POOL_H
 #define MEMORY_POOL_H
 #include <cstddef>
 #include<vector>
#include <memory>
template<typename T>
    class MemoryPool {
    public:
        MemoryPool(size_t pool_size);   // 预分配 pool_size 个 T
        ~MemoryPool();
    
        T* allocate(size_t count);      // 从池里分配 count 个 T
        // void deallocate(T* ptr);        // 归还到池里(后续再实现)
        void reset();
        size_t available() const;       // 剩余可用元素数
        size_t capacity() const;        // 总容量
    private:
    std::unique_ptr<T[]> pool_start_;
    T* pool_ptr_ ; // 模拟显存/内存指针
    size_t available_;       // 剩余可用元素数
    size_t capacity_;        // 总容量
    };
//构造函数
template<typename T>
MemoryPool<T>::MemoryPool(size_t pool_size){
    pool_start_ = std::make_unique<T[]> (pool_size);
    pool_ptr_ = pool_start_.get();
    available_ = pool_size ;
    capacity_ = pool_size ;
 }
//析构函数
template<typename T>
MemoryPool<T> :: ~ MemoryPool()= default ;

//从池里分配 count 个 T
template<typename T>
T* MemoryPool<T> ::  allocate(size_t count){
      if (available_  < count){
        throw std::invalid_argument("MemoryPool: not enough space");;
    }
    T* head_ptr_ = pool_ptr_ ;
    pool_ptr_ += count;
    available_ -= count;
    return head_ptr_;
}

// 归还到池里
template<typename T>
void MemoryPool<T> :: reset() {
        pool_ptr_ = pool_start_.get();  // 指针回到起点
        available_ = capacity_;  // 可用计数恢复
    }
template<typename T>
size_t  MemoryPool<T> ::available() const{return available_;}// 剩余可用元素数
template<typename T>
size_t MemoryPool<T> ::capacity() const {return capacity_;}       // 总容量
#endif