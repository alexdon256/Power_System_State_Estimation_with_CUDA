/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_CUDA_CUDAMEMORYMANAGER_H
#define SLE_CUDA_CUDAMEMORYMANAGER_H

#include <cuda_runtime.h>
#include <sle/Types.h>
#include <vector>
#include <memory>

namespace sle {
namespace cuda {

class CudaMemoryManager {
public:
    CudaMemoryManager();
    ~CudaMemoryManager();
    
    // Allocate device memory
    void* allocateDevice(size_t bytes);
    void freeDevice(void* ptr);
    
    // Allocate unified memory
    void* allocateUnified(size_t bytes);
    void freeUnified(void* ptr);
    
    // Copy data
    void copyToDevice(const void* host, void* device, size_t bytes);
    void copyToHost(const void* device, void* host, size_t bytes);
    
    // Synchronize
    void synchronize();
    
    // Get device properties
    static int getDeviceCount();
    static void getDeviceProperties(int device, cudaDeviceProp& prop);
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

// Helper template for device arrays
template<typename T>
class DeviceArray {
public:
    DeviceArray() : data_(nullptr), size_(0), manager_(nullptr) {}
    
    explicit DeviceArray(CudaMemoryManager* manager, size_t size) 
        : manager_(manager), size_(size) {
        if (manager_ && size_ > 0) {
            data_ = static_cast<T*>(manager_->allocateDevice(size_ * sizeof(T)));
        }
    }
    
    ~DeviceArray() {
        if (data_ && manager_) {
            manager_->freeDevice(data_);
        }
    }
    
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    
    DeviceArray(DeviceArray&& other) noexcept 
        : data_(other.data_), size_(other.size_), manager_(other.manager_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.manager_ = nullptr;
    }
    
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (data_ && manager_) {
                manager_->freeDevice(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            manager_ = other.manager_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.manager_ = nullptr;
        }
        return *this;
    }
    
    void resize(size_t newSize) {
        if (data_ && manager_) {
            manager_->freeDevice(data_);
        }
        size_ = newSize;
        if (manager_ && size_ > 0) {
            data_ = static_cast<T*>(manager_->allocateDevice(size_ * sizeof(T)));
        } else {
            data_ = nullptr;
        }
    }
    
    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return size_; }
    
    void upload(const std::vector<T>& hostData) {
        if (manager_ && data_ && hostData.size() <= size_) {
            manager_->copyToDevice(hostData.data(), data_, hostData.size() * sizeof(T));
        }
    }
    
    void download(std::vector<T>& hostData) const {
        if (manager_ && data_ && size_ > 0) {
            hostData.resize(size_);
            manager_->copyToHost(data_, hostData.data(), size_ * sizeof(T));
        }
    }
    
private:
    T* data_;
    size_t size_;
    CudaMemoryManager* manager_;
};

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDAMEMORYMANAGER_H

