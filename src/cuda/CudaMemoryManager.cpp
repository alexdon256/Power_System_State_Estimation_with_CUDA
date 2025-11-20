/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/cuda/CudaMemoryManager.h>
#include <stdexcept>

namespace sle {
namespace cuda {

struct CudaMemoryManager::Impl {
    // Can add memory pools, statistics, etc. here
};

CudaMemoryManager::CudaMemoryManager() : pImpl_(std::make_unique<Impl>()) {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device");
    }
}

CudaMemoryManager::~CudaMemoryManager() = default;

void* CudaMemoryManager::allocateDevice(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }
    return ptr;
}

void CudaMemoryManager::freeDevice(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void* CudaMemoryManager::allocateUnified(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate unified memory");
    }
    return ptr;
}

void CudaMemoryManager::freeUnified(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void CudaMemoryManager::copyToDevice(const void* host, void* device, size_t bytes) {
    cudaError_t err = cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy to device");
    }
}

void CudaMemoryManager::copyToHost(const void* device, void* host, size_t bytes) {
    cudaError_t err = cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy to host");
    }
}

void CudaMemoryManager::synchronize() {
    cudaDeviceSynchronize();
}

int CudaMemoryManager::getDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void CudaMemoryManager::getDeviceProperties(int device, cudaDeviceProp& prop) {
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get device properties");
    }
}

} // namespace cuda
} // namespace sle

