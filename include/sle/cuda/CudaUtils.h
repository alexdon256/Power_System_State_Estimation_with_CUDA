/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * CUDA utility macros and templates for common patterns
 */

#ifndef SLE_CUDA_CUDAUTILS_H
#define SLE_CUDA_CUDAUTILS_H

#include <sle/Types.h>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#else
using cudaError_t = int;
#define cudaSuccess 0
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA disabled"; }
#endif

namespace sle {
namespace cuda {

// ============================================================================
// CUDA Error Checking Macros
// ============================================================================

// Check CUDA call and handle error
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            /* Error handling can be customized */ \
        } \
    } while(0)

// Check CUDA call and return on error
#define CUDA_CHECK_RETURN(call, ret_val) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            return (ret_val); \
        } \
    } while(0)

// Check CUDA call and throw on error
#define CUDA_CHECK_THROW(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// ============================================================================
// Memory Allocation Macros
// ============================================================================

// Reallocate buffer if needed (grows only, never shrinks)
#define CUDA_REALLOC_IF_NEEDED(ptr, size_var, required_size, type) \
    do { \
        if ((size_var) < (required_size)) { \
            if ((ptr)) cudaFree((ptr)); \
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), \
                                         (required_size) * sizeof(type)); \
            if (err == cudaSuccess) { \
                (size_var) = (required_size); \
            } else { \
                (ptr) = nullptr; \
                (size_var) = 0; \
            } \
        } \
    } while(0)

// Allocate buffer with cleanup on failure
#define CUDA_ALLOC_OR_CLEANUP(ptr, size, type, cleanup_code) \
    do { \
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), \
                                     (size) * sizeof(type)); \
        if (err != cudaSuccess) { \
            cleanup_code; \
            return false; \
        } \
    } while(0)

// Allocate pinned memory with cleanup on failure
#define CUDA_ALLOC_HOST_OR_CLEANUP(ptr, size, type, cleanup_code) \
    do { \
        cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&(ptr)), \
                                         (size) * sizeof(type)); \
        if (err != cudaSuccess) { \
            cleanup_code; \
            return false; \
        } \
    } while(0)

// ============================================================================
// Kernel Launch Configuration (Template Metaprogramming)
// ============================================================================

// Compile-time kernel configuration
template<Index BlockSize = 256>
struct KernelConfig {
    static constexpr Index blockSize = BlockSize;
    
    // Calculate grid size for 1D kernel
    template<typename SizeType>
    static constexpr Index gridSize(SizeType n) {
        return (n + blockSize - 1) / blockSize;
    }
    
    // Calculate grid dimension for 1D kernel
    template<typename SizeType>
    static constexpr dim3 gridDim3(SizeType n) {
        return dim3(gridSize(n));
    }
    
    // Calculate shared memory size for reduction (one value per warp)
    static constexpr size_t reductionSharedMemSize() {
        constexpr Index warpsPerBlock = (blockSize + 32 - 1) / 32;
        return warpsPerBlock * sizeof(Real);
    }
};

// Optimal block size selection based on problem size
template<size_t ProblemSize>
struct OptimalBlockSize {
    static constexpr Index value = 
        (ProblemSize < 1024) ? 128 :
        (ProblemSize < 4096) ? 256 : 512;
};

// ============================================================================
// Kernel Launch Macro with Error Checking
// ============================================================================

// Launch kernel with automatic error checking
#define LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) \
    do { \
        kernel<<<(grid), (block), (shared), (stream)>>>(__VA_ARGS__); \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            /* Error can be logged or handled */ \
        } \
    } while(0)

// Launch 1D kernel with standard configuration
#define LAUNCH_KERNEL_1D(kernel, n, stream, ...) \
    do { \
        constexpr Index blockSize = 256; \
        const Index gridSize = KernelConfig<blockSize>::gridSize(n); \
        LAUNCH_KERNEL(kernel, gridSize, blockSize, 0, (stream), __VA_ARGS__); \
    } while(0)

// Launch kernel with reduction (shared memory for partial sums)
#define LAUNCH_KERNEL_REDUCE(kernel, n, stream, ...) \
    do { \
        constexpr Index blockSize = 256; \
        const Index gridSize = KernelConfig<blockSize>::gridSize(n); \
        const size_t sharedMem = KernelConfig<blockSize>::reductionSharedMemSize(); \
        LAUNCH_KERNEL(kernel, gridSize, blockSize, sharedMem, stream, __VA_ARGS__); \
    } while(0)

// ============================================================================
// Type Traits for CUDA Operations
// ============================================================================

template<typename T>
struct CudaTypeTraits {
    static constexpr bool useFMA = true;
    static constexpr bool useFastMath = std::is_floating_point_v<T>;
    static constexpr bool isDouble = std::is_same_v<T, double>;
    static constexpr bool isFloat = std::is_same_v<T, float>;
};

// ============================================================================
// Memory Management Utilities
// ============================================================================

// Allocate or resize buffer (template function)
template<typename T>
inline bool allocateBuffer(T*& ptr, size_t& currentSize, size_t requiredSize) {
    if (currentSize < requiredSize) {
        if (ptr) {
            cudaFree(ptr);
        }
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), 
                                     requiredSize * sizeof(T));
        if (err == cudaSuccess) {
            currentSize = requiredSize;
            return true;
        } else {
            ptr = nullptr;
            currentSize = 0;
            return false;
        }
    }
    return true;  // Already large enough
}

// Free buffer
template<typename T>
inline void freeBuffer(T*& ptr, size_t& size) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
        size = 0;
    }
}

// Ensure capacity (alias for allocateBuffer)
template<typename T>
inline bool ensureCapacity(T*& ptr, size_t& currentSize, size_t requiredSize) {
    return allocateBuffer(ptr, currentSize, requiredSize);
}

// Allocate pinned memory
template<typename T>
inline bool allocatePinnedBuffer(T*& ptr, size_t& currentSize, size_t requiredSize) {
    if (currentSize < requiredSize) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
        cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&ptr), 
                                         requiredSize * sizeof(T));
        if (err == cudaSuccess) {
            currentSize = requiredSize;
            return true;
        } else {
            ptr = nullptr;
            currentSize = 0;
            return false;
        }
    }
    return true;
}

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDAUTILS_H

