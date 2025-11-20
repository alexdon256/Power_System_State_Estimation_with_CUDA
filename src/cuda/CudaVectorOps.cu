/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/cuda/CudaVectorOps.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace sle {
namespace cuda {

cublasHandle_t CudaVectorOps::handle_ = nullptr;
bool CudaVectorOps::initialized_ = false;

void CudaVectorOps::initialize() {
    if (!initialized_) {
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to initialize cuBLAS");
        }
        initialized_ = true;
    }
}

void CudaVectorOps::shutdown() {
    if (initialized_ && handle_) {
        cublasDestroy(handle_);
        handle_ = nullptr;
        initialized_ = false;
    }
}

void CudaVectorOps::axpy(Index n, Real alpha, const Real* x, Real* y) {
    if (!initialized_) initialize();
    
    Real* d_x = const_cast<Real*>(x);  // cuBLAS doesn't modify, but needs non-const
    Real* d_y = y;
    
    cublasStatus_t status = cublasDaxpy(handle_, n, &alpha, d_x, 1, d_y, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS axpy failed");
    }
}

void CudaVectorOps::scal(Index n, Real alpha, Real* x) {
    if (!initialized_) initialize();
    
    cublasStatus_t status = cublasDscal(handle_, n, &alpha, x, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS scal failed");
    }
}

Real CudaVectorOps::dot(Index n, const Real* x, const Real* y) {
    if (!initialized_) initialize();
    
    Real result = 0.0;
    Real* d_x = const_cast<Real*>(x);
    Real* d_y = const_cast<Real*>(y);
    
    cublasStatus_t status = cublasDdot(handle_, n, d_x, 1, d_y, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS dot failed");
    }
    
    return result;
}

Real CudaVectorOps::normSquared(Index n, const Real* x) {
    if (!initialized_) initialize();
    
    Real result = 0.0;
    Real* d_x = const_cast<Real*>(x);
    
    cublasStatus_t status = cublasDnrm2(handle_, n, d_x, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS nrm2 failed");
    }
    
    return result * result;  // Return squared norm
}

void CudaVectorOps::weightedAccumulate(Index n, Real alpha, const Real* x, 
                                       const Real* w, Real* y) {
    // y[i] = alpha * x[i] * w[i] + y[i]
    // This requires element-wise operations, so we use a custom kernel
    // For now, fallback to CPU or use cuBLAS with temporary arrays
    // This is a placeholder - would need custom kernel
}

Real CudaVectorOps::weightedSumSquares(Index n, const Real* x, const Real* w) {
    // result = sum(w[i] * x[i]^2)
    // This requires element-wise operations, so we use a custom kernel
    // For now, fallback to CPU or use cuBLAS with temporary arrays
    // This is a placeholder - would need custom kernel
    return 0.0;
}

} // namespace cuda
} // namespace sle

