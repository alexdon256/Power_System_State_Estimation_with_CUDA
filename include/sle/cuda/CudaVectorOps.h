/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_CUDA_CUDAVECTOROPS_H
#define SLE_CUDA_CUDAVECTOROPS_H

#include <sle/Types.h>
#include <cublas_v2.h>

namespace sle {
namespace cuda {

// GPU-accelerated vector operations using cuBLAS
class CudaVectorOps {
public:
    static void initialize();
    static void shutdown();
    
    // Vector addition: y = alpha * x + y
    static void axpy(Index n, Real alpha, const Real* x, Real* y);
    
    // Vector scaling: x = alpha * x
    static void scal(Index n, Real alpha, Real* x);
    
    // Dot product: result = x^T * y
    static Real dot(Index n, const Real* x, const Real* y);
    
    // Vector norm squared: result = ||x||^2
    static Real normSquared(Index n, const Real* x);
    
    // Element-wise multiply and accumulate: y[i] = alpha * x[i] * w[i] + y[i]
    static void weightedAccumulate(Index n, Real alpha, const Real* x, 
                                   const Real* w, Real* y);
    
    // Weighted sum of squares: result = sum(w[i] * x[i]^2)
    static Real weightedSumSquares(Index n, const Real* x, const Real* w);

private:
    static cublasHandle_t handle_;
    static bool initialized_;
};

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDAVECTOROPS_H

