/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_CUSOLVERINTEGRATION_H
#define SLE_MATH_CUSOLVERINTEGRATION_H

#include <sle/Types.h>
#include <sle/math/SparseMatrix.h>
#include <vector>
#include <memory>

#ifdef USE_CUDA
#include <cusolverSp.h>
#include <cusparse.h>
#else
using cusolverSpHandle_t = void*;
using cusparseHandle_t = void*;
using cusparseMatDescr_t = void*;
#endif

namespace sle {
namespace math {

class CuSOLVERIntegration {
public:
    CuSOLVERIntegration();
    ~CuSOLVERIntegration();
    
    // Initialize cuSOLVER handles
    void initialize();
    void cleanup();
    
    // Set CUDA stream for all cuSOLVER/cuSPARSE operations
    void setStream(cudaStream_t stream);
    
    // Solve sparse linear system: A * x = b (host data)
    // Using cuSOLVER's sparse direct solver
    bool solveSparse(const SparseMatrix& A, const std::vector<Real>& b,
                    std::vector<Real>& x);
    
    // CUDA-EXCLUSIVE: Solve sparse linear system entirely on GPU: A * x = b
    // d_b and d_x are device pointers, no host transfers
    bool solveSparseGPU(const SparseMatrix& A, const Real* d_b, Real* d_x, Index n);
    
    // Solve using iterative refinement
    bool solveSparseIterative(const SparseMatrix& A, const std::vector<Real>& b,
                             std::vector<Real>& x, Real tolerance = 1e-6,
                             int maxIterations = 100);
    
    // Factorize matrix (for repeated solves)
    void factorize(const SparseMatrix& A);
    
    // Solve using factorization
    bool solveWithFactorization(const std::vector<Real>& b, std::vector<Real>& x);
    
    // Get solver statistics
    struct SolverStats {
        double solveTime;
        int iterations;
        Real residual;
    };
    
    SolverStats getLastStats() const { return lastStats_; }
    
private:
    cusolverSpHandle_t cusolverHandle_;
    cusparseHandle_t cusparseHandle_;
    cusparseMatDescr_t descr_;
    
    // Factorization data
    void* factorData_;
    bool factorized_;
    
    SolverStats lastStats_;
    
    void createHandles();
    void destroyHandles();
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_CUSOLVERINTEGRATION_H

