/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_CUSOLVERINTEGRATION_H
#define SLE_MATH_CUSOLVERINTEGRATION_H

#include <sle/Types.h>
#include <sle/math/SparseMatrix.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <vector>
#include <memory>

namespace sle {
namespace math {

class CuSOLVERIntegration {
public:
    CuSOLVERIntegration();
    ~CuSOLVERIntegration();
    
    // Initialize cuSOLVER handles
    void initialize();
    void cleanup();
    
    // Solve sparse linear system: A * x = b
    // Using cuSOLVER's sparse direct solver
    bool solveSparse(const SparseMatrix& A, const std::vector<Real>& b,
                    std::vector<Real>& x);
    
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

