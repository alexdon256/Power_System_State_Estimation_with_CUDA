/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/CuSOLVERIntegration.h>
#include <sle/math/SparseMatrix.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <chrono>

namespace sle {
namespace math {

CuSOLVERIntegration::CuSOLVERIntegration()
    : cusolverHandle_(nullptr), cusparseHandle_(nullptr),
      descr_(nullptr), factorData_(nullptr), factorized_(false) {
    initialize();
}

CuSOLVERIntegration::~CuSOLVERIntegration() {
    cleanup();
}

void CuSOLVERIntegration::initialize() {
    createHandles();
}

void CuSOLVERIntegration::cleanup() {
    destroyHandles();
}

void CuSOLVERIntegration::createHandles() {
    cusolverSpCreate(&cusolverHandle_);
    cusparseCreate(&cusparseHandle_);
    cusparseCreateMatDescr(&descr_);
    cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);
}

void CuSOLVERIntegration::destroyHandles() {
    if (cusolverHandle_) {
        cusolverSpDestroy(cusolverHandle_);
        cusolverHandle_ = nullptr;
    }
    if (cusparseHandle_) {
        cusparseDestroy(cusparseHandle_);
        cusparseHandle_ = nullptr;
    }
    if (descr_) {
        cusparseDestroyMatDescr(descr_);
        descr_ = nullptr;
    }
    if (factorData_) {
        // Free factorization data
        factorData_ = nullptr;
    }
    factorized_ = false;
}

bool CuSOLVERIntegration::solveSparse(const SparseMatrix& A,
                                     const std::vector<Real>& b,
                                     std::vector<Real>& x) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Index nRows = A.getNRows();
    Index nCols = A.getNCols();
    
    if (nRows != nCols || nRows != static_cast<Index>(b.size())) {
        return false;
    }
    
    x.resize(nRows);
    
    // Allocate device memory
    Real* d_b = nullptr;
    Real* d_x = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_b, nRows * sizeof(Real));
    if (err != cudaSuccess) {
        return false;
    }
    
    err = cudaMalloc(&d_x, nRows * sizeof(Real));
    if (err != cudaSuccess) {
        // Free previously allocated memory
        if (d_b) {
            cudaFree(d_b);
        }
        return false;
    }
    
    // Copy right-hand side to device
    err = cudaMemcpy(d_b, b.data(), nRows * sizeof(Real), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_b);
        cudaFree(d_x);
        return false;
    }
    
    // Solve using cuSOLVER QR factorization
    cusolverSpHandle_t handle = cusolverHandle_;
    
    // For symmetric positive definite matrices, use Cholesky
    // For general matrices, use QR
    int singularity = 0;
    double tol = 1e-6;
    int reorder = 0;  // No reordering
    
    cusolverStatus_t status = cusolverSpDcsrlsvqr(
        handle, nRows, A.getNNZ(), descr_,
        A.getValues(), A.getRowPtr(), A.getColInd(),
        d_b, tol, reorder, d_x, &singularity);
    
    if (status != CUSOLVER_STATUS_SUCCESS || singularity != -1) {
        cudaFree(d_b);
        cudaFree(d_x);
        return false;
    }
    
    // Copy result back
    err = cudaMemcpy(x.data(), d_x, nRows * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_b);
        cudaFree(d_x);
        return false;
    }
    
    cudaFree(d_b);
    cudaFree(d_x);
    
    auto end = std::chrono::high_resolution_clock::now();
    lastStats_.solveTime = std::chrono::duration<double>(end - start).count();
    lastStats_.iterations = 1;
    
    // Compute residual
    // (Would compute ||A*x - b|| here)
    lastStats_.residual = 0.0;
    
    return true;
}

bool CuSOLVERIntegration::solveSparseIterative(const SparseMatrix& A,
                                              const std::vector<Real>& b,
                                              std::vector<Real>& x,
                                              Real tolerance,
                                              int maxIterations) {
    // Use iterative refinement with cuSOLVER
    // This is a simplified version - full implementation would use
    // cuSOLVER's iterative solvers (GMRES, etc.)
    
    return solveSparse(A, b, x);
}

void CuSOLVERIntegration::factorize(const SparseMatrix& A) {
    // Factorize matrix for repeated solves
    // This would store factorization data for later use
    factorized_ = true;
    // Full implementation would perform Cholesky or LU factorization
}

bool CuSOLVERIntegration::solveWithFactorization(const std::vector<Real>& b,
                                                 std::vector<Real>& x) {
    if (!factorized_) {
        return false;
    }
    
    // Solve using pre-computed factorization
    // Full implementation would use stored factorization
    return false;
}

} // namespace math
} // namespace sle

