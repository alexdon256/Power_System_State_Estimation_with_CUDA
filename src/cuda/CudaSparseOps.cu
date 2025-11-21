/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * This file provides cuSPARSE-based sparse matrix operations.
 * cuSPARSE is NVIDIA's highly optimized library for sparse matrix operations,
 * providing better performance than custom kernels for most use cases.
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <sle/Types.h>

namespace sle {
namespace cuda {

// Sparse matrix-vector product: y = A * x
// Uses cuSPARSE for optimal performance
void sparseMatVec(cusparseHandle_t handle,
                  const Real* values, const Index* rowPtr, const Index* colInd,
                  const Real* x, Real* y, Index nRows, Index nCols) {
    if (nRows == 0 || nCols == 0) {
        return;
    }
    
    const Real alpha = 1.0;
    const Real beta = 0.0;
    
    Index nnz = rowPtr[nRows] - rowPtr[0];
    
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCsr(&mat,
                      nRows, nCols, nnz,
                      const_cast<Index*>(rowPtr), const_cast<Index*>(colInd),
                      const_cast<Real*>(values),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, nCols, const_cast<Real*>(x), CUDA_R_64F);
    cusparseCreateDnVec(&vecY, nRows, y, CUDA_R_64F);
    
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, mat, vecX, &beta, vecY,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    if (bufferSize > 0) cudaMalloc(&dBuffer, bufferSize);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, mat, vecX, &beta, vecY,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    if (dBuffer) cudaFree(dBuffer);
    
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(mat);
}

// Weighted sparse matrix-vector product: y = W * A * x
// First computes A * x using cuSPARSE, then applies diagonal weight matrix W element-wise
__global__ void weightedMultiplyKernel(const Real* weights, const Real* temp, Real* y, Index nRows) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows) {
        y[idx] = weights[idx] * temp[idx];
    }
}

void weightedSparseMatVec(cusparseHandle_t handle,
                          const Real* values, const Index* rowPtr, const Index* colInd,
                          const Real* weights, const Real* x, Real* y, 
                          Index nRows, Index nCols) {
    if (nRows == 0 || nCols == 0) {
        return;
    }
    
    // Temporary buffer for A * x
    Real* temp = nullptr;
    cudaError_t err = cudaMalloc(&temp, nRows * sizeof(Real));
    if (err != cudaSuccess) {
        return;  // Allocation failed, cannot proceed
    }
    
    // Compute A * x using cuSPARSE
    sparseMatVec(handle, values, rowPtr, colInd, x, temp, nRows, nCols);
    
    // Apply weights: y = W * (A * x) element-wise using CUDA kernel
    const Index blockSize = 256;
    const Index gridSize = (nRows + blockSize - 1) / blockSize;
    weightedMultiplyKernel<<<gridSize, blockSize>>>(weights, temp, y, nRows);
    
    cudaFree(temp);
}

// Dot product using cuBLAS for optimal performance
// Note: This function requires a cuBLAS handle. For convenience, you can use
// CudaVectorOps::dot() which manages its own handle, or pass a handle here.
Real dotProduct(cublasHandle_t cublasHandle, const Real* a, const Real* b, Index n) {
    if (n == 0) {
        return 0.0;
    }
    
    Real result = 0.0;
    Real* d_a = const_cast<Real*>(a);  // cuBLAS doesn't modify, but needs non-const
    Real* d_b = const_cast<Real*>(b);
    
    cublasStatus_t status = cublasDdot(cublasHandle, n, d_a, 1, d_b, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        // Fallback: return 0 if cuBLAS fails (shouldn't happen in normal operation)
        return 0.0;
    }
    
    return result;
}

} // namespace cuda
} // namespace sle

