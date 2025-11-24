/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * CUDA sparse matrix operations using cuSPARSE
 */

#ifndef SLE_CUDA_CUDASPARSEOPS_H
#define SLE_CUDA_CUDASPARSEOPS_H

#include <sle/Types.h>
#include <cusparse.h>
#include <cublas.h>
#include <cuda_runtime.h>

namespace sle {
namespace cuda {

// Sparse matrix-vector product: y = A * x
void sparseMatVec(cusparseHandle_t handle,
                  const Real* values, const Index* rowPtr, const Index* colInd,
                  const Real* x, Real* y, Index nRows, Index nCols);

// Weighted sparse matrix-vector product: y = W * A * x
void weightedSparseMatVec(cusparseHandle_t handle,
                          const Real* values, const Index* rowPtr, const Index* colInd,
                          const Real* weights, const Real* x, Real* y, 
                          Index nRows, Index nCols);

// Scale sparse matrix rows by weights: H_scaled = W * H
// Note: Extracts stream from cusparseHandle_t if available
void scaleSparseMatrixRows(cusparseHandle_t handle,
                           const Real* H_values, const Index* H_rowPtr,
                           const Real* weights, Real* H_scaled,
                           Index nRows, Index nnz);

// Compute gain matrix G = H^T * W * H on GPU using cuSPARSE SpGEMM
// Allocates G_values and G_colInd on GPU, caller must free them
// If d_WH_values is provided (pooled buffer), uses it instead of allocating
// If d_spgemmBuffer1/2 are provided (pooled workspace), uses them (resizing if needed via reference)
// Returns true on success, false on failure (fallback to CPU)
bool computeGainMatrixGPU(cusparseHandle_t handle,
                          const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                          const Real* weights,
                          Real*& G_values, Index* G_rowPtr, Index*& G_colInd,
                          Index nMeas, Index nStates, Index& G_nnz,
                          Real* d_WH_values = nullptr, size_t WH_valuesSize = 0,
                          void*& d_spgemmBuffer1 = *((void**)nullptr), size_t& spgemmBuffer1Size = *((size_t*)nullptr),
                          void*& d_spgemmBuffer2 = *((void**)nullptr), size_t& spgemmBuffer2Size = *((size_t*)nullptr));

// Compute gain matrix G using direct formula: G_ij = sum_k(H_ki * w_k * H_kj)
// Alternative to SpGEMM, may be faster for very sparse matrices
// Allocates G_values and G_colInd on GPU, caller must free them
// Returns true on success, false on failure
bool computeGainMatrixDirect(cusparseHandle_t handle,
                             const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                             const Real* weights,
                             Real*& G_values, Index* G_rowPtr, Index*& G_colInd,
                             Index nMeas, Index nStates, Index& G_nnz);

// Dot product using cuBLAS
Real dotProduct(cublasHandle_t cublasHandle, const Real* a, const Real* b, Index n);

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDASPARSEOPS_H

