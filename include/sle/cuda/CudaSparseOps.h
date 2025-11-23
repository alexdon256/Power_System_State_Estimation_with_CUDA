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

#ifdef USE_CUDA
#include <cusparse.h>
#include <cublas.h>
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

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
void scaleSparseMatrixRows(cusparseHandle_t handle,
                           const Real* H_values, const Index* H_rowPtr,
                           const Real* weights, Real* H_scaled,
                           Index nRows, Index nnz);

// Compute gain matrix G = H^T * W * H on GPU using cuSPARSE SpGEMM
// Allocates G_values and G_colInd on GPU, caller must free them
// If d_WH_values is provided (pooled buffer), uses it instead of allocating
// Returns true on success, false on failure (fallback to CPU)
bool computeGainMatrixGPU(cusparseHandle_t handle,
                          const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                          const Real* weights,
                          Real*& G_values, Index* G_rowPtr, Index*& G_colInd,
                          Index nMeas, Index nStates, Index& G_nnz,
                          Real* d_WH_values = nullptr, size_t WH_valuesSize = 0);

// Compute gain matrix G using direct formula: G_ij = sum_k(H_ki * w_k * H_kj)
// Alternative to SpGEMM, may be faster for very sparse matrices
// Allocates G_values and G_colInd on GPU, caller must free them
// Returns true on success, false on failure
bool computeGainMatrixDirect(cusparseHandle_t handle,
                             const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                             const Real* weights,
                             Real*& G_values, Index* G_rowPtr, Index*& G_colInd,
                             Index nMeas, Index nStates, Index& G_nnz);

// OPTIMIZATION: Incremental gain matrix update for constant structure systems
// Updates only the parts of G that changed due to changed H values
// H_changedRows: list of row indices in H that changed (nullptr = all rows changed)
// nChangedRows: number of changed rows (0 = all rows changed)
// G_old: previous gain matrix (must have same structure as G)
// G_new: output gain matrix (must have same structure as G_old)
// Returns true on success, false on failure (falls back to full recomputation)
// Note: For best performance, use when structure is constant and only values change
bool computeGainMatrixIncremental(cusparseHandle_t handle,
                                  const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                                  const Real* weights,
                                  const Index* H_changedRows, Index nChangedRows,
                                  const Real* G_old_values, const Index* G_old_rowPtr, const Index* G_old_colInd,
                                  Real* G_new_values, const Index* G_new_rowPtr, const Index* G_new_colInd,
                                  Index nMeas, Index nStates, Index G_nnz);

// Dot product using cuBLAS
Real dotProduct(cublasHandle_t cublasHandle, const Real* a, const Real* b, Index n);

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDASPARSEOPS_H

