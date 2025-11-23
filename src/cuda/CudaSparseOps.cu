/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * CUDA sparse matrix operations using cuSPARSE
 */

#include <sle/cuda/CudaSparseOps.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas.h>
#include <stdexcept>
#include <vector>
#include <cmath>

namespace sle {
namespace cuda {

// Sparse matrix-vector product: y = A * x
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

// GPU kernel to scale sparse matrix rows by weights
__global__ void scaleSparseMatrixRowsKernel(const Real* H_values, const Index* H_rowPtr,
                                            const Real* weights, Real* H_scaled,
                                            Index nRows) {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nRows) {
        Index rowStart = H_rowPtr[row];
        Index rowEnd = H_rowPtr[row + 1];
        Real w = weights[row];
        
        for (Index idx = rowStart; idx < rowEnd; ++idx) {
            H_scaled[idx] = w * H_values[idx];
        }
    }
}

// Scale sparse matrix rows by weights: H_scaled = W * H (element-wise row scaling)
void scaleSparseMatrixRows(cusparseHandle_t handle,
                           const Real* H_values, const Index* H_rowPtr,
                           const Real* weights, Real* H_scaled,
                           Index nRows, Index nnz,
                           cudaStream_t stream) {
    constexpr Index blockSize = 256;
    const Index gridSize = (nRows + blockSize - 1) / blockSize;
    // OPTIMIZATION: Use stream for asynchronous execution
    scaleSparseMatrixRowsKernel<<<gridSize, blockSize, 0, stream>>>(
        H_values, H_rowPtr, weights, H_scaled, nRows);
}

// Compute gain matrix G = H^T * W * H on GPU using cuSPARSE SpGEMM
// Strategy: First compute W*H, then compute H^T*(W*H)
// If d_WH_values is provided (pooled buffer), uses it instead of allocating
// Returns true on success, false on failure (fallback to CPU)
bool computeGainMatrixGPU(cusparseHandle_t handle,
                          const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                          const Real* weights,
                          Real*& G_values, Index* G_rowPtr, Index*& G_colInd,
                          Index nMeas, Index nStates, Index& G_nnz,
                          Real* d_WH_values, size_t WH_valuesSize) {
    if (nMeas == 0 || nStates == 0) {
        G_nnz = 0;
        cudaMemset(G_rowPtr, 0, (nStates + 1) * sizeof(Index));
        return true;
    }
    
    Index H_nnz = H_rowPtr[nMeas] - H_rowPtr[0];
    if (H_nnz == 0) {
        G_nnz = 0;
        cudaMemset(G_rowPtr, 0, (nStates + 1) * sizeof(Index));
        return true;
    }
    
    // Step 1: Scale H by weights: WH = W * H (row-wise scaling)
    // OPTIMIZATION: Use pooled buffer if provided, otherwise allocate
    Real* WH_values = nullptr;
    bool usePooledBuffer = (d_WH_values != nullptr && static_cast<size_t>(H_nnz) <= WH_valuesSize);
    cudaError_t err;
    
    if (usePooledBuffer) {
        WH_values = d_WH_values;  // Use pooled buffer
    } else {
        err = cudaMalloc(&WH_values, H_nnz * sizeof(Real));
        if (err != cudaSuccess) {
            return false;
        }
    }
    scaleSparseMatrixRows(handle, H_values, H_rowPtr, weights, WH_values, nMeas, H_nnz);
    
    // Step 2: Create cuSPARSE descriptors for H and WH
    cusparseSpMatDescr_t matH, matWH, matG;
    cusparseStatus_t status;
    
    // H matrix descriptor (nMeas x nStates) - will be transposed
    status = cusparseCreateCsr(&matH, nMeas, nStates, H_nnz,
                                const_cast<Index*>(H_rowPtr), const_cast<Index*>(H_colInd),
                                const_cast<Real*>(H_values),
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        if (WH_values && !usePooledBuffer) cudaFree(WH_values);
        return false;
    }
    
    // WH matrix descriptor (nMeas x nStates)
    status = cusparseCreateCsr(&matWH, nMeas, nStates, H_nnz,
                                const_cast<Index*>(H_rowPtr), const_cast<Index*>(H_colInd),
                                WH_values,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroySpMat(matH);
        if (WH_values && !usePooledBuffer) cudaFree(WH_values);
        return false;
    }
    
    // G matrix descriptor (nStates x nStates) - output (initially empty)
    Index* G_colInd_temp = nullptr;
    Real* G_values_temp = nullptr;
    status = cusparseCreateCsr(&matG, nStates, nStates, 0,
                                G_rowPtr, G_colInd_temp, G_values_temp,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroySpMat(matH);
        cusparseDestroySpMat(matWH);
        if (WH_values && !usePooledBuffer) cudaFree(WH_values);
        return false;
    }
    
    // Step 3: Compute G = H^T * WH using SpGEMM
    const Real alpha = 1.0;
    const Real beta = 0.0;
    
    cusparseSpGEMMDescr_t spgemmDesc;
    status = cusparseSpGEMM_createDescr(&spgemmDesc);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroySpMat(matH);
        cusparseDestroySpMat(matWH);
        cusparseDestroySpMat(matG);
        if (WH_values && !usePooledBuffer) cudaFree(WH_values);
        return false;
    }
    
    // Phase 1: Work estimation
    size_t bufferSize1 = 0;
    void* dBuffer1 = nullptr;
    status = cusparseSpGEMM_workEstimation(handle,
                                           CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matH, matWH, &beta, matG,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize1, dBuffer1);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseSpGEMM_destroyDescr(spgemmDesc);
        cusparseDestroySpMat(matH);
        cusparseDestroySpMat(matWH);
        cusparseDestroySpMat(matG);
        if (WH_values && !usePooledBuffer) cudaFree(WH_values);
        return false;
    }
    
    if (bufferSize1 > 0) {
        err = cudaMalloc(&dBuffer1, bufferSize1);
        if (err != cudaSuccess) {
            cusparseSpGEMM_destroyDescr(spgemmDesc);
            cusparseDestroySpMat(matH);
            cusparseDestroySpMat(matWH);
            cusparseDestroySpMat(matG);
            if (WH_values && !usePooledBuffer) cudaFree(WH_values);
            return false;
        }
        status = cusparseSpGEMM_workEstimation(handle,
                                               CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matH, matWH, &beta, matG,
                                               CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize1, dBuffer1);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cudaFree(dBuffer1);
            cusparseSpGEMM_destroyDescr(spgemmDesc);
            cusparseDestroySpMat(matH);
            cusparseDestroySpMat(matWH);
            cusparseDestroySpMat(matG);
            if (WH_values && !usePooledBuffer) cudaFree(WH_values);
            return false;
        }
    }
    
    // Phase 2: Compute structure
    size_t bufferSize2 = 0;
    void* dBuffer2 = nullptr;
    status = cusparseSpGEMM_compute(handle,
                                    CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matH, matWH, &beta, matG,
                                    CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize2, dBuffer2);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        if (dBuffer1) cudaFree(dBuffer1);
        cusparseSpGEMM_destroyDescr(spgemmDesc);
        cusparseDestroySpMat(matH);
        cusparseDestroySpMat(matWH);
        cusparseDestroySpMat(matG);
        if (WH_values && !usePooledBuffer) cudaFree(WH_values);
        return false;
    }
    
    if (bufferSize2 > 0) {
        err = cudaMalloc(&dBuffer2, bufferSize2);
        if (err != cudaSuccess) {
            if (dBuffer1) cudaFree(dBuffer1);
            cusparseSpGEMM_destroyDescr(spgemmDesc);
            cusparseDestroySpMat(matH);
            cusparseDestroySpMat(matWH);
            cusparseDestroySpMat(matG);
            if (WH_values && !usePooledBuffer) cudaFree(WH_values);
            return false;
        }
        status = cusparseSpGEMM_compute(handle,
                                        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matH, matWH, &beta, matG,
                                        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize2, dBuffer2);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            if (dBuffer1) cudaFree(dBuffer1);
            if (dBuffer2) cudaFree(dBuffer2);
            cusparseSpGEMM_destroyDescr(spgemmDesc);
            cusparseDestroySpMat(matH);
            cusparseDestroySpMat(matWH);
            cusparseDestroySpMat(matG);
            if (WH_values && !usePooledBuffer) cudaFree(WH_values);
            return false;
        }
    }
    
    // Phase 3: Copy results
    // First get the structure size
    int64_t G_rows, G_cols, G_nnz_int64;
    cusparseSpMatGetSize(matG, &G_rows, &G_cols, &G_nnz_int64);
    G_nnz = static_cast<Index>(G_nnz_int64);
    
    if (G_nnz > 0) {
        // Allocate output buffers
        err = cudaMalloc(&G_values, G_nnz * sizeof(Real));
        if (err != cudaSuccess) {
            if (dBuffer1) cudaFree(dBuffer1);
            if (dBuffer2) cudaFree(dBuffer2);
            cusparseSpGEMM_destroyDescr(spgemmDesc);
            cusparseDestroySpMat(matH);
            cusparseDestroySpMat(matWH);
            cusparseDestroySpMat(matG);
            if (WH_values && !usePooledBuffer) cudaFree(WH_values);
            return false;
        }
        
        err = cudaMalloc(&G_colInd, G_nnz * sizeof(Index));
        if (err != cudaSuccess) {
            cudaFree(G_values);
            if (dBuffer1) cudaFree(dBuffer1);
            if (dBuffer2) cudaFree(dBuffer2);
            cusparseSpGEMM_destroyDescr(spgemmDesc);
            cusparseDestroySpMat(matH);
            cusparseDestroySpMat(matWH);
            cusparseDestroySpMat(matG);
            if (WH_values && !usePooledBuffer) cudaFree(WH_values);
            return false;
        }
        
        // Extract CSR data
        size_t rowPtrSize, colIndSize, valuesSize;
        cusparseCsrGet(G_rowPtr, &rowPtrSize, G_colInd, &colIndSize, G_values, &valuesSize, matG);
    }
    
    // Cleanup
    if (dBuffer1) cudaFree(dBuffer1);
    if (dBuffer2) cudaFree(dBuffer2);
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matH);
    cusparseDestroySpMat(matWH);
    cusparseDestroySpMat(matG);
    // OPTIMIZATION: Only free WH_values if we allocated it (not if using pooled buffer)
    if (WH_values && !usePooledBuffer) {
        cudaFree(WH_values);
    }
    
    return true;
}

// GPU kernel: Compute G_ij = sum_k(H_ki * w_k * H_kj)
// Each thread processes one row k of H and accumulates contributions to G
// Uses atomicAdd for thread-safe accumulation (acceptable for sparse matrices)
__global__ void computeGainMatrixDirectKernel(const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                                             const Real* weights,
                                             Real* G_dense, Index nMeas, Index nStates) {
    Index k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k >= nMeas) return;
    
    Real w_k = weights[k];
    Index rowStart = H_rowPtr[k];
    Index rowEnd = H_rowPtr[k + 1];
    Index nnz_in_row = rowEnd - rowStart;
    
    // For each pair of non-zero elements (i, j) in row k, accumulate to G[i][j]
    // This is O(nnz_per_row^2) but typically nnz_per_row is small for sparse matrices
    for (Index idx_i = 0; idx_i < nnz_in_row; ++idx_i) {
        Index global_idx_i = rowStart + idx_i;
        if (global_idx_i >= rowEnd) break;
        
        Index i = H_colInd[global_idx_i];
        Real H_ki = H_values[global_idx_i];
        
        if (i < 0 || i >= nStates) continue;
        
        for (Index idx_j = 0; idx_j < nnz_in_row; ++idx_j) {
            Index global_idx_j = rowStart + idx_j;
            if (global_idx_j >= rowEnd) break;
            
            Index j = H_colInd[global_idx_j];
            Real H_kj = H_values[global_idx_j];
            
            if (j < 0 || j >= nStates) continue;
            
            // Accumulate: G[i][j] += H[k][i] * w[k] * H[k][j]
            Real contribution = H_ki * w_k * H_kj;
            atomicAdd(&G_dense[i * nStates + j], contribution);
        }
    }
}

// Alternative: Compute G_ij for each (i,j) pair by summing over k
// Each thread computes one element G[i][j]
__global__ void computeGainMatrixElementKernel(const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                                                const Real* weights,
                                                Real* G_dense, Index nMeas, Index nStates) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    Index i = idx / nStates;
    Index j = idx % nStates;
    
    if (i >= nStates || j >= nStates) return;
    
    Real sum = 0.0;
    
    // Sum over all rows k where both H[k][i] and H[k][j] are non-zero
    for (Index k = 0; k < nMeas; ++k) {
        Index rowStart = H_rowPtr[k];
        Index rowEnd = H_rowPtr[k + 1];
        
        Real H_ki = 0.0;
        Real H_kj = 0.0;
        
        // Find H[k][i] and H[k][j] in row k
        for (Index idx_k = rowStart; idx_k < rowEnd; ++idx_k) {
            Index col = H_colInd[idx_k];
            if (col == i) {
                H_ki = H_values[idx_k];
            }
            if (col == j) {
                H_kj = H_values[idx_k];
            }
        }
        
        // Accumulate if both are non-zero
        if (H_ki != 0.0 && H_kj != 0.0) {
            sum += H_ki * weights[k] * H_kj;
        }
    }
    
    G_dense[i * nStates + j] = sum;
}

// Compute gain matrix G = H^T * W * H directly on GPU using formula G_ij = sum_k(H_ki * w_k * H_kj)
// This is an alternative to SpGEMM that may be faster for very sparse matrices
// Returns true on success, false on failure
bool computeGainMatrixDirectGPU(const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                                const Real* weights,
                                Real*& G_dense, Index nMeas, Index nStates) {
    if (nMeas == 0 || nStates == 0) {
        return false;
    }
    
    Index H_nnz = H_rowPtr[nMeas] - H_rowPtr[0];
    if (H_nnz == 0) {
        return false;
    }
    
    // Allocate dense matrix on GPU (nStates x nStates)
    cudaError_t err = cudaMalloc(&G_dense, nStates * nStates * sizeof(Real));
    if (err != cudaSuccess) {
        return false;
    }
    
    // Initialize to zero
    cudaMemset(G_dense, 0, nStates * nStates * sizeof(Real));
    
    // Choose kernel based on sparsity
    // For sparse matrices, use row-based kernel (each thread processes one row k)
    // For denser matrices, use element-based kernel (each thread computes one G[i][j])
    constexpr Index blockSize = 256;
    
    // Use row-based kernel (more efficient for sparse H)
    Index gridSize = (nMeas + blockSize - 1) / blockSize;
    computeGainMatrixDirectKernel<<<gridSize, blockSize>>>(
        H_values, H_rowPtr, H_colInd, weights, G_dense, nMeas, nStates);
    
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        cudaFree(G_dense);
        return false;
    }
    
    // Note: No synchronization needed - caller will sync if results are needed immediately
    // This allows overlapping with other operations
    
    return true;
}

// Count non-zeros per row
__global__ void countNonZerosKernel(const Real* G_dense, Index* rowCounts, 
                                    Index nStates, Real threshold) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= nStates) return;
    
    Index count = 0;
    for (Index j = 0; j < nStates; ++j) {
        Real val = G_dense[i * nStates + j];
        if (fabs(val) > threshold) {
            count++;
        }
    }
    rowCounts[i] = count;
}

// Extract non-zeros from dense matrix to sparse CSR format
__global__ void extractNonZerosKernel(const Real* G_dense, Real* G_values, Index* G_colInd,
                                      const Index* G_rowPtr, Index nStates, Real threshold) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= nStates) return;
    
    Index rowStart = G_rowPtr[i];
    Index count = 0;
    
    for (Index j = 0; j < nStates; ++j) {
        Real val = G_dense[i * nStates + j];
        if (fabs(val) > threshold) {
            Index idx = rowStart + count;
            G_colInd[idx] = j;
            G_values[idx] = val;
            count++;
        }
    }
}

// Compute gain matrix using direct formula on GPU, then convert to sparse
// Returns true on success, false on failure
bool computeGainMatrixDirect(cusparseHandle_t handle,
                             const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                             const Real* weights,
                             Real*& G_values, Index* G_rowPtr, Index*& G_colInd,
                             Index nMeas, Index nStates, Index& G_nnz) {
    if (nMeas == 0 || nStates == 0) {
        G_nnz = 0;
        cudaMemset(G_rowPtr, 0, (nStates + 1) * sizeof(Index));
        return true;
    }
    
    // Step 1: Compute dense G on GPU
    Real* G_dense = nullptr;
    bool success = computeGainMatrixDirectGPU(H_values, H_rowPtr, H_colInd, weights,
                                             G_dense, nMeas, nStates);
    if (!success) {
        return false;
    }
    
    // Step 2: Count non-zeros per row (first pass)
    Index* rowCounts = nullptr;
    cudaError_t err = cudaMalloc(&rowCounts, nStates * sizeof(Index));
    if (err != cudaSuccess) {
        cudaFree(G_dense);
        return false;
    }
    
    constexpr Index blockSize = 256;
    Index gridSize = (nStates + blockSize - 1) / blockSize;
    const Real threshold = 1e-12;
    
    // Count non-zeros per row
    countNonZerosKernel<<<gridSize, blockSize>>>(G_dense, rowCounts, nStates, threshold);
    
    // Step 3: Build row pointer (exclusive scan on host for simplicity)
    std::vector<Index> rowCounts_host(nStates);
    cudaMemcpy(rowCounts_host.data(), rowCounts, nStates * sizeof(Index), cudaMemcpyDeviceToHost);
    cudaFree(rowCounts);
    
    std::vector<Index> G_rowPtr_host(nStates + 1);
    G_rowPtr_host[0] = 0;
    for (Index i = 0; i < nStates; ++i) {
        G_rowPtr_host[i + 1] = G_rowPtr_host[i] + rowCounts_host[i];
    }
    G_nnz = G_rowPtr_host[nStates];
    
    // Step 4: Allocate row pointer and output arrays
    err = cudaMalloc(&G_rowPtr, (nStates + 1) * sizeof(Index));
    if (err != cudaSuccess) {
        cudaFree(G_dense);
        return false;
    }
    
    // Copy row pointer to device
    cudaMemcpy(G_rowPtr, G_rowPtr_host.data(), (nStates + 1) * sizeof(Index), cudaMemcpyHostToDevice);
    
    // Step 5: Allocate output arrays and extract non-zeros
    if (G_nnz > 0) {
        err = cudaMalloc(&G_values, G_nnz * sizeof(Real));
        if (err != cudaSuccess) {
            cudaFree(G_dense);
            cudaFree(G_rowPtr);
            return false;
        }
        
        err = cudaMalloc(&G_colInd, G_nnz * sizeof(Index));
        if (err != cudaSuccess) {
            cudaFree(G_dense);
            cudaFree(G_rowPtr);
            cudaFree(G_values);
            return false;
        }
        
        // Extract non-zeros from dense matrix
        extractNonZerosKernel<<<gridSize, blockSize>>>(
            G_dense, G_values, G_colInd, G_rowPtr, nStates, threshold);
    } else {
        cudaMemset(G_rowPtr, 0, (nStates + 1) * sizeof(Index));
    }
    
    cudaFree(G_dense);
    return true;
}

// Dot product using cuBLAS for optimal performance
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

// OPTIMIZATION: Incremental gain matrix update for constant structure systems
// Updates only the parts of G that changed due to changed H values
// This is more efficient than full recomputation when only a few rows of H changed
__global__ void updateGainMatrixIncrementalKernel(
    const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
    const Real* weights,
    const Index* H_changedRows, Index nChangedRows,
    const Real* G_old_values,
    Real* G_new_values,
    const Index* G_rowPtr, const Index* G_colInd,
    Index nMeas, Index nStates, Index G_nnz) {
    
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < G_nnz) {
        // Find which row and column this element belongs to
        // G is symmetric, stored in CSR format
        Index row = 0;
        for (Index i = 0; i < nStates; ++i) {
            if (idx >= G_rowPtr[i] && idx < G_rowPtr[i + 1]) {
                row = i;
                break;
            }
        }
        Index col = G_colInd[idx];
        
        // Initialize with old value
        Real gValue = G_old_values[idx];
        
        // Update contributions from changed rows
        // G_ij = sum_k(H_ki * w_k * H_kj)
        // Only update contributions from changed rows k
        for (Index kIdx = 0; kIdx < nChangedRows; ++kIdx) {
            Index k = H_changedRows[kIdx];
            if (k < 0 || k >= nMeas) continue;
            
            // Find H_ki and H_kj
            Real h_ki = 0.0;
            Real h_kj = 0.0;
            
            Index rowStart = H_rowPtr[k];
            Index rowEnd = H_rowPtr[k + 1];
            
            for (Index i = rowStart; i < rowEnd; ++i) {
                Index colIdx = H_colInd[i];
                if (colIdx == row) {
                    h_ki = H_values[i];
                }
                if (colIdx == col) {
                    h_kj = H_values[i];
                }
            }
            
            // Update: subtract old contribution, add new contribution
            // For incremental update, we need to know old H values
            // This is a simplified version - full implementation would track old H values
            Real weight = weights[k];
            Real contribution = h_ki * weight * h_kj;
            
            // For diagonal elements (row == col), contribution is doubled in symmetric matrix
            if (row == col) {
                gValue += contribution;
            } else {
                // For off-diagonal, need to update both (i,j) and (j,i) if symmetric
                gValue += contribution;
            }
        }
        
        G_new_values[idx] = gValue;
    }
}

// OPTIMIZATION: Incremental gain matrix update
// For systems with constant structure, only updates changed parts
bool computeGainMatrixIncremental(cusparseHandle_t handle,
                                  const Real* H_values, const Index* H_rowPtr, const Index* H_colInd,
                                  const Real* weights,
                                  const Index* H_changedRows, Index nChangedRows,
                                  const Real* G_old_values, const Index* G_old_rowPtr, const Index* G_old_colInd,
                                  Real* G_new_values, const Index* G_new_rowPtr, const Index* G_new_colInd,
                                  Index nMeas, Index nStates, Index G_nnz) {
    // Validate inputs
    if (nMeas == 0 || nStates == 0 || G_nnz == 0) {
        return false;
    }
    
    // If all rows changed or no tracking, fall back to full recomputation
    if (H_changedRows == nullptr || nChangedRows == 0 || nChangedRows >= nMeas) {
        return false;  // Signal to caller to use full recomputation
    }
    
    // Verify structure matches
    bool structureMatches = true;
    for (Index i = 0; i <= nStates; ++i) {
        if (G_old_rowPtr[i] != G_new_rowPtr[i]) {
            structureMatches = false;
            break;
        }
    }
    if (!structureMatches) {
        return false;  // Structure changed, need full recomputation
    }
    
    // Copy old values to new (will update incrementally)
    cudaMemcpy(G_new_values, G_old_values, G_nnz * sizeof(Real), cudaMemcpyDeviceToDevice);
    
    // Launch kernel to update only changed contributions
    constexpr Index blockSize = 256;
    Index gridSize = (G_nnz + blockSize - 1) / blockSize;
    
    // Note: This is a simplified incremental update
    // Full implementation would require tracking old H values to compute delta
    // For now, this serves as a framework - actual implementation would need:
    // 1. Storage for old H values
    // 2. Delta computation: delta_G = H_new^T * W * H_new - H_old^T * W * H_old
    // 3. Update: G_new = G_old + delta_G
    
    // For now, return false to indicate full recomputation is needed
    // This function provides the interface for future optimization
    return false;
}

} // namespace cuda
} // namespace sle
