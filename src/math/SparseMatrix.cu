/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <cuda_runtime.h>
#include <sle/math/SparseMatrix.h>
#include <stdexcept>
#include <cusparse.h>

namespace sle {
namespace math {

SparseMatrix::SparseMatrix() 
    : d_values_(nullptr), d_rowPtr_(nullptr), d_colInd_(nullptr),
      nRows_(0), nCols_(0), nnz_(0) {
}

SparseMatrix::~SparseMatrix() {
    clear();
}

void SparseMatrix::allocateDeviceMemory() {
    cudaError_t err;
    
    if (nnz_ > 0) {
        err = cudaMalloc(&d_values_, nnz_ * sizeof(Real));
        if (err != cudaSuccess) {
            d_values_ = nullptr;
            throw std::runtime_error("Failed to allocate device memory for sparse matrix values");
        }
        
        err = cudaMalloc(&d_colInd_, nnz_ * sizeof(Index));
        if (err != cudaSuccess) {
            // Free previously allocated memory
            if (d_values_) {
                cudaFree(d_values_);
                d_values_ = nullptr;
            }
            d_colInd_ = nullptr;
            throw std::runtime_error("Failed to allocate device memory for sparse matrix column indices");
        }
    }
    
    if (nRows_ > 0) {
        err = cudaMalloc(&d_rowPtr_, (nRows_ + 1) * sizeof(Index));
        if (err != cudaSuccess) {
            // Free previously allocated memory
            freeDeviceMemory();
            throw std::runtime_error("Failed to allocate device memory for sparse matrix row pointers");
        }
    }
}

void SparseMatrix::freeDeviceMemory() {
    if (d_values_) {
        cudaFree(d_values_);
        d_values_ = nullptr;
    }
    if (d_rowPtr_) {
        cudaFree(d_rowPtr_);
        d_rowPtr_ = nullptr;
    }
    if (d_colInd_) {
        cudaFree(d_colInd_);
        d_colInd_ = nullptr;
    }
}

void SparseMatrix::buildFromCSR(const std::vector<Real>& values,
                                 const std::vector<Index>& rowPtr,
                                 const std::vector<Index>& colInd,
                                 Index nRows, Index nCols) {
    clear();
    
    nRows_ = nRows;
    nCols_ = nCols;
    nnz_ = values.size();
    
    if (nnz_ == 0 || nRows_ == 0 || nCols_ == 0) {
        return;
    }
    
    allocateDeviceMemory();
    
    // Copy data to device
    cudaError_t err;
    err = cudaMemcpy(d_values_, values.data(), nnz_ * sizeof(Real), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        freeDeviceMemory();
        throw std::runtime_error("Failed to copy sparse matrix values to device");
    }
    
    err = cudaMemcpy(d_rowPtr_, rowPtr.data(), (nRows_ + 1) * sizeof(Index), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        freeDeviceMemory();
        throw std::runtime_error("Failed to copy sparse matrix row pointers to device");
    }
    
    err = cudaMemcpy(d_colInd_, colInd.data(), nnz_ * sizeof(Index), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        freeDeviceMemory();
        throw std::runtime_error("Failed to copy sparse matrix column indices to device");
    }
}

void SparseMatrix::multiplyVector(const Real* x, Real* y, cusparseHandle_t handle) const {
    if (nnz_ == 0 || nRows_ == 0) {
        return;
    }
    
    const Real alpha = 1.0;
    const Real beta = 0.0;
    
#if CUSPARSE_VERSION >= 11000
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCsr(&mat,
                      nRows_, nCols_, nnz_,
                      d_rowPtr_, d_colInd_, d_values_,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, nCols_, const_cast<Real*>(x), CUDA_R_64F);
    cusparseCreateDnVec(&vecY, nRows_, y, CUDA_R_64F);
    
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, mat, vecX, &beta, vecY,
                            CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    if (bufferSize > 0) {
        cudaMalloc(&dBuffer, bufferSize);
    }
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, mat, vecX, &beta, vecY,
                 CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);
    if (dBuffer) cudaFree(dBuffer);
    
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(mat);
#else
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    
    cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   nRows_, nCols_, nnz_,
                   &alpha, descr,
                   d_values_, d_rowPtr_, d_colInd_,
                   x, &beta, y);
    
    cusparseDestroyMatDescr(descr);
#endif
}

void SparseMatrix::multiplyVectorTranspose(const Real* x, Real* y, cusparseHandle_t handle) const {
    if (nnz_ == 0 || nCols_ == 0) {
        return;
    }
    
    const Real alpha = 1.0;
    const Real beta = 0.0;
    
#if CUSPARSE_VERSION >= 11000
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCsr(&mat,
                      nRows_, nCols_, nnz_,
                      d_rowPtr_, d_colInd_, d_values_,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, nRows_, const_cast<Real*>(x), CUDA_R_64F);
    cusparseCreateDnVec(&vecY, nCols_, y, CUDA_R_64F);
    
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
                            &alpha, mat, vecX, &beta, vecY,
                            CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    if (bufferSize > 0) {
        cudaMalloc(&dBuffer, bufferSize);
    }
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
                 &alpha, mat, vecX, &beta, vecY,
                 CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);
    if (dBuffer) cudaFree(dBuffer);
    
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(mat);
#else
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    
    cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE,
                   nRows_, nCols_, nnz_,
                   &alpha, descr,
                   d_values_, d_rowPtr_, d_colInd_,
                   x, &beta, y);
    
    cusparseDestroyMatDescr(descr);
#endif
}

void SparseMatrix::clear() {
    freeDeviceMemory();
    nRows_ = 0;
    nCols_ = 0;
    nnz_ = 0;
}

} // namespace math
} // namespace sle

