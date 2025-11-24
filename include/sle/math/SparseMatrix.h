/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_SPARSEMATRIX_H
#define SLE_MATH_SPARSEMATRIX_H

#include <sle/Types.h>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>

namespace sle {
namespace math {

// CSR format sparse matrix
class SparseMatrix {
public:
    SparseMatrix();
    ~SparseMatrix();
    
    // Build from host data (CSR format)
    void buildFromCSR(const std::vector<Real>& values,
                      const std::vector<Index>& rowPtr,
                      const std::vector<Index>& colInd,
                      Index nRows, Index nCols);
    
    // Build from device pointers (takes ownership, avoids host-device copy)
    // WARNING: Do not free the pointers after calling this - SparseMatrix owns them
    void buildFromDevicePointers(Real* d_values, Index* d_rowPtr, Index* d_colInd,
                                 Index nRows, Index nCols, Index nnz);
    
    // Get device pointers
    Real* getValues() { return d_values_; }
    Index* getRowPtr() { return d_rowPtr_; }
    Index* getColInd() { return d_colInd_; }
    
    const Real* getValues() const { return d_values_; }
    const Index* getRowPtr() const { return d_rowPtr_; }
    const Index* getColInd() const { return d_colInd_; }
    
    Index getNRows() const { return nRows_; }
    Index getNCols() const { return nCols_; }
    Index getNNZ() const { return nnz_; }
    
    // Matrix-vector product: y = A * x
    // Optional buffer: If provided, reuses existing buffer (avoids allocation)
    void multiplyVector(const Real* x, Real* y, cusparseHandle_t handle, 
                       void* dBuffer = nullptr, size_t* bufferSize = nullptr) const;
    
    // Matrix transpose-vector product: y = A^T * x
    // Optional buffer: If provided, reuses existing buffer (avoids allocation)
    void multiplyVectorTranspose(const Real* x, Real* y, cusparseHandle_t handle,
                                void* dBuffer = nullptr, size_t* bufferSize = nullptr) const;
    
    void clear();
    
private:
    Real* d_values_;
    Index* d_rowPtr_;
    Index* d_colInd_;
    Index nRows_;
    Index nCols_;
    Index nnz_;
    
    // OPTIMIZATION: Cached cuSPARSE descriptors (for new API >= 11.0)
    // These are created once and reused, avoiding per-call overhead
#if CUSPARSE_VERSION >= 11000
    mutable cusparseSpMatDescr_t cachedSpMatDescr_ = nullptr;
    mutable size_t cachedSpMVBufferSize_ = 0;
    mutable void* cachedSpMVBuffer_ = nullptr;
#endif
    
    void allocateDeviceMemory();
    void freeDeviceMemory();
    void ensureSpMVBuffer(size_t requiredSize) const;
    void freeCachedDescriptors() const;
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_SPARSEMATRIX_H

