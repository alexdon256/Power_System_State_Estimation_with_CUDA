/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_SPARSEMATRIX_H
#define SLE_MATH_SPARSEMATRIX_H

#include <sle/Types.h>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#else
using cusparseHandle_t = void*;
#endif

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
    void multiplyVector(const Real* x, Real* y, cusparseHandle_t handle) const;
    
    // Matrix transpose-vector product: y = A^T * x
    void multiplyVectorTranspose(const Real* x, Real* y, cusparseHandle_t handle) const;
    
    void clear();
    
private:
    Real* d_values_;
    Index* d_rowPtr_;
    Index* d_colInd_;
    Index nRows_;
    Index nCols_;
    Index nnz_;
    
    void allocateDeviceMemory();
    void freeDeviceMemory();
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_SPARSEMATRIX_H

