/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <cuda_runtime.h>
#include <sle/Types.h>
#include <sle/math/SparseMatrix.h>
#include <vector>

namespace sle {
namespace cuda {

// CUDA kernel for sparse matrix-vector product (CSR format)
__global__ void sparseMatVecKernel(const Real* values, const Index* rowPtr, 
                                    const Index* colInd, const Real* x, Real* y,
                                    Index nRows) {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < nRows) {
        Real sum = 0.0;
        Index start = rowPtr[row];
        Index end = rowPtr[row + 1];
        
        for (Index i = start; i < end; ++i) {
            Index col = colInd[i];
            sum += values[i] * x[col];
        }
        
        y[row] = sum;
    }
}

// CUDA kernel for weighted sparse matrix-vector product: y = W * A * x
__global__ void weightedSparseMatVecKernel(const Real* values, const Index* rowPtr,
                                          const Index* colInd, const Real* weights,
                                          const Real* x, Real* y, Index nRows) {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < nRows) {
        Real sum = 0.0;
        Index start = rowPtr[row];
        Index end = rowPtr[row + 1];
        
        for (Index i = start; i < end; ++i) {
            Index col = colInd[i];
            sum += values[i] * x[col];
        }
        
        y[row] = weights[row] * sum;
    }
}

// Parallel reduction for dot product
__global__ void dotProductKernel(const Real* a, const Real* b, Real* result, Index n) {
    extern __shared__ Real sdata[];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();
    
    // Reduction
    for (Index s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Wrapper functions
void sparseMatVec(const Real* values, const Index* rowPtr, const Index* colInd,
                  const Real* x, Real* y, Index nRows, Index nCols) {
    const Index blockSize = 256;
    const Index gridSize = (nRows + blockSize - 1) / blockSize;
    
    sparseMatVecKernel<<<gridSize, blockSize>>>(values, rowPtr, colInd, x, y, nRows);
}

void weightedSparseMatVec(const Real* values, const Index* rowPtr, const Index* colInd,
                          const Real* weights, const Real* x, Real* y, Index nRows) {
    const Index blockSize = 256;
    const Index gridSize = (nRows + blockSize - 1) / blockSize;
    
    weightedSparseMatVecKernel<<<gridSize, blockSize>>>(values, rowPtr, colInd, weights, x, y, nRows);
}

Real dotProduct(const Real* a, const Real* b, Index n) {
    const Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    Real* d_result;
    cudaMalloc(&d_result, gridSize * sizeof(Real));
    
    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(Real)>>>(a, b, d_result, n);
    
    // Final reduction on host (or use another kernel)
    std::vector<Real> h_result(gridSize);
    cudaMemcpy(h_result.data(), d_result, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);
    
    Real sum = 0.0;
    for (Index i = 0; i < gridSize; ++i) {
        sum += h_result[i];
    }
    
    cudaFree(d_result);
    return sum;
}

} // namespace cuda
} // namespace sle

