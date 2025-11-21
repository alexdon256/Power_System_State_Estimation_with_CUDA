/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <cuda_runtime.h>
#include <sle/Types.h>
#include <vector>

// Precision-aware intrinsics: automatically detect Real type (double or float)
// Real is defined as double in Types.h, so use double precision intrinsics
//
// Note: __CUDACC__ is defined when nvcc compiles (both host and device code)
//       __CUDA_ARCH__ is only defined in device code (__device__/__global__ functions)
//       For IDE parsing, neither may be defined, so we fall back to regular operations
#if defined(__CUDA_ARCH__)
    // Device code: __CUDA_ARCH__ is defined
    #if __CUDA_ARCH__ >= 600
        // Compute capability 6.0+ supports double precision FMA
        // Since Real = double, use double precision intrinsics
        #define CUDA_FMA(a, b, c) __fma_rn((a), (b), (c))
    #else
        // Older architectures: use regular operations
        #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #endif
#elif defined(__CUDACC__)
    // Host code in .cu file: __CUDACC__ defined but __CUDA_ARCH__ not defined
    // Use regular operations (host code doesn't need intrinsics)
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
#else
    // IDE parsing or regular C++ compiler: use regular operations
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
#endif

namespace sle {
namespace cuda {

// Optimized GPU kernel for weighted residual using FMA
__global__ void weightedResidualKernel(const Real* x, const Real* w, Real* y, Index n) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread for better memory bandwidth utilization
    const Index stride = blockDim.x * gridDim.x;
    for (Index i = idx; i < n; i += stride) {
        // Use FMA: y[i] = w[i] * x[i] + 0
        y[i] = CUDA_FMA(w[i], x[i], 0.0);
    }
}

// Optimized GPU kernel for weighted sum of squares using warp shuffles
__global__ void weightedSumSquaresKernel(const Real* x, const Real* w, Real* partial, Index n) {
    extern __shared__ Real sdata[];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute local value using FMA
    Real val = (i < n) ? CUDA_FMA(w[i], CUDA_FMA(x[i], x[i], 0.0), 0.0) : 0.0;
    
    // Warp-level reduction using shuffle (faster than shared memory)
    const unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    
    // Write warp sum to shared memory (one per warp)
    const Index warpId = tid / warpSize;
    const Index laneId = tid % warpSize;
    if (laneId == 0) {
        sdata[warpId] = val;
    }
    __syncthreads();
    
    // Final reduction across warps in block (using first warp only)
    const Index numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (warpId == 0 && laneId < numWarps) {
        Real sum = sdata[laneId];
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            Real temp = __shfl_down_sync(mask, sum, offset);
            if (laneId + offset < numWarps) {
                sum += temp;
            }
        }
        if (laneId == 0) {
            partial[blockIdx.x] = sum;
        }
    }
}

// Optimized GPU kernel for element-wise multiply: y[i] = alpha * x[i] * w[i] + y[i]
__global__ void weightedAccumulateKernel(Real alpha, const Real* x, const Real* w, Real* y, Index n) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread for better memory bandwidth
    const Index stride = blockDim.x * gridDim.x;
    for (Index i = idx; i < n; i += stride) {
        // Use FMA: y[i] = alpha * (x[i] * w[i]) + y[i]
        const Real xw = CUDA_FMA(x[i], w[i], 0.0);
        y[i] = CUDA_FMA(alpha, xw, y[i]);
    }
}

// Wrapper functions (optimized with constexpr)
void computeWeightedResidual(const Real* residual, const Real* weights, Real* weightedResidual, Index n) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    weightedResidualKernel<<<gridSize, blockSize>>>(residual, weights, weightedResidual, n);
    // Note: No cudaDeviceSynchronize() - caller should sync if needed
    // This allows overlapping with other operations
}

Real computeWeightedSumSquares(const Real* x, const Real* w, Index n) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    if (gridSize == 0) return 0.0;
    
    Real* d_partial = nullptr;
    cudaError_t err = cudaMalloc(&d_partial, gridSize * sizeof(Real));
    if (err != cudaSuccess) {
        return 0.0;  // Return 0 if allocation fails
    }
    
    // Calculate shared memory size (one Real per warp)
    constexpr Index warpsPerBlock = (blockSize + 32 - 1) / 32;
    const Index sharedMemSize = warpsPerBlock * sizeof(Real);
    
    weightedSumSquaresKernel<<<gridSize, blockSize, sharedMemSize>>>(
        x, w, d_partial, n);
    
    // Final reduction on host (vectorized)
    std::vector<Real> h_partial;
    h_partial.reserve(gridSize);
    h_partial.resize(gridSize);
    cudaMemcpy(h_partial.data(), d_partial, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);
    
    Real sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (Index i = 0; i < gridSize; ++i) {
        sum += h_partial[i];
    }
    
    cudaFree(d_partial);
    return sum;
}

void weightedAccumulate(Real alpha, const Real* x, const Real* w, Real* y, Index n) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    weightedAccumulateKernel<<<gridSize, blockSize>>>(alpha, x, w, y, n);
    // Note: No cudaDeviceSynchronize() - caller should sync if needed
}

} // namespace cuda
} // namespace sle

