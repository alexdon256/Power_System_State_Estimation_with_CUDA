/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * CUDA kernels for solver operations: residual, state update, norm computation, weighted operations
 */

#include <cuda_runtime.h>
#include <sle/Types.h>
#include <sle/cuda/CudaReduction.h>
#include <cmath>

namespace sle {
namespace cuda {

// Precision-aware CUDA intrinsics (embedded from CudaPrecisionHelpers.h)
#if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= 600
        #define CUDA_FMA(a, b, c) __fma_rn((a), (b), (c))
    #else
        #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #endif
#else
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
#endif

// GPU kernel for residual computation: r = z - hx
__global__ void computeResidualKernel(const Real* z, const Real* hx, Real* residual, Index n) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread for better memory bandwidth
    const Index stride = blockDim.x * gridDim.x;
    for (Index i = idx; i < n; i += stride) {
        residual[i] = z[i] - hx[i];
    }
}

// Fused kernel: Compute residual and weighted residual in one pass
// r = z - hx, then wr = weights * r
__global__ void computeResidualAndWeightedKernel(const Real* z, const Real* hx, 
                                                 const Real* weights,
                                                 Real* residual, Real* weightedResidual, 
                                                 Index n) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const Index stride = blockDim.x * gridDim.x;
    for (Index i = idx; i < n; i += stride) {
        Real r = z[i] - hx[i];
        residual[i] = r;
        weightedResidual[i] = CUDA_FMA(weights[i], r, 0.0);
    }
}

// GPU kernel for state update: x_new = x_old + damping * deltaX
__global__ void updateStateKernel(const Real* x_old, const Real* deltaX, Real* x_new, 
                                  Real damping, Index n) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const Index stride = blockDim.x * gridDim.x;
    for (Index i = idx; i < n; i += stride) {
        // x_new[i] = x_old[i] + damping * deltaX[i]
        x_new[i] = CUDA_FMA(damping, deltaX[i], x_old[i]);
    }
}

// Fused kernel: Update state and compute norm squared in one pass
// x_new = x_old + damping * deltaX, and accumulate ||deltaX||^2
__global__ void updateStateAndNormKernel(const Real* x_old, const Real* deltaX, Real* x_new,
                                        Real damping, Real* partialNorm, Index n) {
    extern __shared__ Real sdata[];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update state and compute local norm contribution
    Real deltaX_val = (i < n) ? deltaX[i] : 0.0;
    Real normContrib = CUDA_FMA(deltaX_val, deltaX_val, 0.0);
    
    if (i < n) {
        x_new[i] = CUDA_FMA(damping, deltaX_val, x_old[i]);
    }
    
    // Warp-level reduction for norm
    const unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        normContrib += __shfl_down_sync(mask, normContrib, offset);
    }
    
    const Index warpId = tid / warpSize;
    const Index laneId = tid % warpSize;
    if (laneId == 0) {
        sdata[warpId] = normContrib;
    }
    __syncthreads();
    
    // Final reduction across warps
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
            partialNorm[blockIdx.x] = sum;
        }
    }
}

// GPU kernel for norm squared computation (for convergence check)
__global__ void normSquaredKernel(const Real* x, Real* partial, Index n) {
    extern __shared__ Real sdata[];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute local value
    Real val = (i < n) ? CUDA_FMA(x[i], x[i], 0.0) : 0.0;
    
    // Warp-level reduction using shuffle
    const unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    
    // Write warp sum to shared memory
    const Index warpId = tid / warpSize;
    const Index laneId = tid % warpSize;
    if (laneId == 0) {
        sdata[warpId] = val;
    }
    __syncthreads();
    
    // Final reduction across warps
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

// Wrapper functions
void computeResidual(const Real* z, const Real* hx, Real* residual, Index n) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    computeResidualKernel<<<gridSize, blockSize>>>(z, hx, residual, n);
}

// Fused: Compute residual and weighted residual in one kernel launch
void computeResidualAndWeighted(const Real* z, const Real* hx, const Real* weights,
                                Real* residual, Real* weightedResidual, Index n,
                                cudaStream_t stream) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    // OPTIMIZATION: Use stream for asynchronous execution and overlapping
    computeResidualAndWeightedKernel<<<gridSize, blockSize, 0, stream>>>(
        z, hx, weights, residual, weightedResidual, n);
}

void updateState(const Real* x_old, const Real* deltaX, Real* x_new, Real damping, Index n) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    updateStateKernel<<<gridSize, blockSize>>>(x_old, deltaX, x_new, damping, n);
}

Real computeNormSquared(const Real* x, Index n, cudaStream_t stream) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    if (gridSize == 0) return 0.0;
    
    Real* d_partial = nullptr;
    cudaError_t err = cudaMalloc(&d_partial, gridSize * sizeof(Real));
    if (err != cudaSuccess) {
        return 0.0;
    }
    
    Real result = computeNormSquared(x, n, d_partial, gridSize, stream);
    cudaFree(d_partial);
    return result;
}

Real computeNormSquared(const Real* x, Index n, Real* d_partial, size_t partialSize) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    if (gridSize == 0) return 0.0;
    if (static_cast<size_t>(gridSize) > partialSize) {
        return 0.0;  // Buffer too small
    }
    
    constexpr Index warpsPerBlock = (blockSize + 32 - 1) / 32;
    const Index sharedMemSize = warpsPerBlock * sizeof(Real);
    
    // OPTIMIZATION: Use stream for asynchronous execution
    normSquaredKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(x, d_partial, n);
    
    // OPTIMIZATION: Use GPU reduction instead of host-side reduction
    // Eliminates host-device transfer for small arrays
    return reducePartialSumsGPU(d_partial, gridSize, stream);
}

// Fused: Update state and compute norm squared in one kernel launch
Real updateStateAndComputeNorm(const Real* x_old, const Real* deltaX, Real* x_new,
                              Real damping, Index n, cudaStream_t stream) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    if (gridSize == 0) return 0.0;
    
    Real* d_partial = nullptr;
    cudaError_t err = cudaMalloc(&d_partial, gridSize * sizeof(Real));
    if (err != cudaSuccess) {
        // Fallback: just update state
        updateState(x_old, deltaX, x_new, damping, n);
        return 0.0;
    }
    
    Real result = updateStateAndComputeNorm(x_old, deltaX, x_new, damping, n, d_partial, gridSize, stream);
    cudaFree(d_partial);
    return result;
}

Real updateStateAndComputeNorm(const Real* x_old, const Real* deltaX, Real* x_new,
                              Real damping, Index n, Real* d_partial, size_t partialSize) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    if (gridSize == 0) return 0.0;
    if (static_cast<size_t>(gridSize) > partialSize) {
        // Fallback: just update state
        updateState(x_old, deltaX, x_new, damping, n);
        return 0.0;
    }
    
    constexpr Index warpsPerBlock = (blockSize + 32 - 1) / 32;
    const Index sharedMemSize = warpsPerBlock * sizeof(Real);
    
    // OPTIMIZATION: Use stream for asynchronous execution
    updateStateAndNormKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        x_old, deltaX, x_new, damping, d_partial, n);
    
    // OPTIMIZATION: Use GPU reduction instead of host-side reduction
    // Eliminates host-device transfer for small arrays
    return reducePartialSumsGPU(d_partial, gridSize, stream);
}

// ============================================================================
// Weighted Operations (merged from CudaWeightedOps)
// ============================================================================

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

// Fused kernel: Compute residual and weighted sum of squares in one pass
// r = z - hx, then compute sum(w[i] * r[i]^2)
__global__ void computeResidualAndObjectiveKernel(const Real* z, const Real* hx, const Real* weights,
                                                  Real* residual, Real* partial, Index n) {
    extern __shared__ Real sdata[];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute residual
    Real r = (i < n) ? (z[i] - hx[i]) : 0.0;
    if (i < n) {
        residual[i] = r;
    }
    
    // Compute weighted square: w[i] * r^2
    Real val = (i < n) ? CUDA_FMA(weights[i], CUDA_FMA(r, r, 0.0), 0.0) : 0.0;
    
    // Warp-level reduction
    const unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    
    const Index warpId = tid / warpSize;
    const Index laneId = tid % warpSize;
    if (laneId == 0) {
        sdata[warpId] = val;
    }
    __syncthreads();
    
    // Final reduction across warps
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

// Weighted operation wrappers
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
    std::vector<Real> h_partial(gridSize);
    cudaMemcpy(h_partial.data(), d_partial, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);
    
    Real sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (Index i = 0; i < gridSize; ++i) {
        sum += h_partial[i];
    }
    
    cudaFree(d_partial);
    return sum;
}

// Fused: Compute residual and objective value in one kernel launch
Real computeResidualAndObjective(const Real* z, const Real* hx, const Real* weights,
                                 Real* residual, Index n) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    if (gridSize == 0) return 0.0;
    
    Real* d_partial = nullptr;
    cudaError_t err = cudaMalloc(&d_partial, gridSize * sizeof(Real));
    if (err != cudaSuccess) {
        return 0.0;
    }
    
    Real result = computeResidualAndObjective(z, hx, weights, residual, n, d_partial, gridSize);
    cudaFree(d_partial);
    return result;
}

Real computeResidualAndObjective(const Real* z, const Real* hx, const Real* weights,
                                 Real* residual, Index n, Real* d_partial, size_t partialSize) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    if (gridSize == 0) return 0.0;
    if (static_cast<size_t>(gridSize) > partialSize) {
        return 0.0;  // Buffer too small
    }
    
    constexpr Index warpsPerBlock = (blockSize + 32 - 1) / 32;
    const Index sharedMemSize = warpsPerBlock * sizeof(Real);
    
    // OPTIMIZATION: Use stream for asynchronous execution
    computeResidualAndObjectiveKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        z, hx, weights, residual, d_partial, n);
    
    // OPTIMIZATION: Use GPU reduction instead of host-side reduction
    // Eliminates host-device transfer for small arrays
    return reducePartialSumsGPU(d_partial, gridSize, stream);
}

void weightedAccumulate(Real alpha, const Real* x, const Real* w, Real* y, Index n) {
    constexpr Index blockSize = 256;
    const Index gridSize = (n + blockSize - 1) / blockSize;
    
    weightedAccumulateKernel<<<gridSize, blockSize>>>(alpha, x, w, y, n);
    // Note: No cudaDeviceSynchronize() - caller should sync if needed
}

} // namespace cuda
} // namespace sle

