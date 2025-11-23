/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * GPU reduction kernels for partial sums (optimization: reduce on GPU instead of host)
 */

#include <cuda_runtime.h>
#include <sle/Types.h>

namespace sle {
namespace cuda {

// GPU reduction kernel: Reduce partial sums to single value
// Uses warp shuffles for efficient reduction
__global__ void reducePartialSumsKernel(const Real* partial, Real* result, Index n) {
    extern __shared__ Real sdata[];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load partial sum
    Real val = (i < n) ? partial[i] : 0.0;
    
    // Warp-level reduction using shuffle
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
            result[blockIdx.x] = sum;
        }
    }
}

// GPU reduction: Reduce partial sums array to single value on GPU
// Returns the reduced value
// OPTIMIZATION: Reduces on GPU instead of copying to host and reducing there
Real reducePartialSumsGPU(const Real* d_partial, Index n, cudaStream_t stream) {
    if (n == 0) return 0.0;
    if (n == 1) {
        Real result = 0.0;
        cudaMemcpyAsync(&result, d_partial, sizeof(Real), cudaMemcpyDeviceToHost, stream ? stream : 0);
        if (stream) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }
        return result;
    }
    
    constexpr Index blockSize = 256;
    Index gridSize = (n + blockSize - 1) / blockSize;
    
    // Allocate device memory for intermediate results
    Real* d_result = nullptr;
    cudaError_t err = cudaMalloc(&d_result, gridSize * sizeof(Real));
    if (err != cudaSuccess) {
        return 0.0;
    }
    
    constexpr Index warpsPerBlock = (blockSize + 32 - 1) / 32;
    const Index sharedMemSize = warpsPerBlock * sizeof(Real);
    
    // First reduction pass
    reducePartialSumsKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_partial, d_result, n);
    
    // Recursive reduction if needed
    Index remaining = gridSize;
    Real* d_input = d_result;
    Real* d_output = nullptr;
    
    while (remaining > 1) {
        gridSize = (remaining + blockSize - 1) / blockSize;
        if (gridSize == 1) {
            // Final reduction - result will be in d_input[0]
            break;
        }
        
        err = cudaMalloc(&d_output, gridSize * sizeof(Real));
        if (err != cudaSuccess) {
            cudaFree(d_result);
            return 0.0;
        }
        
        reducePartialSumsKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
            d_input, d_output, remaining);
        
        if (d_input != d_result) {
            cudaFree(d_input);
        }
        d_input = d_output;
        remaining = gridSize;
    }
    
    // Copy final result to host
    Real result = 0.0;
    cudaMemcpyAsync(&result, d_input, sizeof(Real), cudaMemcpyDeviceToHost, stream ? stream : 0);
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    // Cleanup
    if (d_input != d_result) {
        cudaFree(d_input);
    }
    cudaFree(d_result);
    
    return result;
}

} // namespace cuda
} // namespace sle

