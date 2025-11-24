/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * CUDA kernels for network data structure building on GPU
 * Eliminates CPU-GPU transfer overhead for adjacency lists
 */

#include <sle/cuda/CudaPowerFlow.h>
#include <sle/cuda/CudaUtils.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace sle {
namespace cuda {

// Count branches per bus (first pass)
__global__ void countBranchesPerBusKernel(
    const DeviceBranch* branches,
    Index* branchFromBusCounts, Index* branchToBusCounts,
    Index nBranches, Index nBuses) {
    
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nBranches) return;
    
    const DeviceBranch& br = branches[idx];
    
    // Count branches per bus using atomic operations
    if (br.fromBus >= 0 && br.fromBus < nBuses) {
        atomicAdd(&branchFromBusCounts[br.fromBus], 1);
    }
    if (br.toBus >= 0 && br.toBus < nBuses) {
        atomicAdd(&branchToBusCounts[br.toBus], 1);
    }
}

// Second pass: Fill column indices using row pointers
// After exclusive scan, rowPtr[i] contains the starting index for row i
__global__ void fillCSRAdjacencyListsKernel(
    const DeviceBranch* branches,
    Index* branchFromBus, Index* branchToBus,
    const Index* branchFromBusRowPtr, const Index* branchToBusRowPtr,
    Index* branchFromBusOffsets, Index* branchToBusOffsets,
    Index nBranches, Index nBuses) {
    
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nBranches) return;
    
    const DeviceBranch& br = branches[idx];
    
    // Fill column indices using atomic offsets within row range
    if (br.fromBus >= 0 && br.fromBus < nBuses) {
        Index rowStart = branchFromBusRowPtr[br.fromBus];
        Index rowEnd = branchFromBusRowPtr[br.fromBus + 1];
        if (rowStart < rowEnd) {
            Index offset = atomicAdd(&branchFromBusOffsets[br.fromBus], 1);
            if (offset < (rowEnd - rowStart)) {
                branchFromBus[rowStart + offset] = idx;
            }
        }
    }
    if (br.toBus >= 0 && br.toBus < nBuses) {
        Index rowStart = branchToBusRowPtr[br.toBus];
        Index rowEnd = branchToBusRowPtr[br.toBus + 1];
        if (rowStart < rowEnd) {
            Index offset = atomicAdd(&branchToBusOffsets[br.toBus], 1);
            if (offset < (rowEnd - rowStart)) {
                branchToBus[rowStart + offset] = idx;
            }
        }
    }
}

// Optimized single-pass kernel using shared memory for better performance
__global__ void buildCSRAdjacencyListsOptimizedKernel(
    const DeviceBranch* branches,
    Index* branchFromBus, Index* branchToBus,
    Index* branchFromBusRowPtr, Index* branchToBusRowPtr,
    Index nBranches, Index nBuses) {
    
    // Use shared memory to reduce atomic contention
    extern __shared__ Index s_counts[];
    Index* s_fromCounts = s_counts;
    Index* s_toCounts = s_counts + nBuses;
    
    Index tid = threadIdx.x;
    Index bid = blockIdx.x;
    Index idx = bid * blockDim.x + tid;
    
    // Initialize shared memory
    if (tid < nBuses) {
        s_fromCounts[tid] = 0;
        s_toCounts[tid] = 0;
    }
    __syncthreads();
    
    // Count branches per bus in shared memory
    if (idx < nBranches) {
        const DeviceBranch& br = branches[idx];
        if (br.fromBus >= 0 && br.fromBus < nBuses) {
            atomicAdd(&s_fromCounts[br.fromBus], 1);
        }
        if (br.toBus >= 0 && br.toBus < nBuses) {
            atomicAdd(&s_toCounts[br.toBus], 1);
        }
    }
    __syncthreads();
    
    // Write counts to global memory (coalesced)
    if (tid < nBuses) {
        if (s_fromCounts[tid] > 0) {
            atomicAdd(&branchFromBusRowPtr[tid + 1], s_fromCounts[tid]);
        }
        if (s_toCounts[tid] > 0) {
            atomicAdd(&branchToBusRowPtr[tid + 1], s_toCounts[tid]);
        }
    }
}

// GPU-accelerated CSR adjacency list building
// Builds CSR format directly on GPU from DeviceBranch array
// Returns true on success, false on failure
bool buildCSRAdjacencyListsGPU(
    const DeviceBranch* d_branches,
    Index* d_branchFromBus, Index* d_branchToBus,
    Index* d_branchFromBusRowPtr, Index* d_branchToBusRowPtr,
    Index nBranches, Index nBuses,
    cudaStream_t stream) {
    
    if (nBranches == 0 || nBuses == 0) {
        // Initialize row pointers to zero
        if (d_branchFromBusRowPtr) {
            cudaMemsetAsync(d_branchFromBusRowPtr, 0, (nBuses + 1) * sizeof(Index), stream);
        }
        if (d_branchToBusRowPtr) {
            cudaMemsetAsync(d_branchToBusRowPtr, 0, (nBuses + 1) * sizeof(Index), stream);
        }
        return true;
    }
    
    // Allocate temporary count arrays
    Index* d_fromCounts = nullptr;
    Index* d_toCounts = nullptr;
    Index* d_fromOffsets = nullptr;
    Index* d_toOffsets = nullptr;
    
    cudaError_t err = cudaMalloc(&d_fromCounts, nBuses * sizeof(Index));
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_toCounts, nBuses * sizeof(Index));
    if (err != cudaSuccess) {
        cudaFree(d_fromCounts);
        return false;
    }
    err = cudaMalloc(&d_fromOffsets, nBuses * sizeof(Index));
    if (err != cudaSuccess) {
        cudaFree(d_fromCounts);
        cudaFree(d_toCounts);
        return false;
    }
    err = cudaMalloc(&d_toOffsets, nBuses * sizeof(Index));
    if (err != cudaSuccess) {
        cudaFree(d_fromCounts);
        cudaFree(d_toCounts);
        cudaFree(d_fromOffsets);
        return false;
    }
    
    // Initialize counts and offsets to zero
    cudaMemsetAsync(d_fromCounts, 0, nBuses * sizeof(Index), stream);
    cudaMemsetAsync(d_toCounts, 0, nBuses * sizeof(Index), stream);
    cudaMemsetAsync(d_fromOffsets, 0, nBuses * sizeof(Index), stream);
    cudaMemsetAsync(d_toOffsets, 0, nBuses * sizeof(Index), stream);
    cudaMemsetAsync(d_branchFromBusRowPtr, 0, (nBuses + 1) * sizeof(Index), stream);
    cudaMemsetAsync(d_branchToBusRowPtr, 0, (nBuses + 1) * sizeof(Index), stream);
    
    // Pass 1: Count branches per bus
    constexpr Index blockSize = 256;
    const Index gridSize = KernelConfig<blockSize>::gridSize(nBranches);
    
    countBranchesPerBusKernel<<<gridSize, blockSize, 0, stream>>>(
        d_branches, d_fromCounts, d_toCounts, nBranches, nBuses);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_fromCounts);
        cudaFree(d_toCounts);
        cudaFree(d_fromOffsets);
        cudaFree(d_toOffsets);
        return false;
    }
    
    // Build row pointers using exclusive scan (prefix sum)
    // Copy counts to rowPtr+1, then scan
    cudaMemcpyAsync(d_branchFromBusRowPtr + 1, d_fromCounts, nBuses * sizeof(Index), 
                   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_branchToBusRowPtr + 1, d_toCounts, nBuses * sizeof(Index), 
                   cudaMemcpyDeviceToDevice, stream);
    
    // Perform exclusive scan
    thrust::device_ptr<Index> d_fromPtr(d_branchFromBusRowPtr + 1);
    thrust::device_ptr<Index> d_toPtr(d_branchToBusRowPtr + 1);
    
    if (stream) {
        thrust::exclusive_scan(thrust::cuda::par.on(stream),
                               d_fromPtr, d_fromPtr + nBuses, d_fromPtr);
        thrust::exclusive_scan(thrust::cuda::par.on(stream),
                               d_toPtr, d_toPtr + nBuses, d_toPtr);
    } else {
        thrust::exclusive_scan(d_fromPtr, d_fromPtr + nBuses, d_fromPtr);
        thrust::exclusive_scan(d_toPtr, d_toPtr + nBuses, d_toPtr);
    }
    
    // Pass 2: Fill column indices
    fillCSRAdjacencyListsKernel<<<gridSize, blockSize, 0, stream>>>(
        d_branches, d_branchFromBus, d_branchToBus,
        d_branchFromBusRowPtr, d_branchToBusRowPtr,
        d_fromOffsets, d_toOffsets,
        nBranches, nBuses);
    
    err = cudaGetLastError();
    
    // Cleanup
    cudaFree(d_fromCounts);
    cudaFree(d_toCounts);
    cudaFree(d_fromOffsets);
    cudaFree(d_toOffsets);
    
    return (err == cudaSuccess);
}

} // namespace cuda
} // namespace sle

