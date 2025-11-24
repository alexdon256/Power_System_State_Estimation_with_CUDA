/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * GPU reduction utilities for partial sums
 */

#ifndef SLE_CUDA_CUDAREDUCTION_H
#define SLE_CUDA_CUDAREDUCTION_H

#include <sle/Types.h>
#include <cuda_runtime.h>

namespace sle {
namespace cuda {

// GPU reduction: Reduce partial sums array to single value on GPU
// OPTIMIZATION: Reduces on GPU instead of copying to host and reducing there
// Eliminates host-device transfer for small arrays
// stream: Optional CUDA stream for asynchronous execution
Real reducePartialSumsGPU(const Real* d_partial, Index n, cudaStream_t stream = nullptr);

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDAREDUCTION_H


