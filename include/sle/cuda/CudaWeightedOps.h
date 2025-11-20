/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_CUDA_CUDAWEIGHTEDOPS_H
#define SLE_CUDA_CUDAWEIGHTEDOPS_H

#include <sle/Types.h>

namespace sle {
namespace cuda {

// GPU-accelerated weighted operations
void computeWeightedResidual(const Real* residual, const Real* weights, 
                             Real* weightedResidual, Index n);

Real computeWeightedSumSquares(const Real* x, const Real* w, Index n);

void weightedAccumulate(Real alpha, const Real* x, const Real* w, Real* y, Index n);

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDAWEIGHTEDOPS_H

