/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_CUDA_CUDASOLVEROPS_H
#define SLE_CUDA_CUDASOLVEROPS_H

#include <sle/Types.h>

namespace sle {
namespace cuda {

// GPU-accelerated solver operations

// Compute residual: r = z - hx
void computeResidual(const Real* z, const Real* hx, Real* residual, Index n);

// Fused: Compute residual and weighted residual in one kernel
// r = z - hx, then wr = weights * r
// stream: Optional CUDA stream for asynchronous execution
void computeResidualAndWeighted(const Real* z, const Real* hx, const Real* weights,
                                Real* residual, Real* weightedResidual, Index n,
                                cudaStream_t stream = nullptr);

// Update state: x_new = x_old + damping * deltaX
void updateState(const Real* x_old, const Real* deltaX, Real* x_new, Real damping, Index n);

// Compute norm squared: ||x||^2 (for convergence check)
// stream: Optional CUDA stream for asynchronous execution
Real computeNormSquared(const Real* x, Index n, cudaStream_t stream = nullptr);

// Compute norm squared with pooled buffer (avoids allocation)
// stream: Optional CUDA stream for asynchronous execution
Real computeNormSquared(const Real* x, Index n, Real* d_partial, size_t partialSize,
                       cudaStream_t stream = nullptr);

// Fused: Update state and compute norm squared in one kernel
// x_new = x_old + damping * deltaX, returns ||deltaX||^2
Real updateStateAndComputeNorm(const Real* x_old, const Real* deltaX, Real* x_new,
                               Real damping, Index n);

// Fused: Update state and compute norm squared with pooled buffer (avoids allocation)
// stream: Optional CUDA stream for asynchronous execution
Real updateStateAndComputeNorm(const Real* x_old, const Real* deltaX, Real* x_new,
                               Real damping, Index n, Real* d_partial, size_t partialSize,
                               cudaStream_t stream = nullptr);

// ============================================================================
// Weighted Operations (merged from CudaWeightedOps)
// ============================================================================

// Compute weighted residual: wr = weights * residual
void computeWeightedResidual(const Real* residual, const Real* weights, 
                             Real* weightedResidual, Index n);

// Compute weighted sum of squares: sum(w[i] * x[i]^2)
Real computeWeightedSumSquares(const Real* x, const Real* w, Index n);

// Fused: Compute residual and objective value in one kernel
// r = z - hx, returns sum(w[i] * r[i]^2)
Real computeResidualAndObjective(const Real* z, const Real* hx, const Real* weights,
                                 Real* residual, Index n);

// Fused: Compute residual and objective value with pooled buffer (avoids allocation)
// stream: Optional CUDA stream for asynchronous execution
Real computeResidualAndObjective(const Real* z, const Real* hx, const Real* weights,
                                 Real* residual, Index n, Real* d_partial, size_t partialSize,
                                 cudaStream_t stream = nullptr);

// Weighted accumulate: y[i] = alpha * x[i] * w[i] + y[i]
void weightedAccumulate(Real alpha, const Real* x, const Real* w, Real* y, Index n);

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDASOLVEROPS_H

