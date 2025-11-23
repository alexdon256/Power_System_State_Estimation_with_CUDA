/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * GPU-accelerated power flow and injection computation
 */

#ifndef SLE_CUDA_CUDAPOWERFLOW_H
#define SLE_CUDA_CUDAPOWERFLOW_H

#include <sle/Types.h>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

namespace sle {
namespace cuda {

// Device data structures (matching CudaMeasurementKernels.cu)
struct DeviceBus {
    Real baseKV;
    Real gShunt;
    Real bShunt;
};

struct DeviceBranch {
    Index fromBus;
    Index toBus;
    Real r;
    Real x;
    Real b;
    Real tapRatio;
    Real phaseShift;
};

// Note: computeAllPowerFlowsGPU removed - unused (replaced by computeAllPowerFlowsCompleteGPU)

// GPU-accelerated complete power flow computation
// Computes P, Q, MW, MVAR, I_PU, I_Amps for all branches in parallel
// All computations on GPU - eliminates CPU loop and reduces host-device transfers
void computeAllPowerFlowsCompleteGPU(const Real* v, const Real* theta,
                                     const DeviceBranch* branches,
                                     const DeviceBus* buses,
                                     Real baseMVA,
                                     Real* pFlow, Real* qFlow,
                                     Real* pMW, Real* qMVAR,
                                     Real* iPU, Real* iAmps,
                                     Index nBranches, Index nBuses);

// GPU-accelerated power injection computation for all buses
// Computes P and Q injections for all buses in parallel
// Uses CSR format adjacency lists for O(avg_degree) complexity instead of O(nBranches)
// branchFromBus/branchToBus: CSR column indices (branch indices)
// branchFromBusRowPtr/branchToBusRowPtr: CSR row pointers (nBuses+1 elements)
// Performance: O(avg_degree) per bus instead of O(nBranches) - 5-20x speedup for sparse networks
void computeAllPowerInjectionsGPU(const Real* v, const Real* theta,
                                  const DeviceBus* buses,
                                  const DeviceBranch* branches,
                                  const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                  const Index* branchToBus, const Index* branchToBusRowPtr,
                                  Real* pInjection, Real* qInjection,
                                  Index nBuses, Index nBranches);

// Combined GPU kernel: Compute power injections/flows and evaluate measurements
// All operations on GPU, data stays on GPU - eliminates host-device transfers
// stream: Optional CUDA stream for asynchronous execution
void computeMeasurementsGPU(
    const Real* v, const Real* theta,
    const DeviceBus* buses, const DeviceBranch* branches,
    const Index* branchFromBus, const Index* branchFromBusRowPtr,
    const Index* branchToBus, const Index* branchToBusRowPtr,
    const Index* measurementTypes,
    const Index* measurementLocations,
    const Index* measurementBranches,
    Real* pInjection, Real* qInjection,
    Real* pFlow, Real* qFlow,
    Real* hx,
    Index nBuses, Index nBranches, Index nMeasurements,
    cudaStream_t stream = nullptr);

// GPU-accelerated computation of MW/MVAR/I_PU/I_Amps from existing P/Q flows
// Reuses P/Q flows already computed (e.g., during solver iterations) - no recomputation
// Used to optimize: avoid recomputing P/Q flows when they're already on GPU
void computePowerFlowDerivedGPU(
    const Real* v,
    const Real* pFlow, const Real* qFlow,
    const DeviceBranch* branches,
    const DeviceBus* buses,
    Real baseMVA,
    Real* pMW, Real* qMVAR,
    Real* iPU, Real* iAmps,
    Index nBranches, Index nBuses);

// GPU-accelerated Jacobian matrix computation
// Computes Jacobian elements for all measurements in parallel
// Uses CSR adjacency lists for O(avg_degree) complexity in power injection Jacobian
void computeJacobian(const Real* v, const Real* theta,
                    const DeviceBus* buses, const DeviceBranch* branches,
                    const Index* branchFromBus, const Index* branchFromBusRowPtr,
                    const Index* branchToBus, const Index* branchToBusRowPtr,
                    const Index* measurementTypes,
                    const Index* measurementLocations,
                    const Index* measurementBranches,
                    const Index* jacobianRowPtr,
                    const Index* jacobianColInd,
                    Real* jacobianValues,
                    Index nMeasurements, Index nBuses, Index nBranches);

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDAPOWERFLOW_H

