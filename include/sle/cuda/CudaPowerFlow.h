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

// GPU-accelerated power flow computation for all branches
// Computes P and Q flows for all branches in parallel
void computeAllPowerFlowsGPU(const Real* v, const Real* theta,
                             const DeviceBranch* branches,
                             Real* pFlow, Real* qFlow,
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

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDAPOWERFLOW_H

