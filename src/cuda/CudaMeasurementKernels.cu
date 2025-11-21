/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <cuda_runtime.h>
#include <sle/Types.h>
#include <sle/model/NetworkModel.h>
#include <cmath>

// Precision-aware intrinsics: automatically detect Real type (double or float)
// Real is defined as double in Types.h, so use double precision intrinsics
//
// Note: __CUDACC__ is defined when nvcc compiles (both host and device code)
//       __CUDA_ARCH__ is only defined in device code (__device__/__global__ functions)
//       For IDE parsing, neither may be defined, so we fall back to regular operations
#if defined(__CUDA_ARCH__)
    // Device code: __CUDA_ARCH__ is defined
#if __CUDA_ARCH__ >= 600
    #define CUDA_FMA(a, b, c) __fma_rn((a), (b), (c))
    #define CUDA_SINCOS(x, s, c) __dsincos((x), (s), (c))
    #define CUDA_DIV(a, b) ((a) / (b))
#else
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #define CUDA_SINCOS(x, s, c) do { *(s) = sin(x); *(c) = cos(x); } while(0)
    #define CUDA_DIV(a, b) ((a) / (b))
#endif
#elif defined(__CUDACC__)
    // Host code in .cu file: __CUDACC__ defined but __CUDA_ARCH__ not defined
    // Use regular operations (host code doesn't need intrinsics)
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #define CUDA_SINCOS(x, s, c) do { *(s) = sin(x); *(c) = cos(x); } while(0)
    #define CUDA_DIV(a, b) ((a) / (b))
#else
    // IDE parsing or regular C++ compiler: use regular operations
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #define CUDA_SINCOS(x, s, c) do { *(s) = sin(x); *(c) = cos(x); } while(0)
    #define CUDA_DIV(a, b) ((a) / (b))
#endif

namespace sle {
namespace cuda {

// Device data structures are declared in CudaPowerFlow.h

// Optimized measurement function kernels with FMA and fast math
// Combined power flow computation: computes both P and Q in a single pass
// This eliminates redundant calculations when both values are needed
// Returns P via first parameter, Q via second parameter
__device__ __forceinline__ void computePowerFlowPQ(const Real* v, const Real* theta,
                                   const DeviceBranch* branches, Index branchIdx,
                                   Index fromBus, Index toBus, Index nBuses,
                                   Real& p, Real& q) {
    // Bounds checking: validate bus indices before accessing arrays
    if (fromBus < 0 || fromBus >= nBuses || toBus < 0 || toBus >= nBuses) {
        p = 0.0;
        q = 0.0;
        return;
    }
    
    // Validate branch index
    if (branchIdx < 0) {
        p = 0.0;
        q = 0.0;
        return;
    }
    
    const DeviceBranch& br = branches[branchIdx];
    
    // Coalesced memory access (now safe after bounds checking)
    const Real vi = v[fromBus];
    const Real vj = v[toBus];
    const Real thetai = theta[fromBus];
    const Real thetaj = theta[toBus];
    
    // Optimized admittance calculation using FMA (with division by zero check)
    const Real z2 = CUDA_FMA(br.r, br.r, br.x * br.x);
    if (z2 < 1e-12) {  // Avoid division by zero (very small impedance)
        p = 0.0;
        q = 0.0;
        return;
    }
    const Real inv_z2 = CUDA_DIV(1.0, z2);
    const Real g = CUDA_FMA(br.r, inv_z2, 0.0);  // br.r * inv_z2
    const Real b = CUDA_FMA(-br.x, inv_z2, 0.0);  // -br.x * inv_z2
    
    const Real tap = br.tapRatio;
    if (fabs(tap) < 1e-12) {  // Avoid division by zero (tap ratio too small)
        p = 0.0;
        q = 0.0;
        return;
    }
    const Real phase = br.phaseShift;
    const Real thetaDiff = thetai - thetaj - phase;
    
    // Use sincos for simultaneous sin/cos (precision-aware)
    // This is computed once and reused for both P and Q
    Real sinDiff, cosDiff;
    CUDA_SINCOS(thetaDiff, &sinDiff, &cosDiff);
    
    // Pre-compute common terms (shared between P and Q calculations)
    const Real tap2 = CUDA_FMA(tap, tap, 0.0);  // tap * tap
    const Real inv_tap2 = CUDA_DIV(1.0, tap2);
    const Real vi2 = CUDA_FMA(vi, vi, 0.0);  // vi * vi
    const Real vivj = CUDA_FMA(vi, vj, 0.0);  // vi * vj
    const Real inv_tap = CUDA_DIV(1.0, tap);
    
    // Compute P (active power)
    const Real term1_p = CUDA_FMA(vi2, g * inv_tap2, 0.0);  // vi^2 * g / tap^2
    const Real gcos = CUDA_FMA(g, cosDiff, 0.0);
    const Real bsin = CUDA_FMA(b, sinDiff, 0.0);
    const Real term2_p = CUDA_FMA(vivj, (gcos + bsin) * inv_tap, 0.0);  // vi*vj*(g*cos+b*sin)/tap
    p = CUDA_FMA(term1_p, 1.0, -term2_p);  // term1_p - term2_p
    
    // Compute Q (reactive power) - reuses pre-computed values
    const Real b_total = CUDA_FMA(b, 1.0, br.b * 0.5);  // b + br.b/2
    const Real term1_q = CUDA_FMA(vi2, b_total * inv_tap2, 0.0);  // vi^2 * (b + br.b/2) / tap^2
    const Real gsin = CUDA_FMA(g, sinDiff, 0.0);
    const Real bcos = CUDA_FMA(b, cosDiff, 0.0);
    const Real term2_q = CUDA_FMA(vivj, (gsin - bcos) * inv_tap, 0.0);  // vi*vj*(g*sin-b*cos)/tap
    q = CUDA_FMA(-term1_q, 1.0, -term2_q);  // -term1_q - term2_q
}

// Combined power injection computation: computes both P and Q in a single pass
// Uses CSR adjacency lists for O(avg_degree) complexity instead of O(nBranches)
// branchFromBus/branchToBus are CSR format: branchFromBus[rowPtr[i]:rowPtr[i+1]] contains branches from bus i
__device__ __forceinline__ void computePowerInjectionPQ(const Real* v, const Real* theta,
                                       const DeviceBus* buses, const DeviceBranch* branches,
                                       const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                       const Index* branchToBus, const Index* branchToBusRowPtr,
                                       Index busIdx, Index nBuses, Index nBranches,
                                       Real& p, Real& q) {
    // Bounds checking: validate bus index before accessing arrays
    if (busIdx < 0 || busIdx >= nBuses) {
        p = 0.0;
        q = 0.0;
        return;
    }
    
    const DeviceBus& bus = buses[busIdx];
    
    // Shunt contribution using FMA (shared v^2 computation)
    const Real vi2 = CUDA_FMA(v[busIdx], v[busIdx], 0.0);
    p = CUDA_FMA(vi2, bus.gShunt, 0.0);
    q = CUDA_FMA(-vi2, bus.bShunt, 0.0);
    
    // Outgoing branch contributions (using CSR adjacency list)
    // Validate busIdx+1 is within row pointer array bounds (array has nBuses+1 elements)
    if (busIdx + 1 <= nBuses) {
        Index fromStart = branchFromBusRowPtr[busIdx];
        Index fromEnd = branchFromBusRowPtr[busIdx + 1];
        // Validate CSR format: fromEnd >= fromStart and within reasonable bounds
        if (fromEnd >= fromStart && fromEnd <= static_cast<Index>(nBranches)) {
            for (Index i = fromStart; i < fromEnd; ++i) {
                // Bounds check: ensure CSR index is valid
                if (i >= 0 && i < static_cast<Index>(nBranches)) {
                    Index brIdx = branchFromBus[i];
                    // Validate branch index is within bounds
                    if (brIdx >= 0 && brIdx < nBranches) {
                        const DeviceBranch& br = branches[brIdx];
                        // computePowerFlowPQ will validate bus indices internally
                        Real pFlow, qFlow;
                        computePowerFlowPQ(v, theta, branches, brIdx, br.fromBus, br.toBus, nBuses, pFlow, qFlow);
                        p = CUDA_FMA(p, 1.0, pFlow);  // p += pFlow
                        q = CUDA_FMA(q, 1.0, qFlow);  // q += qFlow
                    }
                }
            }
        }
    }
    
    // Incoming branch contributions (reverse direction, using CSR adjacency list)
    // Validate busIdx+1 is within row pointer array bounds
    if (busIdx + 1 <= nBuses) {
        Index toStart = branchToBusRowPtr[busIdx];
        Index toEnd = branchToBusRowPtr[busIdx + 1];
        // Validate CSR format: toEnd >= toStart and within reasonable bounds
        if (toEnd >= toStart && toEnd <= static_cast<Index>(nBranches)) {
            for (Index i = toStart; i < toEnd; ++i) {
                // Bounds check: ensure CSR index is valid
                if (i >= 0 && i < static_cast<Index>(nBranches)) {
                    Index brIdx = branchToBus[i];
                    // Validate branch index is within bounds
                    if (brIdx >= 0 && brIdx < nBranches) {
                        const DeviceBranch& br = branches[brIdx];
                        // computePowerFlowPQ will validate bus indices internally
                        Real pFlow, qFlow;
                        // Compute in reverse direction (to -> from)
                        computePowerFlowPQ(v, theta, branches, brIdx, br.toBus, br.fromBus, nBuses, pFlow, qFlow);
                        p = CUDA_FMA(p, 1.0, -pFlow);  // p -= pFlow (reverse direction)
                        q = CUDA_FMA(q, 1.0, -qFlow);  // q -= qFlow (reverse direction)
                    }
                }
            }
        }
    }
}

// GPU-accelerated power flow computation for all branches
// Uses combined computePowerFlowPQ to compute both P and Q in a single pass
__global__ void computeAllPowerFlowsKernel(const Real* v, const Real* theta,
                                           const DeviceBranch* branches,
                                           Real* pFlow, Real* qFlow,
                                           Index nBranches, Index nBuses) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= 0 && idx < nBranches) {
        const DeviceBranch& br = branches[idx];
        
        // Validate branch bus indices before computing power flow
        if (br.fromBus >= 0 && br.fromBus < nBuses && 
            br.toBus >= 0 && br.toBus < nBuses) {
            // Use combined function to compute both P and Q in one pass
            // This eliminates redundant calculations (sincos, admittance, etc.)
            Real p, q;
            computePowerFlowPQ(v, theta, branches, idx, br.fromBus, br.toBus, nBuses, p, q);
            
            pFlow[idx] = p;
            qFlow[idx] = q;
        } else {
            // Invalid bus indices - set power flow to zero
            pFlow[idx] = 0.0;
            qFlow[idx] = 0.0;
        }
    }
}

// GPU-accelerated power injection computation for all buses
// Optimized with CSR adjacency lists: O(avg_degree) instead of O(nBranches) per bus
// branchFromBus/branchToBus are CSR column indices, rowPtr are row pointers
__global__ void computeAllPowerInjectionsKernel(const Real* v, const Real* theta,
                                                const DeviceBus* buses,
                                                const DeviceBranch* branches,
                                                const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                                const Index* branchToBus, const Index* branchToBusRowPtr,
                                                Real* pInjection, Real* qInjection,
                                                Index nBuses, Index nBranches) {
    Index busIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (busIdx < nBuses) {
        const DeviceBus& bus = buses[busIdx];
        
        // Shunt contribution (shared v^2 computation)
        Real vi2 = CUDA_FMA(v[busIdx], v[busIdx], 0.0);
        Real p = CUDA_FMA(vi2, bus.gShunt, 0.0);  // v^2 * g
        Real q = CUDA_FMA(-vi2, bus.bShunt, 0.0);  // -v^2 * b
        
        // Outgoing branch contributions (using CSR adjacency list)
        // Bounds checking: ensure row pointers are valid and within CSR array bounds
        Index fromStart = branchFromBusRowPtr[busIdx];
        Index fromEnd = branchFromBusRowPtr[busIdx + 1];
        // Validate: fromEnd >= fromStart (CSR format requirement)
        if (fromEnd >= fromStart) {
            for (Index i = fromStart; i < fromEnd; ++i) {
                // Bounds check: ensure CSR index is valid (defensive programming)
                // Note: In correct CSR format, i should always be valid, but we check anyway
                Index brIdx = branchFromBus[i];
                // Validate branch index is within bounds
                if (brIdx >= 0 && brIdx < nBranches) {
                    const DeviceBranch& br = branches[brIdx];
                    // Validate branch connects to this bus (sanity check)
                    if (br.fromBus == busIdx) {
                        Real pFlow, qFlow;
                        // Use combined function to compute both P and Q in one pass
                        computePowerFlowPQ(v, theta, branches, brIdx, br.fromBus, br.toBus, nBuses, pFlow, qFlow);
                        p = CUDA_FMA(p, 1.0, pFlow);  // p += pFlow
                        q = CUDA_FMA(q, 1.0, qFlow);  // q += qFlow
                    }
                }
            }
        }
        
        // Incoming branch contributions (reverse direction, using CSR adjacency list)
        Index toStart = branchToBusRowPtr[busIdx];
        Index toEnd = branchToBusRowPtr[busIdx + 1];
        // Validate: toEnd >= toStart (CSR format requirement)
        if (toEnd >= toStart) {
            for (Index i = toStart; i < toEnd; ++i) {
                // Bounds check: ensure CSR index is valid
                Index brIdx = branchToBus[i];
                // Validate branch index is within bounds
                if (brIdx >= 0 && brIdx < nBranches) {
                    const DeviceBranch& br = branches[brIdx];
                    // Validate branch connects to this bus (sanity check)
                    if (br.toBus == busIdx) {
                        Real pFlow, qFlow;
                        // Compute in reverse direction (to -> from)
                        computePowerFlowPQ(v, theta, branches, brIdx, br.toBus, br.fromBus, nBuses, pFlow, qFlow);
                        p = CUDA_FMA(p, 1.0, -pFlow);  // p -= pFlow (reverse direction)
                        q = CUDA_FMA(q, 1.0, -qFlow);  // q -= qFlow (reverse direction)
                    }
                }
            }
        }
        
        pInjection[busIdx] = p;
        qInjection[busIdx] = q;
    }
}

// Wrapper functions
void computeAllPowerFlowsGPU(const Real* v, const Real* theta,
                            const DeviceBranch* branches,
                            Real* pFlow, Real* qFlow,
                            Index nBranches, Index nBuses) {
    constexpr Index blockSize = 256;
    const Index gridSize = (nBranches + blockSize - 1) / blockSize;
    
    computeAllPowerFlowsKernel<<<gridSize, blockSize>>>(
        v, theta, branches, pFlow, qFlow, nBranches, nBuses);
}

void computeAllPowerInjectionsGPU(const Real* v, const Real* theta,
                                 const DeviceBus* buses,
                                 const DeviceBranch* branches,
                                 const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                 const Index* branchToBus, const Index* branchToBusRowPtr,
                                 Real* pInjection, Real* qInjection,
                                 Index nBuses, Index nBranches) {
    constexpr Index blockSize = 256;
    const Index gridSize = (nBuses + blockSize - 1) / blockSize;
    
    computeAllPowerInjectionsKernel<<<gridSize, blockSize>>>(
        v, theta, buses, branches, 
        branchFromBus, branchFromBusRowPtr,
        branchToBus, branchToBusRowPtr,
        pInjection, qInjection, nBuses, nBranches);
}

} // namespace cuda
} // namespace sle

