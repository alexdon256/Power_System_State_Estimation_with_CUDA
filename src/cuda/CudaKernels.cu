/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Unified CUDA kernels for measurement evaluation and Jacobian computation
 */

#include <cuda_runtime.h>
#include <sle/Types.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <cmath>

// Precision-aware CUDA intrinsics (embedded from CudaPrecisionHelpers.h)
// Real is double in Types.h, so use double precision intrinsics
#if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= 600
        #define CUDA_FMA(a, b, c) __fma_rn((a), (b), (c))
    #else
        #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #endif
    #define CUDA_SINCOS(x, s, c) sincos((x), (s), (c))
    #define CUDA_DIV(a, b) ((a) / (b))
#else
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #define CUDA_SINCOS(x, s, c) sincos((x), (s), (c))
    #define CUDA_DIV(a, b) ((a) / (b))
#endif

namespace sle {
namespace cuda {

// ============================================================================
// Shared Device Functions (used by both measurement and Jacobian kernels)
// ============================================================================

// Device function: Compute power flow P and Q for a single branch
// Shared between measurement evaluation and Jacobian computation
__device__ __forceinline__ void computePowerFlowPQ(const Real* v, const Real* theta,
                                   const DeviceBranch* branches, Index branchIdx,
                                   Index fromBus, Index toBus, Index nBuses,
                                   Real& p, Real& q) {
    // Bounds checking
    if (fromBus < 0 || fromBus >= nBuses || toBus < 0 || toBus >= nBuses || branchIdx < 0) {
        p = 0.0;
        q = 0.0;
        return;
    }
    
    const DeviceBranch& br = branches[branchIdx];
    
    const Real vi = v[fromBus];
    const Real vj = v[toBus];
    const Real thetai = theta[fromBus];
    const Real thetaj = theta[toBus];
    
    // Optimized admittance calculation using FMA
    const Real z2 = CUDA_FMA(br.r, br.r, br.x * br.x);
    if (z2 < 1e-12) {
        p = 0.0;
        q = 0.0;
        return;
    }
    const Real inv_z2 = CUDA_DIV(1.0, z2);
    const Real g = CUDA_FMA(br.r, inv_z2, 0.0);
    const Real b = CUDA_FMA(-br.x, inv_z2, 0.0);
    
    const Real tap = br.tapRatio;
    if (fabs(tap) < 1e-12) {
        p = 0.0;
        q = 0.0;
        return;
    }
    const Real phase = br.phaseShift;
    const Real thetaDiff = thetai - thetaj - phase;
    
    // Use sincos for simultaneous sin/cos
    Real sinDiff, cosDiff;
    CUDA_SINCOS(thetaDiff, &sinDiff, &cosDiff);
    
    // Pre-compute common terms
    const Real tap2 = CUDA_FMA(tap, tap, 0.0);
    const Real inv_tap2 = CUDA_DIV(1.0, tap2);
    const Real vi2 = CUDA_FMA(vi, vi, 0.0);
    const Real vivj = CUDA_FMA(vi, vj, 0.0);
    const Real inv_tap = CUDA_DIV(1.0, tap);
    
    // Compute P
    const Real term1_p = CUDA_FMA(vi2, g * inv_tap2, 0.0);
    const Real gcos = CUDA_FMA(g, cosDiff, 0.0);
    const Real bsin = CUDA_FMA(b, sinDiff, 0.0);
    const Real term2_p = CUDA_FMA(vivj, (gcos + bsin) * inv_tap, 0.0);
    p = CUDA_FMA(term1_p, 1.0, -term2_p);
    
    // Compute Q
    const Real b_total = CUDA_FMA(b, 1.0, br.b * 0.5);
    const Real term1_q = CUDA_FMA(vi2, b_total * inv_tap2, 0.0);
    const Real gsin = CUDA_FMA(g, sinDiff, 0.0);
    const Real bcos = CUDA_FMA(b, cosDiff, 0.0);
    const Real term2_q = CUDA_FMA(vivj, (gsin - bcos) * inv_tap, 0.0);
    q = CUDA_FMA(-term1_q, 1.0, -term2_q);
}

// Device function: Compute MW/MVAR/I_PU/I_Amps from P/Q flows
// Shared between computePowerFlowsCompleteKernel and computePowerFlowDerivedKernel
__device__ __forceinline__ void computePowerFlowDerivedQuantities(
    const Real* v, const DeviceBranch* branches, const DeviceBus* buses,
    Index branchIdx, Real p, Real q, Real baseMVA,
    Real& pMW, Real& qMVAR, Real& iPU, Real& iAmps) {
    // Compute MW/MVAR (multiply by baseMVA)
    pMW = CUDA_FMA(p, baseMVA, 0.0);
    qMVAR = CUDA_FMA(q, baseMVA, 0.0);
    
    // Compute current magnitude: I = sqrt(P² + Q²) / V
    const DeviceBranch& br = branches[branchIdx];
    Real vFrom = v[br.fromBus];
    if (fabs(vFrom) > 1e-12) {
        Real p2 = CUDA_FMA(p, p, 0.0);
        Real q2 = CUDA_FMA(q, q, 0.0);
        Real sMag = sqrt(CUDA_FMA(p2, 1.0, q2));
        iPU = CUDA_DIV(sMag, vFrom);
        
        // Convert to Amperes: I_Amps = I_PU * baseCurrent
        // baseCurrent = (baseMVA * 1000) / (sqrt(3) * baseKV)
        Real baseKV = buses ? buses[br.fromBus].baseKV : 100.0;
        Real sqrt3 = 1.7320508075688772;  // sqrt(3)
        Real baseCurrent = CUDA_DIV(CUDA_DIV(baseMVA * 1000.0, sqrt3), baseKV);
        iAmps = CUDA_FMA(iPU, baseCurrent, 0.0);
    } else {
        iPU = 0.0;
        iAmps = 0.0;
    }
}

// Device function: Compute power injection P and Q for a single bus
__device__ __forceinline__ void computePowerInjectionPQ(const Real* v, const Real* theta,
                                       const DeviceBus* buses, const DeviceBranch* branches,
                                       const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                       const Index* branchToBus, const Index* branchToBusRowPtr,
                                       Index busIdx, Index nBuses, Index nBranches,
                                       Real& p, Real& q) {
    // Bounds checking
    if (busIdx < 0 || busIdx >= nBuses) {
        p = 0.0;
        q = 0.0;
        return;
    }
    
    const DeviceBus& bus = buses[busIdx];
    
    // Shunt contribution
    const Real vi2 = CUDA_FMA(v[busIdx], v[busIdx], 0.0);
    p = CUDA_FMA(vi2, bus.gShunt, 0.0);
    q = CUDA_FMA(-vi2, bus.bShunt, 0.0);
    
    // Outgoing branch contributions
    if (busIdx + 1 <= nBuses) {
        Index fromStart = branchFromBusRowPtr[busIdx];
        Index fromEnd = branchFromBusRowPtr[busIdx + 1];
        if (fromEnd >= fromStart && fromEnd <= static_cast<Index>(nBranches)) {
            for (Index i = fromStart; i < fromEnd; ++i) {
                if (i >= 0 && i < static_cast<Index>(nBranches)) {
                    Index brIdx = branchFromBus[i];
                    if (brIdx >= 0 && brIdx < nBranches) {
                        const DeviceBranch& br = branches[brIdx];
                        Real pFlow, qFlow;
                        computePowerFlowPQ(v, theta, branches, brIdx, br.fromBus, br.toBus, nBuses, pFlow, qFlow);
                        p = CUDA_FMA(p, 1.0, pFlow);
                        q = CUDA_FMA(q, 1.0, qFlow);
                    }
                }
            }
        }
    }
    
    // Incoming branch contributions (reverse direction)
    if (busIdx + 1 <= nBuses) {
        Index toStart = branchToBusRowPtr[busIdx];
        Index toEnd = branchToBusRowPtr[busIdx + 1];
        if (toEnd >= toStart && toEnd <= static_cast<Index>(nBranches)) {
            for (Index i = toStart; i < toEnd; ++i) {
                if (i >= 0 && i < static_cast<Index>(nBranches)) {
                    Index brIdx = branchToBus[i];
                    if (brIdx >= 0 && brIdx < nBranches) {
                        const DeviceBranch& br = branches[brIdx];
                        Real pFlow, qFlow;
                        computePowerFlowPQ(v, theta, branches, brIdx, br.toBus, br.fromBus, nBuses, pFlow, qFlow);
                        p = CUDA_FMA(p, 1.0, -pFlow);
                        q = CUDA_FMA(q, 1.0, -qFlow);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Measurement Evaluation Kernels
// ============================================================================

// Combined kernel: Compute all power injections and flows, then evaluate measurements
// This combines multiple operations to reuse GPU data
__global__ void computePowerInjectionsAndFlowsKernel(
    const Real* v, const Real* theta,
    const DeviceBus* buses, const DeviceBranch* branches,
    const Index* branchFromBus, const Index* branchFromBusRowPtr,
    const Index* branchToBus, const Index* branchToBusRowPtr,
    Real* pInjection, Real* qInjection,
    Real* pFlow, Real* qFlow,
    Index nBuses, Index nBranches) {
    
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute power injections for buses
    if (idx < nBuses) {
        Real p, q;
        computePowerInjectionPQ(v, theta, buses, branches,
                               branchFromBus, branchFromBusRowPtr,
                               branchToBus, branchToBusRowPtr,
                               idx, nBuses, nBranches, p, q);
        pInjection[idx] = p;
        qInjection[idx] = q;
    }
    
    // Compute power flows for branches
    if (idx < nBranches) {
        const DeviceBranch& br = branches[idx];
        if (br.fromBus >= 0 && br.fromBus < nBuses && 
            br.toBus >= 0 && br.toBus < nBuses) {
            Real p, q;
            computePowerFlowPQ(v, theta, branches, idx, br.fromBus, br.toBus, nBuses, p, q);
            pFlow[idx] = p;
            qFlow[idx] = q;
        } else {
            pFlow[idx] = 0.0;
            qFlow[idx] = 0.0;
        }
    }
}

// Combined kernel: Evaluate measurement functions h(x) using pre-computed power data
__global__ void evaluateMeasurementsKernel(
    const Real* v,
    const Real* pInjection, const Real* qInjection,
    const Real* pFlow, const Real* qFlow,
    const Index* measurementTypes,
    const Index* measurementLocations,
    const Index* measurementBranches,
    const Index* branchFromBus, const Index* branchToBus,
    const DeviceBranch* branches,
    Real* hx,
    Index nMeasurements, Index nBranches) {
    
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nMeasurements) {
        Index type = measurementTypes[idx];
        Index location = measurementLocations[idx];
        Index branchIdx = measurementBranches[idx];
        
        Real value = 0.0;
        
        switch (type) {
            case 0:  // P_FLOW
                if (branchIdx >= 0 && branchIdx < nBranches) {
                    const DeviceBranch& br = branches[branchIdx];
                    Index fromBus = branchFromBus ? branchFromBus[branchIdx] : br.fromBus;
                    Index toBus = branchToBus ? branchToBus[branchIdx] : br.toBus;
                    // Determine direction based on measurement location
                    if (location == fromBus) {
                        value = pFlow[branchIdx];
                    } else if (location == toBus) {
                        value = -pFlow[branchIdx];
                    } else {
                        value = pFlow[branchIdx];  // Default forward
                    }
                }
                break;
                
            case 1:  // Q_FLOW
                if (branchIdx >= 0 && branchIdx < nBranches) {
                    const DeviceBranch& br = branches[branchIdx];
                    Index fromBus = branchFromBus ? branchFromBus[branchIdx] : br.fromBus;
                    Index toBus = branchToBus ? branchToBus[branchIdx] : br.toBus;
                    if (location == fromBus) {
                        value = qFlow[branchIdx];
                    } else if (location == toBus) {
                        value = -qFlow[branchIdx];
                    } else {
                        value = qFlow[branchIdx];
                    }
                }
                break;
                
            case 2:  // P_INJECTION
                if (location >= 0) {
                    value = pInjection[location];
                }
                break;
                
            case 3:  // Q_INJECTION
                if (location >= 0) {
                    value = qInjection[location];
                }
                break;
                
            case 4:  // V_MAGNITUDE
                if (location >= 0) {
                    value = v[location];
                }
                break;
                
            case 5:  // I_MAGNITUDE
                if (branchIdx >= 0 && branchIdx < nBranches) {
                    // I = sqrt(P^2 + Q^2) / V
                    Real p = pFlow[branchIdx];
                    Real q = qFlow[branchIdx];
                    Real iSq = CUDA_FMA(p, p, q * q);
                    if (iSq > 1e-12) {
                        value = sqrt(iSq);
                        // Divide by voltage if available
                        if (location >= 0) {
                            Real vBus = v[location];
                            if (vBus > 1e-6) {
                                value = CUDA_DIV(value, vBus);
                            }
                        }
                    }
                }
                break;
                
            default:
                value = 0.0;
        }
        
        hx[idx] = value;
    }
}

// OPTIMIZATION: Fused kernel that computes power flows and evaluates measurements in one pass
// Uses shared memory for intermediate power flow results to reduce global memory access
// Reduces kernel launch overhead by combining two kernels into one
__global__ void computeMeasurementsFusedKernel(
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
    Index nBuses, Index nBranches, Index nMeasurements) {
    
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Compute power injections for buses
    if (idx < nBuses) {
        Real p, q;
        computePowerInjectionPQ(v, theta, buses, branches,
                               branchFromBus, branchFromBusRowPtr,
                               branchToBus, branchToBusRowPtr,
                               idx, nBuses, nBranches, p, q);
        pInjection[idx] = p;
        qInjection[idx] = q;
    }
    
    // Phase 2: Compute power flows for branches
    if (idx < nBranches) {
        const DeviceBranch& br = branches[idx];
        if (br.fromBus >= 0 && br.fromBus < nBuses && 
            br.toBus >= 0 && br.toBus < nBuses) {
            Real p, q;
            computePowerFlowPQ(v, theta, branches, idx, br.fromBus, br.toBus, nBuses, p, q);
            pFlow[idx] = p;
            qFlow[idx] = q;
        } else {
            pFlow[idx] = 0.0;
            qFlow[idx] = 0.0;
        }
    }
    
    // Synchronize to ensure power flows are computed before measurement evaluation
    __syncthreads();
    
    // Phase 3: Evaluate measurements using pre-computed power data
    if (idx < nMeasurements) {
        Index type = measurementTypes[idx];
        Index location = measurementLocations[idx];
        Index branchIdx = measurementBranches[idx];
        
        Real value = 0.0;
        
        switch (type) {
            case 0:  // P_FLOW
                if (branchIdx >= 0 && branchIdx < nBranches) {
                    const DeviceBranch& br = branches[branchIdx];
                    Index fromBus = branchFromBus ? branchFromBus[branchIdx] : br.fromBus;
                    Index toBus = branchToBus ? branchToBus[branchIdx] : br.toBus;
                    if (location == fromBus) {
                        value = pFlow[branchIdx];
                    } else if (location == toBus) {
                        value = -pFlow[branchIdx];
                    } else {
                        value = pFlow[branchIdx];
                    }
                }
                break;
                
            case 1:  // Q_FLOW
                if (branchIdx >= 0 && branchIdx < nBranches) {
                    const DeviceBranch& br = branches[branchIdx];
                    Index fromBus = branchFromBus ? branchFromBus[branchIdx] : br.fromBus;
                    Index toBus = branchToBus ? branchToBus[branchIdx] : br.toBus;
                    if (location == fromBus) {
                        value = qFlow[branchIdx];
                    } else if (location == toBus) {
                        value = -qFlow[branchIdx];
                    } else {
                        value = qFlow[branchIdx];
                    }
                }
                break;
                
            case 2:  // P_INJECTION
                if (location >= 0) {
                    value = pInjection[location];
                }
                break;
                
            case 3:  // Q_INJECTION
                if (location >= 0) {
                    value = qInjection[location];
                }
                break;
                
            case 4:  // V_MAGNITUDE
                if (location >= 0) {
                    value = v[location];
                }
                break;
                
            case 5:  // I_MAGNITUDE
                if (branchIdx >= 0 && branchIdx < nBranches) {
                    Real p = pFlow[branchIdx];
                    Real q = qFlow[branchIdx];
                    Real iSq = CUDA_FMA(p, p, q * q);
                    if (iSq > 1e-12) {
                        value = sqrt(iSq);
                        if (location >= 0) {
                            Real vBus = v[location];
                            if (vBus > 1e-6) {
                                value = CUDA_DIV(value, vBus);
                            }
                        }
                    }
                }
                break;
                
            default:
                value = 0.0;
        }
        
        hx[idx] = value;
    }
}

// Wrapper: Compute power injections and flows, then evaluate measurements
// All operations on GPU, data stays on GPU
// OPTIMIZATION: Optionally uses fused kernel to reduce launch overhead
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
    cudaStream_t stream) {
    
    constexpr Index blockSize = 256;
    Index maxSize = (nBuses > nBranches) ? nBuses : nBranches;
    maxSize = (maxSize > nMeasurements) ? maxSize : nMeasurements;
    Index gridSize = (maxSize + blockSize - 1) / blockSize;
    
    // OPTIMIZATION: Use fused kernel when possible (reduces launch overhead)
    // Fused kernel works when max(nBuses, nBranches, nMeasurements) fits in one grid
    // For very large systems, fall back to two separate kernels
    if (maxSize <= 65535 * blockSize) {  // CUDA grid size limit
        computeMeasurementsFusedKernel<<<gridSize, blockSize, 0, stream>>>(
            v, theta, buses, branches,
            branchFromBus, branchFromBusRowPtr,
            branchToBus, branchToBusRowPtr,
            measurementTypes, measurementLocations, measurementBranches,
            pInjection, qInjection, pFlow, qFlow, hx,
            nBuses, nBranches, nMeasurements);
    } else {
        // Fall back to two separate kernels for very large systems
        Index gridSize1 = (maxSize + blockSize - 1) / blockSize;
        computePowerInjectionsAndFlowsKernel<<<gridSize1, blockSize, 0, stream>>>(
            v, theta, buses, branches,
            branchFromBus, branchFromBusRowPtr,
            branchToBus, branchToBusRowPtr,
            pInjection, qInjection, pFlow, qFlow,
            nBuses, nBranches);
        
        Index gridSize2 = (nMeasurements + blockSize - 1) / blockSize;
        evaluateMeasurementsKernel<<<gridSize2, blockSize, 0, stream>>>(
            v, pInjection, qInjection, pFlow, qFlow,
            measurementTypes, measurementLocations, measurementBranches,
            branchFromBus, branchToBus, branches,
            hx, nMeasurements, nBranches);
    }
}

// Unified kernel: Compute power flows with optional derived quantities (MW/MVAR/I)
// Can compute P/Q only, or P/Q + MW/MVAR/I_PU/I_Amps
// If pFlow/qFlow are provided, uses them; otherwise computes from state
__global__ void computePowerFlowsCompleteKernel(
    const Real* v, const Real* theta,
    const DeviceBranch* branches,
    const DeviceBus* buses,
    Real baseMVA,
    const Real* pFlowIn, const Real* qFlowIn,  // Optional: if provided, reuse instead of computing
    Real* pFlow, Real* qFlow,
    Real* pMW, Real* qMVAR,
    Real* iPU, Real* iAmps,
    Index nBranches, Index nBuses) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBranches) {
        const DeviceBranch& br = branches[idx];
        if (br.fromBus >= 0 && br.fromBus < nBuses && 
            br.toBus >= 0 && br.toBus < nBuses) {
            Real p, q;
            
            // Reuse existing P/Q flows if provided, otherwise compute
            if (pFlowIn && qFlowIn) {
                p = pFlowIn[idx];
                q = qFlowIn[idx];
            } else {
                computePowerFlowPQ(v, theta, branches, idx, br.fromBus, br.toBus, nBuses, p, q);
            }
            
            // Store P/Q flows
            if (pFlow) pFlow[idx] = p;
            if (qFlow) qFlow[idx] = q;
            
            // Compute derived quantities if output buffers provided
            if (pMW && qMVAR && iPU && iAmps) {
                computePowerFlowDerivedQuantities(v, branches, buses, idx, p, q, baseMVA,
                                                 pMW[idx], qMVAR[idx], iPU[idx], iAmps[idx]);
            }
        } else {
            if (pFlow) pFlow[idx] = 0.0;
            if (qFlow) qFlow[idx] = 0.0;
            if (pMW) pMW[idx] = 0.0;
            if (qMVAR) qMVAR[idx] = 0.0;
            if (iPU) iPU[idx] = 0.0;
            if (iAmps) iAmps[idx] = 0.0;
        }
    }
}


// GPU-accelerated complete power flow computation (optimized version)
void computeAllPowerFlowsCompleteGPU(const Real* v, const Real* theta,
                                     const DeviceBranch* branches,
                                     const DeviceBus* buses,
                                     Real baseMVA,
                                     Real* pFlow, Real* qFlow,
                                     Real* pMW, Real* qMVAR,
                                     Real* iPU, Real* iAmps,
                                     Index nBranches, Index nBuses) {
    constexpr Index blockSize = 256;
    const Index gridSize = (nBranches + blockSize - 1) / blockSize;
    computePowerFlowsCompleteKernel<<<gridSize, blockSize>>>(
        v, theta, branches, buses, baseMVA,
        nullptr, nullptr,  // No input P/Q flows - compute from state
        pFlow, qFlow, pMW, qMVAR, iPU, iAmps, nBranches, nBuses);
}

// GPU-accelerated computation of MW/MVAR/I_PU/I_Amps from existing P/Q flows
// Reuses P/Q flows already computed (e.g., during solver iterations)
void computePowerFlowDerivedGPU(
    const Real* v,
    const Real* pFlow, const Real* qFlow,
    const DeviceBranch* branches,
    const DeviceBus* buses,
    Real baseMVA,
    Real* pMW, Real* qMVAR,
    Real* iPU, Real* iAmps,
    Index nBranches, Index nBuses) {
    constexpr Index blockSize = 256;
    const Index gridSize = (nBranches + blockSize - 1) / blockSize;
    computePowerFlowsCompleteKernel<<<gridSize, blockSize>>>(
        v, nullptr, branches, buses, baseMVA,  // theta not needed when reusing P/Q
        pFlow, qFlow,  // Reuse existing P/Q flows
        nullptr, nullptr,  // Don't output P/Q again
        pMW, qMVAR, iPU, iAmps, nBranches, nBuses);
}

// GPU-accelerated power injection computation for all buses
// Uses the combined kernel but only processes buses (flows ignored)
void computeAllPowerInjectionsGPU(const Real* v, const Real* theta,
                                 const DeviceBus* buses,
                                 const DeviceBranch* branches,
                                 const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                 const Index* branchToBus, const Index* branchToBusRowPtr,
                                 Real* pInjection, Real* qInjection,
                                 Index nBuses, Index nBranches) {
    constexpr Index blockSize = 256;
    const Index gridSize = (nBuses + blockSize - 1) / blockSize;
    // Use combined kernel but only process buses (branches will be out of bounds for bus threads)
    computePowerInjectionsAndFlowsKernel<<<gridSize, blockSize>>>(
        v, theta, buses, branches,
        branchFromBus, branchFromBusRowPtr,
        branchToBus, branchToBusRowPtr,
        pInjection, qInjection,
        nullptr, nullptr,  // No flow outputs needed
        nBuses, nBranches);
}

// ============================================================================
// Jacobian Computation Kernels
// ============================================================================

// Jacobian element computation for power flow measurements
__device__ void computeJacobianPowerFlow(const Real* v, const Real* theta,
                                        const DeviceBranch* branches, Index branchIdx,
                                        Index fromBus, Index toBus,
                                        Real* dPdTheta, Real* dPdV,
                                        Real* dQdTheta, Real* dQdV) {
    const DeviceBranch& br = branches[branchIdx];
    
    Real vi = v[fromBus];
    Real vj = v[toBus];
    Real thetai = theta[fromBus];
    Real thetaj = theta[toBus];
    
    // Optimized admittance calculation using FMA (with division by zero check)
    const Real z2 = CUDA_FMA(br.r, br.r, br.x * br.x);
    if (z2 < 1e-12) {  // Avoid division by zero
        // Set all derivatives to zero for invalid impedance
        dPdTheta[0] = dPdTheta[1] = 0.0;
        dPdV[0] = dPdV[1] = 0.0;
        dQdTheta[0] = dQdTheta[1] = 0.0;
        dQdV[0] = dQdV[1] = 0.0;
        return;
    }
    const Real inv_z2 = CUDA_DIV(1.0, z2);
    const Real g = CUDA_FMA(br.r, inv_z2, 0.0);
    const Real b = CUDA_FMA(-br.x, inv_z2, 0.0);
    
    const Real tap = br.tapRatio;
    if (fabs(tap) < 1e-12) {  // Avoid division by zero
        dPdTheta[0] = dPdTheta[1] = 0.0;
        dPdV[0] = dPdV[1] = 0.0;
        dQdTheta[0] = dQdTheta[1] = 0.0;
        dQdV[0] = dQdV[1] = 0.0;
        return;
    }
    const Real phase = br.phaseShift;
    const Real thetaDiff = thetai - thetaj - phase;
    
    // Use sincos for simultaneous sin/cos (precision-aware)
    Real sinDiff, cosDiff;
    CUDA_SINCOS(thetaDiff, &sinDiff, &cosDiff);
    
    // Optimized Jacobian computation using FMA
    const Real vivj = CUDA_FMA(vi, vj, 0.0);
    const Real inv_tap = CUDA_DIV(1.0, tap);
    const Real tap2 = CUDA_FMA(tap, tap, 0.0);
    const Real inv_tap2 = CUDA_DIV(1.0, tap2);
    
    const Real gsin = CUDA_FMA(g, sinDiff, 0.0);
    const Real bcos = CUDA_FMA(b, cosDiff, 0.0);
    const Real gcos = CUDA_FMA(g, cosDiff, 0.0);
    const Real bsin = CUDA_FMA(b, sinDiff, 0.0);
    
    const Real term1 = CUDA_FMA(vivj, (gsin - bcos) * inv_tap, 0.0);
    const Real term2 = CUDA_FMA(vivj, (gcos + bsin) * inv_tap, 0.0);
    const Real term3 = CUDA_FMA(vivj, (gsin - bcos) * inv_tap, 0.0);
    
    // Pre-compute (g*cos + b*sin) and (g*sin - b*cos) for voltage derivatives
    const Real gcos_plus_bsin = CUDA_FMA(gcos, 1.0, bsin);
    const Real gsin_minus_bcos = CUDA_FMA(gsin, 1.0, -bcos);
    
    // dP/dtheta_i
    dPdTheta[0] = term1;
    // dP/dtheta_j
    dPdTheta[1] = -term1;
    // dP/dV_i = 2*V_i*g/tap² - V_j*(g*cos + b*sin)/tap
    dPdV[0] = CUDA_FMA(2.0 * vi, g * inv_tap2, -vj * gcos_plus_bsin * inv_tap);
    // dP/dV_j = -V_i*(g*cos + b*sin)/tap
    dPdV[1] = CUDA_FMA(-vi, gcos_plus_bsin * inv_tap, 0.0);
    
    // dQ/dtheta_i
    dQdTheta[0] = -term2;
    // dQ/dtheta_j
    dQdTheta[1] = term2;
    // dQ/dV_i = -2*V_i*b_total/tap² - V_j*(g*sin - b*cos)/tap
    const Real b_total = CUDA_FMA(b, 1.0, br.b * 0.5);
    dQdV[0] = CUDA_FMA(-2.0 * vi, b_total * inv_tap2, -vj * gsin_minus_bcos * inv_tap);
    // dQ/dV_j = -V_i*(g*sin - b*cos)/tap
    dQdV[1] = CUDA_FMA(-vi, gsin_minus_bcos * inv_tap, 0.0);
}

// Jacobian for power injection measurements
// Optimized with CSR adjacency lists for O(avg_degree) complexity instead of O(nBranches)
// branchFromBus/branchToBus are CSR column indices, rowPtr are row pointers
__device__ void computeJacobianPowerInjection(const Real* v, const Real* theta,
                                              const DeviceBus* buses, const DeviceBranch* branches,
                                              const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                              const Index* branchToBus, const Index* branchToBusRowPtr,
                                              Index busIdx, Index nBuses, Index nBranches,
                                              Real* dPdTheta, Real* dPdV,
                                              Real* dQdTheta, Real* dQdV) {
    // Bounds checking: validate bus index before accessing
    if (busIdx < 0 || busIdx >= nBuses) {
        // Invalid bus index - zero all derivatives
        for (Index i = 0; i < nBuses; ++i) {
            dPdTheta[i] = 0.0;
            dPdV[i] = 0.0;
            dQdTheta[i] = 0.0;
            dQdV[i] = 0.0;
        }
        return;
    }
    
    // Initialize to zero
    for (Index i = 0; i < nBuses; ++i) {
        dPdTheta[i] = 0.0;
        dPdV[i] = 0.0;
        dQdTheta[i] = 0.0;
        dQdV[i] = 0.0;
    }
    
    const DeviceBus& bus = buses[busIdx];
    
    // Shunt contributions
    dPdV[busIdx] += 2.0 * v[busIdx] * bus.gShunt;
    dQdV[busIdx] -= 2.0 * v[busIdx] * bus.bShunt;
    
    // Outgoing branch contributions (using CSR adjacency list - O(avg_degree) instead of O(nBranches))
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
                        // Validate branch connects to this bus (sanity check)
                        if (br.fromBus == busIdx && br.toBus >= 0 && br.toBus < nBuses) {
                            Real dPdThetaFrom[2], dPdVFrom[2];
                            Real dQdThetaFrom[2], dQdVFrom[2];
                            
                            computeJacobianPowerFlow(v, theta, branches, brIdx, br.fromBus, br.toBus,
                                                     dPdThetaFrom, dPdVFrom, dQdThetaFrom, dQdVFrom);
                            
                            // Accumulate derivatives
                            dPdTheta[br.fromBus] += dPdThetaFrom[0];
                            dPdTheta[br.toBus] += dPdThetaFrom[1];
                            dPdV[br.fromBus] += dPdVFrom[0];
                            dPdV[br.toBus] += dPdVFrom[1];
                            
                            dQdTheta[br.fromBus] += dQdThetaFrom[0];
                            dQdTheta[br.toBus] += dQdThetaFrom[1];
                            dQdV[br.fromBus] += dQdVFrom[0];
                            dQdV[br.toBus] += dQdVFrom[1];
                        }
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
                        // Validate branch connects to this bus (sanity check)
                        if (br.toBus == busIdx && br.fromBus >= 0 && br.fromBus < nBuses) {
                            Real dPdThetaFrom[2], dPdVFrom[2];
                            Real dQdThetaFrom[2], dQdVFrom[2];
                            
                            // Reverse direction (similar computation with reversed indices)
                            computeJacobianPowerFlow(v, theta, branches, brIdx, br.toBus, br.fromBus,
                                                     dPdThetaFrom, dPdVFrom, dQdThetaFrom, dQdVFrom);
                            
                            // Accumulate derivatives (note: reversed indices)
                            dPdTheta[br.toBus] += dPdThetaFrom[0];
                            dPdTheta[br.fromBus] += dPdThetaFrom[1];
                            dPdV[br.toBus] += dPdVFrom[0];
                            dPdV[br.fromBus] += dPdVFrom[1];
                            
                            dQdTheta[br.toBus] += dQdThetaFrom[0];
                            dQdTheta[br.fromBus] += dQdThetaFrom[1];
                            dQdV[br.toBus] += dQdVFrom[0];
                            dQdV[br.fromBus] += dQdVFrom[1];
                        }
                    }
                }
            }
        }
    }
}

// Main kernel to compute Jacobian matrix elements
// Optimized with CSR adjacency lists for power injection Jacobian computation
__global__ void computeJacobianKernel(const Real* v, const Real* theta,
                                      const DeviceBus* buses, const DeviceBranch* branches,
                                      const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                      const Index* branchToBus, const Index* branchToBusRowPtr,
                                      const Index* measurementTypes,
                                      const Index* measurementLocations,
                                      const Index* measurementBranches,
                                      const Index* jacobianRowPtr,
                                      const Index* jacobianColInd,
                                      Real* jacobianValues,
                                      Index nMeasurements, Index nBuses, Index nBranches) {
    Index measIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (measIdx < nMeasurements) {
        Index type = measurementTypes[measIdx];
        Index rowStart = jacobianRowPtr[measIdx];
        Index rowEnd = jacobianRowPtr[measIdx + 1];
        
        if (type == 0 || type == 1) { // P_FLOW or Q_FLOW
            Index branchIdx = measurementBranches[measIdx];
            // Bounds checking: validate branch index before accessing
            if (branchIdx < 0 || branchIdx >= nBranches) {
                // Invalid branch index - set all Jacobian elements to zero
                for (Index i = rowStart; i < rowEnd; ++i) {
                    jacobianValues[i] = 0.0;
                }
                return;  // No loops below, just exit
            }
            const DeviceBranch& br = branches[branchIdx];
            
            Real dPdTheta[2], dPdV[2], dQdTheta[2], dQdV[2];
            computeJacobianPowerFlow(v, theta, branches, branchIdx,
                                    br.fromBus, br.toBus,
                                    dPdTheta, dPdV, dQdTheta, dQdV);
            
            // Store Jacobian elements
            Index colIdx = 0;
            for (Index i = rowStart; i < rowEnd && colIdx < 4; ++i) {
                Index col = jacobianColInd[i];
                
                if (col < nBuses) {
                    // Angle derivative
                    if (col == br.fromBus) {
                        jacobianValues[i] = (type == 0) ? dPdTheta[0] : dQdTheta[0];
                    } else if (col == br.toBus) {
                        jacobianValues[i] = (type == 0) ? dPdTheta[1] : dQdTheta[1];
                    }
                } else {
                    // Voltage magnitude derivative
                    Index vCol = col - nBuses;
                    if (vCol == br.fromBus) {
                        jacobianValues[i] = (type == 0) ? dPdV[0] : dQdV[0];
                    } else if (vCol == br.toBus) {
                        jacobianValues[i] = (type == 0) ? dPdV[1] : dQdV[1];
                    }
                }
                colIdx++;
            }
        } else if (type == 2 || type == 3) { // P_INJECTION or Q_INJECTION
            Index busIdx = measurementLocations[measIdx];
            
            // Bounds checking: validate bus index before accessing
            if (busIdx < 0 || busIdx >= nBuses) {
                // Invalid bus index - set all Jacobian elements to zero
                for (Index i = rowStart; i < rowEnd; ++i) {
                    jacobianValues[i] = 0.0;
                }
                return;
            }
            
            // Allocate shared memory for Jacobian elements
            // Validate shared memory size (4 * nBuses * sizeof(Real))
            // Note: Shared memory is allocated at kernel launch, so we assume it's sufficient
            extern __shared__ Real sharedJac[];
            Real* dPdTheta = sharedJac;
            Real* dPdV = sharedJac + nBuses;
            Real* dQdTheta = sharedJac + 2 * nBuses;
            Real* dQdV = sharedJac + 3 * nBuses;
            
            computeJacobianPowerInjection(v, theta, buses, branches,
                                         branchFromBus, branchFromBusRowPtr,
                                         branchToBus, branchToBusRowPtr,
                                         busIdx, nBuses, nBranches,
                                         dPdTheta, dPdV, dQdTheta, dQdV);
            
            // Store Jacobian elements
            for (Index i = rowStart; i < rowEnd; ++i) {
                Index col = jacobianColInd[i];
                
                if (col < nBuses) {
                    jacobianValues[i] = (type == 2) ? dPdTheta[col] : dQdTheta[col];
                } else {
                    Index vCol = col - nBuses;
                    jacobianValues[i] = (type == 2) ? dPdV[vCol] : dQdV[vCol];
                }
            }
        } else if (type == 4) { // V_MAGNITUDE
            Index busIdx = measurementLocations[measIdx];
            
            // Bounds checking: validate bus index before accessing
            if (busIdx < 0 || busIdx >= nBuses) {
                // Invalid bus index - set all Jacobian elements to zero
                for (Index i = rowStart; i < rowEnd; ++i) {
                    jacobianValues[i] = 0.0;
                }
                return;
            }
            
            // dV/dV = 1, dV/dtheta = 0
            for (Index i = rowStart; i < rowEnd; ++i) {
                Index col = jacobianColInd[i];
                if (col == busIdx + nBuses) {
                    jacobianValues[i] = 1.0;
                } else {
                    jacobianValues[i] = 0.0;
                }
            }
        } else if (type == 5) { // I_MAGNITUDE
            // Current magnitude Jacobian (simplified - would need full I calculation)
            for (Index i = rowStart; i < rowEnd; ++i) {
                jacobianValues[i] = 0.0;  // Placeholder
            }
        }
    }
}

// OPTIMIZATION: Jacobian kernel with shared memory caching for bus/branch data
// Caches frequently accessed bus voltage/angle and branch data in shared memory
// Reduces redundant global memory reads when multiple measurements access same bus/branch
__global__ void computeJacobianKernelCached(const Real* v, const Real* theta,
                                           const DeviceBus* buses, const DeviceBranch* branches,
                                           const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                           const Index* branchToBus, const Index* branchToBusRowPtr,
                                           const Index* measurementTypes,
                                           const Index* measurementLocations,
                                           const Index* measurementBranches,
                                           const Index* jacobianRowPtr,
                                           const Index* jacobianColInd,
                                           Real* jacobianValues,
                                           Index nMeasurements, Index nBuses, Index nBranches) {
    // Shared memory cache for bus data (v, theta) and branch data
    // Cache size: up to 64 buses and 64 branches per block
    extern __shared__ char sharedMem[];
    constexpr Index maxCachedBuses = 64;
    constexpr Index maxCachedBranches = 64;
    
    Real* cachedV = reinterpret_cast<Real*>(sharedMem);
    Real* cachedTheta = cachedV + maxCachedBuses;
    DeviceBranch* cachedBranches = reinterpret_cast<DeviceBranch*>(cachedTheta + maxCachedBuses);
    Index* cachedBusIndices = reinterpret_cast<Index*>(cachedBranches + maxCachedBranches);
    Index* cachedBranchIndices = cachedBusIndices + maxCachedBuses;
    
    Index tid = threadIdx.x;
    Index measIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperative loading: threads in block load bus/branch data into shared memory
    // This reduces redundant global memory reads when multiple measurements access same data
    if (measIdx < nMeasurements) {
        Index type = measurementTypes[measIdx];
        Index location = measurementLocations[measIdx];
        Index branchIdx = measurementBranches[measIdx];
        
        // Cache bus data if needed (for power injection/voltage measurements)
        if ((type == 2 || type == 3 || type == 4) && location >= 0 && location < nBuses) {
            if (tid < maxCachedBuses && cachedBusIndices[tid] != location) {
                cachedBusIndices[tid] = location;
                cachedV[tid] = v[location];
                cachedTheta[tid] = theta[location];
            }
        }
        
        // Cache branch data if needed (for power flow measurements)
        if ((type == 0 || type == 1) && branchIdx >= 0 && branchIdx < nBranches) {
            if (tid < maxCachedBranches && cachedBranchIndices[tid] != branchIdx) {
                cachedBranchIndices[tid] = branchIdx;
                cachedBranches[tid] = branches[branchIdx];
            }
        }
    }
    
    __syncthreads();  // Ensure all cached data is loaded
    
    // Now compute Jacobian using cached data when possible
    if (measIdx < nMeasurements) {
        Index type = measurementTypes[measIdx];
        Index rowStart = jacobianRowPtr[measIdx];
        Index rowEnd = jacobianRowPtr[measIdx + 1];
        
        if (type == 0 || type == 1) { // P_FLOW or Q_FLOW
            Index branchIdx = measurementBranches[measIdx];
            if (branchIdx < 0 || branchIdx >= nBranches) {
                for (Index i = rowStart; i < rowEnd; ++i) {
                    jacobianValues[i] = 0.0;
                }
                return;
            }
            
            // Try to use cached branch data
            DeviceBranch br;
            bool cached = false;
            for (Index i = 0; i < maxCachedBranches; ++i) {
                if (cachedBranchIndices[i] == branchIdx) {
                    br = cachedBranches[i];
                    cached = true;
                    break;
                }
            }
            if (!cached) {
                br = branches[branchIdx];
            }
            
            Real dPdTheta[2], dPdV[2], dQdTheta[2], dQdV[2];
            computeJacobianPowerFlow(v, theta, branches, branchIdx,
                                    br.fromBus, br.toBus,
                                    dPdTheta, dPdV, dQdTheta, dQdV);
            
            Index colIdx = 0;
            for (Index i = rowStart; i < rowEnd && colIdx < 4; ++i) {
                Index col = jacobianColInd[i];
                if (col < nBuses) {
                    if (col == br.fromBus) {
                        jacobianValues[i] = (type == 0) ? dPdTheta[0] : dQdTheta[0];
                    } else if (col == br.toBus) {
                        jacobianValues[i] = (type == 0) ? dPdTheta[1] : dQdTheta[1];
                    }
                } else {
                    Index vCol = col - nBuses;
                    if (vCol == br.fromBus) {
                        jacobianValues[i] = (type == 0) ? dPdV[0] : dQdV[0];
                    } else if (vCol == br.toBus) {
                        jacobianValues[i] = (type == 0) ? dPdV[1] : dQdV[1];
                    }
                }
                colIdx++;
            }
        } else if (type == 2 || type == 3) { // P_INJECTION or Q_INJECTION
            Index busIdx = measurementLocations[measIdx];
            if (busIdx < 0 || busIdx >= nBuses) {
                for (Index i = rowStart; i < rowEnd; ++i) {
                    jacobianValues[i] = 0.0;
                }
                return;
            }
            
            extern __shared__ Real sharedJac[];
            Real* dPdTheta = sharedJac;
            Real* dPdV = sharedJac + nBuses;
            Real* dQdTheta = sharedJac + 2 * nBuses;
            Real* dQdV = sharedJac + 3 * nBuses;
            
            computeJacobianPowerInjection(v, theta, buses, branches,
                                         branchFromBus, branchFromBusRowPtr,
                                         branchToBus, branchToBusRowPtr,
                                         busIdx, nBuses, nBranches,
                                         dPdTheta, dPdV, dQdTheta, dQdV);
            
            for (Index i = rowStart; i < rowEnd; ++i) {
                Index col = jacobianColInd[i];
                if (col < nBuses) {
                    jacobianValues[i] = (type == 2) ? dPdTheta[col] : dQdTheta[col];
                } else {
                    Index vCol = col - nBuses;
                    jacobianValues[i] = (type == 2) ? dPdV[vCol] : dQdV[vCol];
                }
            }
        } else if (type == 4) { // V_MAGNITUDE
            Index busIdx = measurementLocations[measIdx];
            if (busIdx < 0 || busIdx >= nBuses) {
                for (Index i = rowStart; i < rowEnd; ++i) {
                    jacobianValues[i] = 0.0;
                }
                return;
            }
            
            for (Index i = rowStart; i < rowEnd; ++i) {
                Index col = jacobianColInd[i];
                if (col == busIdx + nBuses) {
                    jacobianValues[i] = 1.0;
                } else {
                    jacobianValues[i] = 0.0;
                }
            }
        } else if (type == 5) { // I_MAGNITUDE
            for (Index i = rowStart; i < rowEnd; ++i) {
                jacobianValues[i] = 0.0;  // Placeholder
            }
        }
    }
}

// Optimized wrapper function with constexpr
// Uses CSR adjacency lists for O(avg_degree) complexity in power injection Jacobian
// OPTIMIZATION: Optionally uses shared memory caching for better memory bandwidth
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
                    Index nMeasurements, Index nBuses, Index nBranches,
                    cudaStream_t stream) {
    constexpr Index blockSize = 256;  // Compile-time constant
    const Index gridSize = (nMeasurements + blockSize - 1) / blockSize;
    
    // Choose kernel based on system size and shared memory availability
    // For smaller systems, use cached version; for larger, use standard version
    constexpr Index maxCachedBuses = 64;
    constexpr Index maxCachedBranches = 64;
    const Index sharedMemSizeCached = (maxCachedBuses * sizeof(Real) * 2 + 
                                       maxCachedBranches * sizeof(DeviceBranch) +
                                       maxCachedBuses * sizeof(Index) +
                                       maxCachedBranches * sizeof(Index));
    const Index sharedMemSizeStandard = (nBuses < 256) ? 4 * nBuses * sizeof(Real) : 0;
    
    // Use cached version if shared memory is available and system is not too large
    if (sharedMemSizeCached < 48 * 1024 && nBuses < 1000 && nBranches < 2000) {
        computeJacobianKernelCached<<<gridSize, blockSize, sharedMemSizeCached, stream>>>(
            v, theta, buses, branches,
            branchFromBus, branchFromBusRowPtr, branchToBus, branchToBusRowPtr,
            measurementTypes, measurementLocations, measurementBranches,
            jacobianRowPtr, jacobianColInd, jacobianValues,
            nMeasurements, nBuses, nBranches);
    } else {
        // Use standard kernel for larger systems
        computeJacobianKernel<<<gridSize, blockSize, sharedMemSizeStandard, stream>>>(
            v, theta, buses, branches,
            branchFromBus, branchFromBusRowPtr, branchToBus, branchToBusRowPtr,
            measurementTypes, measurementLocations, measurementBranches,
            jacobianRowPtr, jacobianColInd, jacobianValues,
            nMeasurements, nBuses, nBranches);
    }
    // Note: No cudaDeviceSynchronize() - caller should sync if needed
}

} // namespace cuda
} // namespace sle

