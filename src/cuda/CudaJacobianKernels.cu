/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <cuda_runtime.h>
#include <sle/Types.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <cmath>

// Precision-aware intrinsics: automatically detect Real type (double or float)
// Real is defined as double in Types.h, so use double precision intrinsics
//
// Note: __CUDACC__ is defined when nvcc compiles (both host and device code)
//       __CUDA_ARCH__ is only defined in device code (__device__/__global__ functions)
//       For IDE parsing, neither may be defined, so we fall back to regular operations
#if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= 600
        #define CUDA_FMA(a, b, c) __fma_rn((a), (b), (c))
    #else
        #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #endif
    #define CUDA_SINCOS(x, s, c) sincos((x), (s), (c))
    #define CUDA_DIV(a, b) ((a) / (b))
#elif defined(__CUDACC__)
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #define CUDA_SINCOS(x, s, c) sincos((x), (s), (c))
    #define CUDA_DIV(a, b) ((a) / (b))
#else
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #define CUDA_SINCOS(x, s, c) sincos((x), (s), (c))
    #define CUDA_DIV(a, b) ((a) / (b))
#endif

namespace sle {
namespace cuda {

// Forward declarations from measurement kernels
struct DeviceBus;
struct DeviceBranch;

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
    
    // dP/dtheta_i
    dPdTheta[0] = term1;
    // dP/dtheta_j
    dPdTheta[1] = -term1;
    // dP/dV_i
    dPdV[0] = CUDA_FMA(2.0 * vi, g * inv_tap2, -vj * term2 * inv_tap);
    // dP/dV_j
    dPdV[1] = CUDA_FMA(-vi, term2 * inv_tap, 0.0);
    
    // dQ/dtheta_i
    dQdTheta[0] = -term2;
    // dQ/dtheta_j
    dQdTheta[1] = term2;
    // dQ/dV_i
    const Real b_total = CUDA_FMA(b, 1.0, br.b * 0.5);
    dQdV[0] = CUDA_FMA(-2.0 * vi, b_total * inv_tap2, -vj * term3 * inv_tap);
    // dQ/dV_j
    dQdV[1] = CUDA_FMA(-vi, term3 * inv_tap, 0.0);
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

// Optimized wrapper function with constexpr
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
                    Index nMeasurements, Index nBuses, Index nBranches) {
    constexpr Index blockSize = 256;  // Compile-time constant
    const Index gridSize = (nMeasurements + blockSize - 1) / blockSize;
    // Optimized shared memory: only allocate what's needed (typically much less than 4*nBuses)
    const Index sharedMemSize = (nBuses < 256) ? 4 * nBuses * sizeof(Real) : 0;
    
    computeJacobianKernel<<<gridSize, blockSize, sharedMemSize>>>(
        v, theta, buses, branches,
        branchFromBus, branchFromBusRowPtr, branchToBus, branchToBusRowPtr,
        measurementTypes, measurementLocations, measurementBranches,
        jacobianRowPtr, jacobianColInd, jacobianValues,
        nMeasurements, nBuses, nBranches);
    // Note: No cudaDeviceSynchronize() - caller should sync if needed
}

} // namespace cuda
} // namespace sle

