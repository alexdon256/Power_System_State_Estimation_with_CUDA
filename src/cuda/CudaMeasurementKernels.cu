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
        // Compute capability 6.0+ supports double precision FMA
        // Since Real = double, use double precision intrinsics
        #define CUDA_FMA(a, b, c) __fma_rn((a), (b), (c))
        #define CUDA_SINCOS(x, s, c) __sincos((x), (s), (c))
        #define CUDA_DIV(a, b) ((a) / (b))  // Regular division for double (no fast div)
    #else
        // Older architectures: use regular operations
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

// Device data structures
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

// Optimized measurement function kernels with FMA and fast math
__device__ __forceinline__ Real computePowerFlowP(const Real* v, const Real* theta,
                                   const DeviceBranch* branches, Index branchIdx,
                                   Index fromBus, Index toBus, Index nBuses) {
    const DeviceBranch& br = branches[branchIdx];
    
    // Coalesced memory access
    const Real vi = v[fromBus];
    const Real vj = v[toBus];
    const Real thetai = theta[fromBus];
    const Real thetaj = theta[toBus];
    
    // Optimized admittance calculation using FMA (with division by zero check)
    const Real z2 = CUDA_FMA(br.r, br.r, br.x * br.x);
    if (z2 < 1e-12) {  // Avoid division by zero (very small impedance)
        return 0.0;
    }
    const Real inv_z2 = CUDA_DIV(1.0, z2);
    const Real g = CUDA_FMA(br.r, inv_z2, 0.0);  // br.r * inv_z2
    const Real b = CUDA_FMA(-br.x, inv_z2, 0.0);  // -br.x * inv_z2
    
    const Real tap = br.tapRatio;
    if (fabs(tap) < 1e-12) {  // Avoid division by zero (tap ratio too small)
        return 0.0;
    }
    const Real phase = br.phaseShift;
    const Real thetaDiff = thetai - thetaj - phase;
    
    // Use sincos for simultaneous sin/cos (precision-aware)
    Real sinDiff, cosDiff;
    CUDA_SINCOS(thetaDiff, &sinDiff, &cosDiff);
    
    // Optimized power calculation using FMA
    const Real tap2 = CUDA_FMA(tap, tap, 0.0);  // tap * tap
    const Real inv_tap2 = CUDA_DIV(1.0, tap2);
    const Real vi2 = CUDA_FMA(vi, vi, 0.0);  // vi * vi
    const Real vivj = CUDA_FMA(vi, vj, 0.0);  // vi * vj
    const Real inv_tap = CUDA_DIV(1.0, tap);
    
    const Real term1 = CUDA_FMA(vi2, g * inv_tap2, 0.0);  // vi^2 * g / tap^2
    const Real gcos = CUDA_FMA(g, cosDiff, 0.0);
    const Real bsin = CUDA_FMA(b, sinDiff, 0.0);
    const Real term2 = CUDA_FMA(vivj, (gcos + bsin) * inv_tap, 0.0);  // vi*vj*(g*cos+b*sin)/tap
    
    const Real p = CUDA_FMA(term1, 1.0, -term2);  // term1 - term2
    
    return p;
}

__device__ __forceinline__ Real computePowerFlowQ(const Real* v, const Real* theta,
                                  const DeviceBranch* branches, Index branchIdx,
                                  Index fromBus, Index toBus, Index nBuses) {
    const DeviceBranch& br = branches[branchIdx];
    
    // Coalesced memory access
    const Real vi = v[fromBus];
    const Real vj = v[toBus];
    const Real thetai = theta[fromBus];
    const Real thetaj = theta[toBus];
    
    // Optimized admittance calculation using FMA (with division by zero check)
    const Real z2 = CUDA_FMA(br.r, br.r, br.x * br.x);
    if (z2 < 1e-12) {  // Avoid division by zero
        return 0.0;
    }
    const Real inv_z2 = CUDA_DIV(1.0, z2);
    const Real g = CUDA_FMA(br.r, inv_z2, 0.0);
    const Real b = CUDA_FMA(-br.x, inv_z2, 0.0);
    
    const Real tap = br.tapRatio;
    if (fabs(tap) < 1e-12) {  // Avoid division by zero
        return 0.0;
    }
    const Real phase = br.phaseShift;
    const Real thetaDiff = thetai - thetaj - phase;
    
    // Use sincos for simultaneous sin/cos (precision-aware)
    Real sinDiff, cosDiff;
    CUDA_SINCOS(thetaDiff, &sinDiff, &cosDiff);
    
    // Optimized reactive power calculation using FMA
    const Real tap2 = CUDA_FMA(tap, tap, 0.0);
    const Real inv_tap2 = CUDA_DIV(1.0, tap2);
    const Real vi2 = CUDA_FMA(vi, vi, 0.0);
    const Real vivj = CUDA_FMA(vi, vj, 0.0);
    const Real inv_tap = CUDA_DIV(1.0, tap);
    
    const Real b_total = CUDA_FMA(b, 1.0, br.b * 0.5);  // b + br.b/2
    const Real term1 = CUDA_FMA(vi2, b_total * inv_tap2, 0.0);  // vi^2 * (b + br.b/2) / tap^2
    const Real gsin = CUDA_FMA(g, sinDiff, 0.0);
    const Real bcos = CUDA_FMA(b, cosDiff, 0.0);
    const Real term2 = CUDA_FMA(vivj, (gsin - bcos) * inv_tap, 0.0);  // vi*vj*(g*sin-b*cos)/tap
    
    const Real q = CUDA_FMA(-term1, 1.0, -term2);  // -term1 - term2
    
    return q;
}

__device__ __forceinline__ Real computePowerInjectionP(const Real* v, const Real* theta,
                                       const DeviceBus* buses, const DeviceBranch* branches,
                                       Index busIdx, Index nBuses, Index nBranches) {
    Real p = 0.0;
    const DeviceBus& bus = buses[busIdx];
    
    // Shunt contribution using FMA
    const Real vi2 = CUDA_FMA(v[busIdx], v[busIdx], 0.0);
    p = CUDA_FMA(vi2, bus.gShunt, 0.0);
    
    // Branch contributions (loop can be unrolled by compiler for small nBranches)
    #pragma unroll 4
    for (Index i = 0; i < nBranches; ++i) {
        const DeviceBranch& br = branches[i];
        if (br.fromBus == busIdx) {
            p += computePowerFlowP(v, theta, branches, i, br.fromBus, br.toBus, nBuses);
        } else if (br.toBus == busIdx) {
            // Reverse direction - optimized
            const Real vi = v[br.toBus];
            const Real vj = v[br.fromBus];
            const Real thetai = theta[br.toBus];
            const Real thetaj = theta[br.fromBus];
            
            // Optimized admittance calculation using FMA (with division by zero check)
            const Real z2 = CUDA_FMA(br.r, br.r, br.x * br.x);
            if (z2 < 1e-12) continue;  // Skip if impedance too small
            const Real inv_z2 = CUDA_DIV(1.0, z2);
            const Real g = CUDA_FMA(br.r, inv_z2, 0.0);
            const Real b = CUDA_FMA(-br.x, inv_z2, 0.0);
            const Real tap = br.tapRatio;
            if (fabs(tap) < 1e-12) continue;  // Skip if tap too small
            const Real phase = br.phaseShift;
            const Real thetaDiff = thetai - thetaj + phase;  // Note: reversed
            
            // Use sincos for simultaneous sin/cos (precision-aware)
            Real sinDiff, cosDiff;
            CUDA_SINCOS(thetaDiff, &sinDiff, &cosDiff);
            
            // Optimized power calculation using FMA
            const Real tap2 = CUDA_FMA(tap, tap, 0.0);
            const Real inv_tap2 = CUDA_DIV(1.0, tap2);
            const Real vi2_rev = CUDA_FMA(vi, vi, 0.0);
            const Real vivj_rev = CUDA_FMA(vi, vj, 0.0);
            const Real inv_tap = CUDA_DIV(1.0, tap);
            
            const Real term1 = CUDA_FMA(vi2_rev, g * inv_tap2, 0.0);
            const Real gcos = CUDA_FMA(g, cosDiff, 0.0);
            const Real bsin = CUDA_FMA(b, sinDiff, 0.0);
            const Real term2 = CUDA_FMA(vivj_rev, (gcos + bsin) * inv_tap, 0.0);
            
            p += CUDA_FMA(term1, 1.0, -term2);
        }
    }
    
    return p;
}

__device__ __forceinline__ Real computePowerInjectionQ(const Real* v, const Real* theta,
                                      const DeviceBus* buses, const DeviceBranch* branches,
                                      Index busIdx, Index nBuses, Index nBranches) {
    Real q = 0.0;
    const DeviceBus& bus = buses[busIdx];
    
    // Shunt contribution using FMA
    const Real vi2 = CUDA_FMA(v[busIdx], v[busIdx], 0.0);
    q = CUDA_FMA(-vi2, bus.bShunt, 0.0);
    
    // Branch contributions (loop can be unrolled by compiler)
    #pragma unroll 4
    for (Index i = 0; i < nBranches; ++i) {
        const DeviceBranch& br = branches[i];
        if (br.fromBus == busIdx) {
            q += computePowerFlowQ(v, theta, branches, i, br.fromBus, br.toBus, nBuses);
        } else if (br.toBus == busIdx) {
            // Reverse direction Q calculation - optimized
            const Real vi = v[br.toBus];
            const Real vj = v[br.fromBus];
            const Real thetai = theta[br.toBus];
            const Real thetaj = theta[br.fromBus];
            
            // Optimized admittance calculation using FMA (with division by zero check)
            const Real z2 = CUDA_FMA(br.r, br.r, br.x * br.x);
            if (z2 < 1e-12) continue;  // Skip if impedance too small
            const Real inv_z2 = CUDA_DIV(1.0, z2);
            const Real g = CUDA_FMA(br.r, inv_z2, 0.0);
            const Real b = CUDA_FMA(-br.x, inv_z2, 0.0);
            const Real tap = br.tapRatio;
            if (fabs(tap) < 1e-12) continue;  // Skip if tap too small
            const Real phase = br.phaseShift;
            const Real thetaDiff = thetai - thetaj + phase;
            
            // Use sincos for simultaneous sin/cos (precision-aware)
            Real sinDiff, cosDiff;
            CUDA_SINCOS(thetaDiff, &sinDiff, &cosDiff);
            
            // Optimized reactive power calculation using FMA
            const Real tap2 = CUDA_FMA(tap, tap, 0.0);
            const Real inv_tap2 = CUDA_DIV(1.0, tap2);
            const Real vi2_rev = CUDA_FMA(vi, vi, 0.0);
            const Real vivj_rev = CUDA_FMA(vi, vj, 0.0);
            const Real inv_tap = CUDA_DIV(1.0, tap);
            
            const Real b_total = CUDA_FMA(b, 1.0, br.b * 0.5);
            const Real term1 = CUDA_FMA(vi2_rev, b_total * inv_tap2, 0.0);
            const Real gsin = CUDA_FMA(g, sinDiff, 0.0);
            const Real bcos = CUDA_FMA(b, cosDiff, 0.0);
            const Real term2 = CUDA_FMA(vivj_rev, (gsin - bcos) * inv_tap, 0.0);
            
            q += CUDA_FMA(-term1, 1.0, -term2);
        }
    }
    
    return q;
}

// Main kernel to evaluate all measurement functions
__global__ void evaluateMeasurementsKernel(const Real* v, const Real* theta,
                                           const DeviceBus* buses, const DeviceBranch* branches,
                                           const Index* measurementTypes,
                                           const Index* measurementLocations,
                                           const Index* measurementBranches,
                                           Real* hx, Index nMeasurements, Index nBuses, Index nBranches) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nMeasurements) {
        Index type = measurementTypes[idx];
        Real result = 0.0;
        
        switch (type) {
            case 0: { // P_FLOW
                Index branchIdx = measurementBranches[idx];
                const DeviceBranch& br = branches[branchIdx];
                result = computePowerFlowP(v, theta, branches, branchIdx, br.fromBus, br.toBus, nBuses);
                break;
            }
            case 1: { // Q_FLOW
                Index branchIdx = measurementBranches[idx];
                const DeviceBranch& br = branches[branchIdx];
                result = computePowerFlowQ(v, theta, branches, branchIdx, br.fromBus, br.toBus, nBuses);
                break;
            }
            case 2: { // P_INJECTION
                Index busIdx = measurementLocations[idx];
                result = computePowerInjectionP(v, theta, buses, branches, busIdx, nBuses, nBranches);
                break;
            }
            case 3: { // Q_INJECTION
                Index busIdx = measurementLocations[idx];
                result = computePowerInjectionQ(v, theta, buses, branches, busIdx, nBuses, nBranches);
                break;
            }
            case 4: { // V_MAGNITUDE
                Index busIdx = measurementLocations[idx];
                result = v[busIdx];
                break;
            }
            case 5: { // I_MAGNITUDE
                // Current magnitude calculation (optimized)
                Index branchIdx = measurementBranches[idx];
                const DeviceBranch& br = branches[branchIdx];
                const Real vi = v[br.fromBus];
                const Real vj = v[br.toBus];
                const Real thetai = theta[br.fromBus];
                const Real thetaj = theta[br.toBus];
                
                // Optimized admittance calculation using FMA (with division by zero check)
                const Real z2 = CUDA_FMA(br.r, br.r, br.x * br.x);
                if (z2 < 1e-12 || fabs(vi) < 1e-12) {  // Avoid division by zero
                    result = 0.0;
                    break;
                }
                const Real inv_z2 = CUDA_DIV(1.0, z2);
                const Real g = CUDA_FMA(br.r, inv_z2, 0.0);
                const Real b = CUDA_FMA(-br.x, inv_z2, 0.0);
                const Real tap = br.tapRatio;
                if (fabs(tap) < 1e-12) {
                    result = 0.0;
                    break;
                }
                const Real phase = br.phaseShift;
                const Real thetaDiff = thetai - thetaj - phase;
                
                // Use sincos for simultaneous sin/cos (precision-aware)
                Real sinDiff, cosDiff;
                CUDA_SINCOS(thetaDiff, &sinDiff, &cosDiff);
                
                // Optimized power calculation using FMA
                const Real tap2 = CUDA_FMA(tap, tap, 0.0);
                const Real inv_tap2 = CUDA_DIV(1.0, tap2);
                const Real vi2 = CUDA_FMA(vi, vi, 0.0);
                const Real vivj = CUDA_FMA(vi, vj, 0.0);
                const Real inv_tap = CUDA_DIV(1.0, tap);
                
                const Real gcos = CUDA_FMA(g, cosDiff, 0.0);
                const Real bsin = CUDA_FMA(b, sinDiff, 0.0);
                const Real gsin = CUDA_FMA(g, sinDiff, 0.0);
                const Real bcos = CUDA_FMA(b, cosDiff, 0.0);
                
                const Real p = CUDA_FMA(vi2, g * inv_tap2, -vivj * (gcos + bsin) * inv_tap);
                const Real b_total = CUDA_FMA(b, 1.0, br.b * 0.5);
                const Real q = CUDA_FMA(-vi2, b_total * inv_tap2, -vivj * (gsin - bcos) * inv_tap);
                
                // Compute current magnitude: I = sqrt(P^2 + Q^2) / V
                const Real s2 = CUDA_FMA(p, p, q * q);
                const Real s = sqrt(s2);  // Use regular sqrt for double precision
                result = CUDA_DIV(s, vi);  // I = S / V
                break;
            }
            default:
                result = 0.0;
        }
        
        hx[idx] = result;
    }
}

// Optimized wrapper function with constexpr
void evaluateMeasurements(const Real* v, const Real* theta,
                         const DeviceBus* buses, const DeviceBranch* branches,
                         const Index* measurementTypes,
                         const Index* measurementLocations,
                         const Index* measurementBranches,
                         Real* hx, Index nMeasurements, Index nBuses, Index nBranches) {
    constexpr Index blockSize = 256;  // Compile-time constant for better optimization
    const Index gridSize = (nMeasurements + blockSize - 1) / blockSize;
    
    evaluateMeasurementsKernel<<<gridSize, blockSize>>>(
        v, theta, buses, branches,
        measurementTypes, measurementLocations, measurementBranches,
        hx, nMeasurements, nBuses, nBranches);
    // Note: No cudaDeviceSynchronize() - caller should sync if needed
    // This allows overlapping with other operations
}

} // namespace cuda
} // namespace sle

