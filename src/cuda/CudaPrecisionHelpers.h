/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Precision-aware CUDA intrinsics helper macros
 */

#ifndef SLE_CUDA_PRECISION_HELPERS_H
#define SLE_CUDA_PRECISION_HELPERS_H

#include <cuda_runtime.h>
#include <sle/Types.h>
#include <cmath>

namespace sle {
namespace cuda {

// Precision-aware intrinsics: automatically use correct precision based on Real type
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
    // Device code
    #if __CUDA_ARCH__ >= 600
        // Compute capability 6.0+ supports double precision FMA
        #ifdef USE_DOUBLE_PRECISION
            // Double precision intrinsics
            #define CUDA_FMA(a, b, c) __fma_rn((a), (b), (c))
            #define CUDA_SINCOS(x, s, c) __sincos((x), (s), (c))
            #define CUDA_DIV(a, b) ((a) / (b))  // Regular division for double
        #else
            // Single precision intrinsics
            #define CUDA_FMA(a, b, c) __fmaf_rn((a), (b), (c))
            #define CUDA_SINCOS(x, s, c) __sincosf((x), (s), (c))
            #define CUDA_DIV(a, b) __fdividef((a), (b))
        #endif
    #else
        // Older architectures: use regular operations
        #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
        #define CUDA_SINCOS(x, s, c) do { *(s) = sin(x); *(c) = cos(x); } while(0)
        #define CUDA_DIV(a, b) ((a) / (b))
    #endif
#else
    // Host code: use regular operations
    #define CUDA_FMA(a, b, c) ((a) * (b) + (c))
    #define CUDA_SINCOS(x, s, c) do { *(s) = sin(x); *(c) = cos(x); } while(0)
    #define CUDA_DIV(a, b) ((a) / (b))
#endif

// Check if Real is double (at compile time)
template<typename T> struct is_double { static constexpr bool value = false; };
template<> struct is_double<double> { static constexpr bool value = true; };

// Runtime precision check (for device code)
__device__ __forceinline__ bool isDoublePrecision() {
    return sizeof(Real) == sizeof(double);
}

// Safe division with zero check
__device__ __forceinline__ Real safeDivide(Real a, Real b, Real defaultValue = 0.0) {
    if (fabs(b) < 1e-12) {
        return defaultValue;
    }
    return CUDA_DIV(a, b);
}

// Safe sqrt with non-negative check
__device__ __forceinline__ Real safeSqrt(Real x) {
    if (x < 0.0) {
        return 0.0;  // Return 0 for negative values
    }
    return sqrt(x);
}

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_PRECISION_HELPERS_H

