/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_OPTIMIZEDMATH_H
#define SLE_MATH_OPTIMIZEDMATH_H

#include <sle/Types.h>
#include <sle/utils/TemplateHelpers.h>
#include <cmath>
#include <algorithm>

namespace sle {
namespace math {

// Template-based optimized mathematical operations
template<typename T = Real>
class OptimizedMath {
public:
    // Fast admittance computation with template specialization
    template<bool UseFMA = true>
    static inline T computeAdmittanceReal(T r, T x) {
        if constexpr (UseFMA) {
            // Use fused multiply-add when available
            const T z2 = r * r + x * x;
            return r / z2;
        } else {
            return utils::FastMath<T>::computeAdmittanceReal(r, x);
        }
    }
    
    template<bool UseFMA = true>
    static inline T computeAdmittanceImag(T r, T x) {
        if constexpr (UseFMA) {
            const T z2 = r * r + x * x;
            return -x / z2;
        } else {
            return utils::FastMath<T>::computeAdmittanceImag(r, x);
        }
    }
    
    // Vectorized operations with compile-time size
    template<size_t N>
    static inline void computeAdmittanceBatch(const T* r, const T* x, T* g, T* b) {
        utils::FastMath<T>::template computeAdmittanceBatch<N>(r, x, g, b);
    }
    
    // Fast norm computation
    template<size_t N>
    static inline T computeNorm(const T* vec) {
        T sum = T(0);
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            sum += vec[i] * vec[i];
        }
        return std::sqrt(sum);
    }
    
    // Fast dot product with unrolling
    template<size_t N>
    static inline T dotProduct(const T* a, const T* b) {
        T sum = T(0);
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
};

// Specialized version for small fixed-size vectors
template<typename T, size_t N>
struct VectorOps {
    static inline T norm(const T* vec) {
        return OptimizedMath<T>::template computeNorm<N>(vec);
    }
    
    static inline T dot(const T* a, const T* b) {
        return OptimizedMath<T>::template dotProduct<N>(a, b);
    }
    
    static inline void add(const T* a, const T* b, T* result) {
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    static inline void scale(const T* vec, T factor, T* result) {
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = vec[i] * factor;
        }
    }
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_OPTIMIZEDMATH_H

