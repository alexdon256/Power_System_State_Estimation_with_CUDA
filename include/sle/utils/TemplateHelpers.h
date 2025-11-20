/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_UTILS_TEMPLATEHELPERS_H
#define SLE_UTILS_TEMPLATEHELPERS_H

#include <sle/Types.h>
#include <type_traits>
#include <cmath>

namespace sle {
namespace utils {

// Compile-time power calculation
template<int N>
struct Power {
    template<typename T>
    static constexpr T compute(T x) {
        return x * Power<N-1>::compute(x);
    }
};

template<>
struct Power<1> {
    template<typename T>
    static constexpr T compute(T x) {
        return x;
    }
};

template<>
struct Power<0> {
    template<typename T>
    static constexpr T compute(T) {
        return T(1);
    }
};

// Compile-time selection of precision
template<bool UseDouble>
struct PrecisionType {
    using type = double;
};

template<>
struct PrecisionType<false> {
    using type = float;
};

// Fast inverse square root using template metaprogramming
template<typename T>
struct FastMath {
    // Optimized admittance calculation: Y = 1/Z where Z = R + jX
    // Computes: g = R/(R²+X²), b = -X/(R²+X²)
    static inline T computeAdmittanceReal(T r, T x) {
        const T z2 = r * r + x * x;
        // Use reciprocal instead of division for better performance
        const T inv_z2 = T(1.0) / z2;
        return r * inv_z2;
    }
    
    static inline T computeAdmittanceImag(T r, T x) {
        const T z2 = r * r + x * x;
        const T inv_z2 = T(1.0) / z2;
        return -x * inv_z2;
    }
    
    // Vectorized operations
    template<size_t N>
    static inline void computeAdmittanceBatch(const T* r, const T* x, T* g, T* b) {
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            const T z2 = r[i] * r[i] + x[i] * x[i];
            const T inv_z2 = T(1.0) / z2;
            g[i] = r[i] * inv_z2;
            b[i] = -x[i] * inv_z2;
        }
    }
};

// Compile-time loop unrolling
template<size_t N, typename Func>
struct Unroll {
    static inline void execute(Func&& f) {
        Unroll<N-1, Func>::execute(std::forward<Func>(f));
        f(N-1);
    }
};

template<typename Func>
struct Unroll<0, Func> {
    static inline void execute(Func&&) {}
};

// Type traits for measurement types
template<MeasurementType Type>
struct MeasurementTraits;

template<>
struct MeasurementTraits<MeasurementType::P_FLOW> {
    static constexpr bool isFlow = true;
    static constexpr bool isInjection = false;
    static constexpr bool isVoltage = false;
    static constexpr bool isCurrent = false;
};

template<>
struct MeasurementTraits<MeasurementType::Q_FLOW> {
    static constexpr bool isFlow = true;
    static constexpr bool isInjection = false;
    static constexpr bool isVoltage = false;
    static constexpr bool isCurrent = false;
};

template<>
struct MeasurementTraits<MeasurementType::P_INJECTION> {
    static constexpr bool isFlow = false;
    static constexpr bool isInjection = true;
    static constexpr bool isVoltage = false;
    static constexpr bool isCurrent = false;
};

template<>
struct MeasurementTraits<MeasurementType::V_MAGNITUDE> {
    static constexpr bool isFlow = false;
    static constexpr bool isInjection = false;
    static constexpr bool isVoltage = true;
    static constexpr bool isCurrent = false;
};

// Compile-time constant extraction
template<Real Value>
struct Constant {
    static constexpr Real value = Value;
    using type = Real;
};

// Compile-time conditional compilation
template<bool Condition>
using EnableIf = std::enable_if_t<Condition>;

} // namespace utils
} // namespace sle

#endif // SLE_UTILS_TEMPLATEHELPERS_H

