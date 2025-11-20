/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_UTILS_COMPILETIMECONFIG_H
#define SLE_UTILS_COMPILETIMECONFIG_H

#include <type_traits>

namespace sle {
namespace config {

// Compile-time configuration flags
struct CompileTimeConfig {
    // Precision selection
    static constexpr bool UseDoublePrecision = true;
    using RealType = std::conditional_t<UseDoublePrecision, double, float>;
    
    // CUDA optimization flags
    static constexpr bool UseFMA = true;
    static constexpr int DefaultBlockSize = 256;
    
    // Algorithm selection
    static constexpr bool UseIncrementalEstimation = true;
    static constexpr bool UseBadDataDetection = true;
    static constexpr bool UseObservabilityCheck = true;
    
    // Performance tuning
    static constexpr int MaxUnrollSize = 8;
    static constexpr bool EnableVectorization = true;
};

// Template-based configuration
template<typename Config = CompileTimeConfig>
struct ConfigTraits {
    using Real = typename Config::RealType;
    static constexpr int BlockSize = Config::DefaultBlockSize;
};

} // namespace config
} // namespace sle

#endif // SLE_UTILS_COMPILETIMECONFIG_H

