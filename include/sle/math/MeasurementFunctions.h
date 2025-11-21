/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_MEASUREMENTFUNCTIONS_H
#define SLE_MATH_MEASUREMENTFUNCTIONS_H

#include <sle/Types.h>
#include <vector>
#include <memory>

// Forward declarations
namespace sle {
namespace model {
    class StateVector;
    class NetworkModel;
    class TelemetryData;
}
}

namespace sle {
namespace math {

class MeasurementFunctions {
public:
    MeasurementFunctions();
    ~MeasurementFunctions();
    
    // Evaluate measurement functions h(x)
    void evaluate(const model::StateVector& state, const model::NetworkModel& network,
                  const model::TelemetryData& telemetry, std::vector<Real>& hx);
    
    // Evaluate on GPU
    void evaluateGPU(const model::StateVector& state, const model::NetworkModel& network,
                     const model::TelemetryData& telemetry, std::vector<Real>& hx);
    
    // Compute residual r = z - h(x)
    void computeResidual(const std::vector<Real>& z, const std::vector<Real>& hx,
                        std::vector<Real>& residual);
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_MEASUREMENTFUNCTIONS_H

