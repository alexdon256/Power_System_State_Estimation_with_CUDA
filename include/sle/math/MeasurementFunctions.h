/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_MEASUREMENTFUNCTIONS_H
#define SLE_MATH_MEASUREMENTFUNCTIONS_H

#include <sle/Types.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <vector>

namespace sle {
namespace math {

class MeasurementFunctions {
public:
    MeasurementFunctions();
    ~MeasurementFunctions();
    
    // Evaluate measurement functions h(x)
    void evaluate(const StateVector& state, const NetworkModel& network,
                  const TelemetryData& telemetry, std::vector<Real>& hx);
    
    // Evaluate on GPU
    void evaluateGPU(const StateVector& state, const NetworkModel& network,
                     const TelemetryData& telemetry, std::vector<Real>& hx);
    
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

