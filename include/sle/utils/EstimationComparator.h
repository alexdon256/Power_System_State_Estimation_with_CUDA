/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_UTILS_ESTIMATIONCOMPARATOR_H
#define SLE_UTILS_ESTIMATIONCOMPARATOR_H

#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <string>
#include <vector>

namespace sle {
namespace utils {

struct MeasurementComparison {
    std::string deviceId;
    Real measuredValue;
    Real estimatedValue;
    Real residual;
    Real relativeError;
};

class EstimationComparator {
public:
    // Compare measured vs. estimated values
    static std::vector<MeasurementComparison> compare(
        const model::TelemetryData& telemetry,
        const model::StateVector& state,
        const model::NetworkModel& network);
    
    // Generate comparison report
    static std::string generateReport(
        const std::vector<MeasurementComparison>& comparisons);
};

} // namespace utils
} // namespace sle

#endif // SLE_UTILS_ESTIMATIONCOMPARATOR_H

