/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/utils/EstimationComparator.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/model/NetworkModel.h>
#include <sle/math/MeasurementFunctions.h>
#include <sle/Types.h>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace sle {
namespace utils {

std::vector<MeasurementComparison> EstimationComparator::compare(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network) {
    
    std::vector<MeasurementComparison> comparisons;
    
    math::MeasurementFunctions measFuncs;
    std::vector<Real> hx;
    measFuncs.evaluate(state, network, telemetry, hx);
    
    const auto& measurements = telemetry.getMeasurements();
    
    for (size_t i = 0; i < measurements.size() && i < hx.size(); ++i) {
        MeasurementComparison comp;
        comp.deviceId = measurements[i]->getDeviceId();
        comp.measuredValue = measurements[i]->getValue();
        comp.estimatedValue = hx[i];
        comp.residual = comp.measuredValue - comp.estimatedValue;
        comp.relativeError = std::abs(comp.residual) / (std::abs(comp.measuredValue) + 1e-6);
        comparisons.push_back(comp);
    }
    
    return comparisons;
}

std::string EstimationComparator::generateReport(
    const std::vector<MeasurementComparison>& comparisons) {
    
    std::ostringstream oss;
    oss << "Measured vs. Estimated Comparison\n";
    oss << "==================================\n\n";
    
    for (const auto& comp : comparisons) {
        oss << "Device: " << comp.deviceId << "\n";
        oss << "  Measured: " << comp.measuredValue << "\n";
        oss << "  Estimated: " << comp.estimatedValue << "\n";
        oss << "  Residual: " << comp.residual << "\n";
        oss << "  Relative Error: " << comp.relativeError * 100.0 << "%\n\n";
    }
    
    return oss.str();
}

} // namespace utils
} // namespace sle

