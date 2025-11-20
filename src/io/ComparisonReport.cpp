/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/io/ComparisonReport.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/model/NetworkModel.h>
#include <sle/math/MeasurementFunctions.h>
#include <fstream>
#include <cmath>

namespace sle {
namespace io {

std::vector<MeasurementComparison> ComparisonReport::compare(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network) {
    
    std::vector<MeasurementComparison> comparisons;
    
    // Evaluate measurement functions to get estimated values
    math::MeasurementFunctions measFuncs;
    std::vector<Real> hx;
    measFuncs.evaluate(state, network, telemetry, hx);
    
    const auto& measurements = telemetry.getMeasurements();
    
    for (size_t i = 0; i < measurements.size() && i < hx.size(); ++i) {
        MeasurementComparison comp;
        comp.deviceId = measurements[i]->getDeviceId();
        comp.type = measurements[i]->getType();
        comp.measuredValue = measurements[i]->getValue();
        comp.estimatedValue = hx[i];
        comp.residual = comp.measuredValue - comp.estimatedValue;
        
        // Normalized residual
        Real stdDev = measurements[i]->getStdDev();
        comp.normalizedResidual = std::abs(comp.residual) / stdDev;
        
        // Bad data threshold (typically 3-5 sigma)
        comp.isBad = comp.normalizedResidual > 3.0;
        
        comparisons.push_back(comp);
    }
    
    return comparisons;
}

std::string ComparisonReport::generateReport(
    const std::vector<MeasurementComparison>& comparisons) {
    
    std::ostringstream oss;
    oss << "Measurement Comparison Report\n";
    oss << "============================\n\n";
    
    int badCount = 0;
    for (const auto& comp : comparisons) {
        oss << "Device: " << comp.deviceId << "\n";
        oss << "  Type: " << static_cast<int>(comp.type) << "\n";
        oss << "  Measured: " << comp.measuredValue << "\n";
        oss << "  Estimated: " << comp.estimatedValue << "\n";
        oss << "  Residual: " << comp.residual << "\n";
        oss << "  Normalized Residual: " << comp.normalizedResidual << "\n";
        oss << "  Status: " << (comp.isBad ? "BAD" : "OK") << "\n\n";
        
        if (comp.isBad) badCount++;
    }
    
    oss << "Summary: " << badCount << " bad measurements out of " 
        << comparisons.size() << " total\n";
    
    return oss.str();
}

void ComparisonReport::writeReport(const std::string& filepath,
                                  const std::vector<MeasurementComparison>& comparisons) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    
    file << generateReport(comparisons);
}

} // namespace io
} // namespace sle

