/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_IO_COMPARISONREPORT_H
#define SLE_IO_COMPARISONREPORT_H

#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/model/NetworkModel.h>
#include <string>
#include <vector>

namespace sle {
namespace io {

struct MeasurementComparison {
    std::string deviceId;
    MeasurementType type;
    Real measuredValue;
    Real estimatedValue;
    Real residual;
    Real normalizedResidual;
    bool isBad;
};

class ComparisonReport {
public:
    static std::vector<MeasurementComparison> compare(
        const model::TelemetryData& telemetry,
        const model::StateVector& state,
        const model::NetworkModel& network);
    
    static std::string generateReport(const std::vector<MeasurementComparison>& comparisons);
    static void writeReport(const std::string& filepath, 
                           const std::vector<MeasurementComparison>& comparisons);
};

} // namespace io
} // namespace sle

#endif // SLE_IO_COMPARISONREPORT_H

