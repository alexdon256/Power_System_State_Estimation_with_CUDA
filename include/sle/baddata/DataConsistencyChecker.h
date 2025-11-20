/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_BADDATA_DATACONSISTENCYCHECKER_H
#define SLE_BADDATA_DATACONSISTENCYCHECKER_H

#include <sle/model/TelemetryData.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <string>
#include <vector>

namespace sle {
namespace baddata {

struct ConsistencyCheckResult {
    bool isConsistent;
    std::vector<std::string> inconsistencies;
    std::vector<std::string> warnings;
};

class DataConsistencyChecker {
public:
    DataConsistencyChecker();
    
    // Check data consistency before estimation
    ConsistencyCheckResult checkConsistency(
        const model::TelemetryData& telemetry,
        const model::NetworkModel& network);
    
    // Validate measurement ranges
    bool validateMeasurementRanges(const model::TelemetryData& telemetry);
    
    // Check for missing critical measurements
    bool checkCriticalMeasurements(const model::TelemetryData& telemetry,
                                   const model::NetworkModel& network);
    
private:
    // Check if measurements are within reasonable ranges
    bool isMeasurementInRange(MeasurementType type, Real value);
    
    // Check for duplicate measurements
    void checkDuplicates(const model::TelemetryData& telemetry,
                        ConsistencyCheckResult& result);
};

} // namespace baddata
} // namespace sle

#endif // SLE_BADDATA_DATACONSISTENCYCHECKER_H

