/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/baddata/DataConsistencyChecker.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <cmath>
#include <set>
#include <string>

namespace sle {
namespace baddata {

DataConsistencyChecker::DataConsistencyChecker() {
}

ConsistencyCheckResult DataConsistencyChecker::checkConsistency(
    const model::TelemetryData& telemetry,
    const model::NetworkModel& network) {
    
    ConsistencyCheckResult result;
    result.isConsistent = true;
    
    // Check measurement ranges
    if (!validateMeasurementRanges(telemetry)) {
        result.isConsistent = false;
        result.inconsistencies.push_back("Some measurements are out of range");
    }
    
    // Check for duplicates
    checkDuplicates(telemetry, result);
    
    // Check critical measurements
    if (!checkCriticalMeasurements(telemetry, network)) {
        result.warnings.push_back("Some critical measurements may be missing");
    }
    
    return result;
}

bool DataConsistencyChecker::validateMeasurementRanges(
    const model::TelemetryData& telemetry) {
    
    auto measurements = telemetry.getMeasurements();
    
    for (const auto* meas : measurements) {
        if (meas && !isMeasurementInRange(meas->getType(), meas->getValue())) {
            return false;
        }
    }
    
    return true;
}

bool DataConsistencyChecker::checkCriticalMeasurements(
    const model::TelemetryData& telemetry,
    const model::NetworkModel& network) {
    
    // Check if reference bus has voltage measurement
    BusId refBus = network.getReferenceBus();
    if (refBus >= 0) {
        bool hasRefVoltage = false;
        auto measurements = telemetry.getMeasurements();
        
        for (const auto* meas : measurements) {
            if (meas && meas->getType() == MeasurementType::V_MAGNITUDE &&
                meas->getLocation() == refBus) {
                hasRefVoltage = true;
                break;
            }
        }
        
        if (!hasRefVoltage) {
            return false;
        }
    }
    
    return true;
}

bool DataConsistencyChecker::isMeasurementInRange(MeasurementType type, Real value) {
    switch (type) {
        case MeasurementType::V_MAGNITUDE:
            return value >= 0.8 && value <= 1.2;  // 0.8-1.2 p.u.
        case MeasurementType::P_FLOW:
        case MeasurementType::P_INJECTION:
        case MeasurementType::Q_FLOW:
        case MeasurementType::Q_INJECTION:
            return std::abs(value) <= 10.0;  // Reasonable power range
        case MeasurementType::I_MAGNITUDE:
            return value >= 0.0 && value <= 5.0;  // Reasonable current range
        default:
            return true;
    }
}

void DataConsistencyChecker::checkDuplicates(
    const model::TelemetryData& telemetry,
    ConsistencyCheckResult& result) {
    
    std::set<std::string> deviceIds;
    auto measurements = telemetry.getMeasurements();
    
    for (const auto* meas : measurements) {
        if (!meas) continue;
        const std::string& deviceId = meas->getDeviceId();
        if (deviceIds.count(deviceId) > 0) {
            result.warnings.push_back("Duplicate device ID: " + deviceId);
        }
        deviceIds.insert(deviceId);
    }
}

} // namespace baddata
} // namespace sle

