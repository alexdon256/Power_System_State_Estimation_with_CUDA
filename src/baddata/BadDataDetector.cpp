/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/baddata/BadDataDetector.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/model/NetworkModel.h>
#include <sle/math/MeasurementFunctions.h>
#include <cmath>
#include <algorithm>

namespace sle {
namespace baddata {

BadDataDetector::BadDataDetector() : normalizedResidualThreshold_(3.0) {
}

BadDataResult BadDataDetector::detectBadDataChiSquare(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network) {
    
    BadDataResult result;
    result.hasBadData = false;
    
    // Evaluate measurement functions
    math::MeasurementFunctions measFuncs;
    std::vector<Real> hx;
    measFuncs.evaluate(state, network, telemetry, hx);
    
    // Compute residuals and weights
    std::vector<Real> residuals;
    std::vector<Real> weights;
    const auto& measurements = telemetry.getMeasurements();
    
    for (size_t i = 0; i < measurements.size() && i < hx.size(); ++i) {
        Real residual = measurements[i]->getValue() - hx[i];
        residuals.push_back(residual);
        weights.push_back(measurements[i]->getWeight());
    }
    
    // Compute chi-square statistic
    Real chiSquare = computeChiSquare(residuals, weights);
    result.chiSquareStatistic = chiSquare;
    
    // Degrees of freedom = number of measurements - number of states
    size_t nMeas = measurements.size();
    size_t nStates = 2 * network.getBusCount();
    size_t dof = (nMeas > nStates) ? (nMeas - nStates) : 1;
    
    // Chi-square threshold (95% confidence)
    result.chiSquareThreshold = dof * 1.96;  // Simplified
    
    if (chiSquare > result.chiSquareThreshold) {
        result.hasBadData = true;
        
        // Find measurements with large normalized residuals
        std::vector<Real> normalizedResiduals = computeNormalizedResiduals(
            telemetry, state, network);
        
        for (size_t i = 0; i < normalizedResiduals.size(); ++i) {
            if (std::abs(normalizedResiduals[i]) > normalizedResidualThreshold_) {
                result.badMeasurementIndices.push_back(i);
                result.badDeviceIds.push_back(measurements[i]->getDeviceId());
            }
        }
    }
    
    return result;
}

BadDataResult BadDataDetector::detectBadDataLNR(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network) {
    
    BadDataResult result;
    result.hasBadData = false;
    
    std::vector<Real> normalizedResiduals = computeNormalizedResiduals(
        telemetry, state, network);
    
    // Find largest normalized residual
    auto maxIt = std::max_element(normalizedResiduals.begin(), 
                                  normalizedResiduals.end(),
                                  [](Real a, Real b) {
                                      return std::abs(a) < std::abs(b);
                                  });
    
    if (maxIt != normalizedResiduals.end()) {
        Real maxResidual = std::abs(*maxIt);
        Index maxIdx = std::distance(normalizedResiduals.begin(), maxIt);
        
        if (maxResidual > normalizedResidualThreshold_) {
            result.hasBadData = true;
            result.badMeasurementIndices.push_back(maxIdx);
            
            const auto& measurements = telemetry.getMeasurements();
            if (maxIdx >= 0 && static_cast<size_t>(maxIdx) < measurements.size()) {
                result.badDeviceIds.push_back(measurements[maxIdx]->getDeviceId());
            }
        }
    }
    
    return result;
}

BadDataResult BadDataDetector::detectBadData(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network) {
    
    // Use chi-square test first, then LNR for identification
    BadDataResult result = detectBadDataChiSquare(telemetry, state, network);
    
    if (result.hasBadData) {
        // Refine using LNR
        BadDataResult lnrResult = detectBadDataLNR(telemetry, state, network);
        // Combine results
    }
    
    return result;
}

void BadDataDetector::removeBadMeasurements(model::TelemetryData& telemetry,
                                           const BadDataResult& result) {
    // Remove measurements by device ID (preferred method - O(1) average case)
    for (const std::string& deviceId : result.badDeviceIds) {
        if (!deviceId.empty()) {
            telemetry.removeMeasurement(deviceId);
        }
    }
    
    // Also handle indices if device IDs are not available
    // Extract device IDs from indices before any removals (to avoid index shifting)
    if (!result.badMeasurementIndices.empty()) {
        const auto& measurements = telemetry.getMeasurements();
        std::vector<std::string> deviceIdsFromIndices;
        
        // Collect device IDs from indices (before any removals)
        for (Index idx : result.badMeasurementIndices) {
            if (idx >= 0 && static_cast<size_t>(idx) < measurements.size()) {
                const std::string& deviceId = measurements[idx]->getDeviceId();
                if (!deviceId.empty()) {
                    // Check if not already in badDeviceIds (avoid duplicate removal)
                    bool alreadyRemoved = false;
                    for (const std::string& existingId : result.badDeviceIds) {
                        if (existingId == deviceId) {
                            alreadyRemoved = true;
                            break;
                        }
                    }
                    if (!alreadyRemoved) {
                        deviceIdsFromIndices.push_back(deviceId);
                    }
                }
            }
        }
        
        // Remove by device ID (avoids index shifting issues)
        for (const std::string& deviceId : deviceIdsFromIndices) {
            telemetry.removeMeasurement(deviceId);
        }
    }
}

std::vector<Real> BadDataDetector::computeNormalizedResiduals(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network) {
    
    math::MeasurementFunctions measFuncs;
    std::vector<Real> hx;
    measFuncs.evaluate(state, network, telemetry, hx);
    
    std::vector<Real> normalizedResiduals;
    const auto& measurements = telemetry.getMeasurements();
    
    for (size_t i = 0; i < measurements.size() && i < hx.size(); ++i) {
        Real residual = measurements[i]->getValue() - hx[i];
        Real stdDev = measurements[i]->getStdDev();
        normalizedResiduals.push_back(residual / stdDev);
    }
    
    return normalizedResiduals;
}

Real BadDataDetector::computeChiSquare(const std::vector<Real>& residuals,
                                       const std::vector<Real>& weights) {
    Real chiSquare = 0.0;
    
    for (size_t i = 0; i < residuals.size() && i < weights.size(); ++i) {
        chiSquare += weights[i] * residuals[i] * residuals[i];
    }
    
    return chiSquare;
}

} // namespace baddata
} // namespace sle

