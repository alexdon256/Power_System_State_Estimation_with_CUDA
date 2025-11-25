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
    const model::NetworkModel& network,
    math::MeasurementFunctions* measFuncs,
    const std::vector<Real>* residualsOverride) {
    
    BadDataResult result;
    result.hasBadData = false;
    
    std::vector<Real> residuals;
    
    if (residualsOverride && !residualsOverride->empty()) {
        // Use provided residuals
        residuals = *residualsOverride;
    } else {
    // Evaluate measurement functions
        // Reuse measFuncs if provided, otherwise create local
        std::unique_ptr<math::MeasurementFunctions> localMeasFuncs;
        math::MeasurementFunctions* pMeasFuncs = measFuncs;
        
        if (!pMeasFuncs) {
            localMeasFuncs = std::make_unique<math::MeasurementFunctions>();
            pMeasFuncs = localMeasFuncs.get();
        }
        
    std::vector<Real> hx;
        pMeasFuncs->evaluate(state, network, telemetry, hx);
        
        const auto& measurements = telemetry.getMeasurements();
        residuals.reserve(measurements.size());
        for (size_t i = 0; i < measurements.size() && i < hx.size(); ++i) {
            residuals.push_back(measurements[i]->getValue() - hx[i]);
        }
    }
    
    // Compute weights and normalized residuals in a single pass
    std::vector<Real> weights;
    std::vector<Real> normalizedResiduals;
    const auto& measurements = telemetry.getMeasurements();
    
    for (size_t i = 0; i < measurements.size() && i < residuals.size(); ++i) {
        weights.push_back(measurements[i]->getWeight());
        Real stdDev = measurements[i]->getStdDev();
        normalizedResiduals.push_back(stdDev > 0 ? residuals[i] / stdDev : 0.0);
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
        result.normalizedResiduals = normalizedResiduals;
        
        for (size_t i = 0; i < normalizedResiduals.size(); ++i) {
            if (std::abs(normalizedResiduals[i]) > normalizedResidualThreshold_) {
                result.badMeasurementIndices.push_back(i);
                if (measurements[i]->getDevice()) {
                    result.badDeviceIds.push_back(measurements[i]->getDevice()->getId());
                }
            }
        }
    } else {
        result.normalizedResiduals.clear();
    }
    
    return result;
}

BadDataResult BadDataDetector::detectBadDataLNR(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network,
    const std::vector<Real>* normalizedResidualsOverride,
    math::MeasurementFunctions* measFuncs) {
    
    BadDataResult result;
    result.hasBadData = false;
    
    std::vector<Real> normalizedResiduals;
    if (normalizedResidualsOverride && !normalizedResidualsOverride->empty()) {
        normalizedResiduals = *normalizedResidualsOverride;
    } else {
        normalizedResiduals = computeNormalizedResiduals(telemetry, state, network, measFuncs);
    }
    
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
                if (measurements[maxIdx]->getDevice()) {
                    result.badDeviceIds.push_back(measurements[maxIdx]->getDevice()->getId());
                }
            }
        }
    }
    
    return result;
}

BadDataResult BadDataDetector::detectBadData(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network,
    math::MeasurementFunctions* measFuncs,
    const std::vector<Real>* residualsOverride) {
    
    // Use chi-square test first, then LNR for identification
    BadDataResult result = detectBadDataChiSquare(telemetry, state, network, measFuncs, residualsOverride);
    
    if (result.hasBadData) {
        // Refine using LNR without re-evaluating measurement functions
        // Note: detectBadDataLNR calls computeNormalizedResiduals which now handles override
        BadDataResult lnrResult = detectBadDataLNR(
            telemetry, state, network, &result.normalizedResiduals, measFuncs);
        if (lnrResult.hasBadData) {
            // Combine LNR identification details
            result.badDeviceIds.insert(result.badDeviceIds.end(),
                                       lnrResult.badDeviceIds.begin(),
                                       lnrResult.badDeviceIds.end());
            result.badMeasurementIndices.insert(result.badMeasurementIndices.end(),
                                                lnrResult.badMeasurementIndices.begin(),
                                                lnrResult.badMeasurementIndices.end());
        }
    }
    
    return result;
}

void BadDataDetector::removeBadMeasurements(model::TelemetryData& telemetry,
                                           const BadDataResult& result) {
    // Strategy 1: Remove all measurements from bad devices (device-level bad data)
    for (const std::string& deviceId : result.badDeviceIds) {
        if (!deviceId.empty()) {
            telemetry.removeAllMeasurementsFromDevice(deviceId);
        }
    }
    
    // Strategy 2: Remove specific bad measurements by index (measurement-level bad data)
    // Process indices in reverse order to avoid index shifting issues
    if (!result.badMeasurementIndices.empty()) {
        const auto& measurements = telemetry.getMeasurements();
        
        // Sort indices in descending order to avoid index shifting during removal
        std::vector<Index> sortedIndices = result.badMeasurementIndices;
        std::sort(sortedIndices.begin(), sortedIndices.end(), std::greater<Index>());
        
        // Remove measurements by index (from highest to lowest)
        for (Index idx : sortedIndices) {
            if (idx >= 0 && static_cast<size_t>(idx) < measurements.size()) {
                MeasurementModel* m = measurements[idx].get();
                if (m) {
                    // Check if this device was already removed (avoid duplicate removal)
                    bool deviceAlreadyRemoved = false;
                    if (m->getDevice()) {
                        const std::string& deviceId = m->getDevice()->getId();
                        for (const std::string& badDeviceId : result.badDeviceIds) {
                            if (deviceId == badDeviceId) {
                                deviceAlreadyRemoved = true;
                                break;
                            }
                        }
                    }
                    
                    // Only remove if device wasn't already removed
                    if (!deviceAlreadyRemoved && m->getDevice()) {
                        telemetry.removeMeasurement(m->getDevice()->getId(), m->getType());
                    }
                }
            }
        }
    }
}

std::vector<Real> BadDataDetector::computeNormalizedResiduals(
    const model::TelemetryData& telemetry,
    const model::StateVector& state,
    const model::NetworkModel& network,
    math::MeasurementFunctions* measFuncs,
    const std::vector<Real>* residualsOverride) {
    
    std::vector<Real> residuals;
    const auto& measurements = telemetry.getMeasurements();
    
    if (residualsOverride && !residualsOverride->empty()) {
        residuals = *residualsOverride;
    } else {
        // Reuse measFuncs if provided
        std::unique_ptr<math::MeasurementFunctions> localMeasFuncs;
        math::MeasurementFunctions* pMeasFuncs = measFuncs;
        
        if (!pMeasFuncs) {
            localMeasFuncs = std::make_unique<math::MeasurementFunctions>();
            pMeasFuncs = localMeasFuncs.get();
        }
        
    std::vector<Real> hx;
        pMeasFuncs->evaluate(state, network, telemetry, hx);
        
        residuals.reserve(measurements.size());
        for (size_t i = 0; i < measurements.size() && i < hx.size(); ++i) {
            residuals.push_back(measurements[i]->getValue() - hx[i]);
        }
    }
    
    std::vector<Real> normalizedResiduals;
    
    for (size_t i = 0; i < measurements.size() && i < residuals.size(); ++i) {
        Real stdDev = measurements[i]->getStdDev();
        normalizedResiduals.push_back(stdDev > 0 ? residuals[i] / stdDev : 0.0);
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
