/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_BADDATA_BADDATADETECTOR_H
#define SLE_BADDATA_BADDATADETECTOR_H

#include <sle/Export.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <vector>
#include <string>

// Forward declaration
namespace sle {
namespace math {
    class MeasurementFunctions;
}
}

namespace sle {
namespace baddata {

struct SLE_API BadDataResult {
    std::vector<std::string> badDeviceIds;
    std::vector<Index> badMeasurementIndices;
    Real chiSquareStatistic;
    Real chiSquareThreshold;
    bool hasBadData;
    std::vector<Real> normalizedResiduals;
};

class SLE_API BadDataDetector {
public:
    BadDataDetector();
    
    // Chi-square test for bad data detection
    // measFuncs: Optional pointer to reuse existing MeasurementFunctions (avoids reallocation)
    // residualsOverride: Optional pointer to reuse existing residuals (avoids re-calculation)
    BadDataResult detectBadDataChiSquare(
        const model::TelemetryData& telemetry,
        const model::StateVector& state,
        const model::NetworkModel& network,
        math::MeasurementFunctions* measFuncs = nullptr,
        const std::vector<Real>* residualsOverride = nullptr);
    
    // Largest normalized residual test
    // measFuncs: Optional pointer to reuse existing MeasurementFunctions (avoids reallocation)
    BadDataResult detectBadDataLNR(
        const model::TelemetryData& telemetry,
        const model::StateVector& state,
        const model::NetworkModel& network,
        const std::vector<Real>* normalizedResidualsOverride = nullptr,
        math::MeasurementFunctions* measFuncs = nullptr);
    
    // Combined detection method
    // measFuncs: Optional pointer to reuse existing MeasurementFunctions (avoids reallocation)
    // residualsOverride: Optional pointer to reuse existing residuals (avoids re-calculation)
    BadDataResult detectBadData(
        const model::TelemetryData& telemetry,
        const model::StateVector& state,
        const model::NetworkModel& network,
        math::MeasurementFunctions* measFuncs = nullptr,
        const std::vector<Real>* residualsOverride = nullptr);
    
    // Remove bad measurements
    void removeBadMeasurements(model::TelemetryData& telemetry,
                               const BadDataResult& result);
    
    // Set threshold for normalized residual
    void setNormalizedResidualThreshold(Real threshold) {
        normalizedResidualThreshold_ = threshold;
    }
    
    Real getNormalizedResidualThreshold() const {
        return normalizedResidualThreshold_;
    }
    
private:
    Real normalizedResidualThreshold_;
    
    // Compute normalized residuals
    std::vector<Real> computeNormalizedResiduals(
        const model::TelemetryData& telemetry,
        const model::StateVector& state,
        const model::NetworkModel& network,
        math::MeasurementFunctions* measFuncs,
        const std::vector<Real>* residualsOverride = nullptr);
    
    // Compute chi-square statistic
    Real computeChiSquare(const std::vector<Real>& residuals,
                         const std::vector<Real>& weights);
};

} // namespace baddata
} // namespace sle

#endif // SLE_BADDATA_BADDATADETECTOR_H
