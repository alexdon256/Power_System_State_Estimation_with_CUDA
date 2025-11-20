/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MEASUREMENTS_PSEUDOMEASUREMENTGENERATOR_H
#define SLE_MEASUREMENTS_PSEUDOMEASUREMENTGENERATOR_H

#include <sle/model/TelemetryData.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <vector>

namespace sle {
namespace measurements {

class PseudoMeasurementGenerator {
public:
    // Generate pseudo measurements from load forecasts
    static void generateFromLoadForecasts(model::TelemetryData& telemetry,
                                         const model::NetworkModel& network,
                                         const std::vector<Real>& loadForecasts);
    
    // Generate pseudo measurements from historical data
    static void generateFromHistorical(model::TelemetryData& telemetry,
                                      const model::NetworkModel& network,
                                      const std::vector<Real>& historicalLoads);
};

} // namespace measurements
} // namespace sle

#endif // SLE_MEASUREMENTS_PSEUDOMEASUREMENTGENERATOR_H

