/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/measurements/PseudoMeasurementGenerator.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>

namespace sle {
namespace measurements {

void PseudoMeasurementGenerator::generateFromLoadForecasts(
    model::TelemetryData& telemetry,
    const model::NetworkModel& network,
    const std::vector<Real>& loadForecasts) {
    
    auto buses = network.getBuses();
    Real stdDev = 0.1;  // High uncertainty for pseudo measurements
    
    for (size_t i = 0; i < buses.size() && i < loadForecasts.size(); ++i) {
        Real forecast = loadForecasts[i];
        
        // Add pseudo load measurement (P Injection = -Load)
        // Use P_INJECTION type so Solver can process it
        auto pMeas = std::make_unique<model::MeasurementModel>(
            MeasurementType::P_INJECTION, -forecast, stdDev, 
            "PSEUDO_P_" + std::to_string(buses[i]->getId()));
        pMeas->setLocation(buses[i]->getId());
        telemetry.addMeasurement(std::move(pMeas));
    }
}

void PseudoMeasurementGenerator::generateFromHistorical(
    model::TelemetryData& telemetry,
    const model::NetworkModel& network,
    const std::vector<Real>& historicalLoads) {
    
    // Similar to load forecasts but with historical data
    generateFromLoadForecasts(telemetry, network, historicalLoads);
}

} // namespace measurements
} // namespace sle

