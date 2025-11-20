/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/measurements/VirtualMeasurementGenerator.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>

namespace sle {
namespace measurements {

void VirtualMeasurementGenerator::generateZeroInjection(
    model::TelemetryData& telemetry,
    const model::NetworkModel& network) {
    
    auto buses = network.getBuses();
    
    for (auto* bus : buses) {
        if (bus->isZeroInjection()) {
            // Add zero injection constraints
            auto pMeas = std::make_unique<model::MeasurementModel>(
                MeasurementType::VIRTUAL, 0.0, 1e-6, "VIRTUAL_P_" + std::to_string(bus->getId()));
            pMeas->setLocation(bus->getId());
            telemetry.addMeasurement(std::move(pMeas));
            
            auto qMeas = std::make_unique<model::MeasurementModel>(
                MeasurementType::VIRTUAL, 0.0, 1e-6, "VIRTUAL_Q_" + std::to_string(bus->getId()));
            qMeas->setLocation(bus->getId());
            telemetry.addMeasurement(std::move(qMeas));
        }
    }
}

void VirtualMeasurementGenerator::generateReferenceBus(
    model::TelemetryData& telemetry,
    const model::NetworkModel& network) {
    
    BusId refBus = network.getReferenceBus();
    if (refBus >= 0) {
        auto* bus = network.getBus(refBus);
        if (bus) {
            // Reference bus has known angle (typically 0)
            auto angleMeas = std::make_unique<model::MeasurementModel>(
                MeasurementType::VIRTUAL, 0.0, 1e-6, "VIRTUAL_ANGLE_REF");
            angleMeas->setLocation(refBus);
            telemetry.addMeasurement(std::move(angleMeas));
        }
    }
}

} // namespace measurements
} // namespace sle

