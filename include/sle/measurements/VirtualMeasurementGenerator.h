/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MEASUREMENTS_VIRTUALMEASUREMENTGENERATOR_H
#define SLE_MEASUREMENTS_VIRTUALMEASUREMENTGENERATOR_H

#include <sle/model/TelemetryData.h>
#include <sle/model/NetworkModel.h>

namespace sle {
namespace measurements {

class VirtualMeasurementGenerator {
public:
    // Generate virtual measurements for zero injection buses
    static void generateZeroInjection(model::TelemetryData& telemetry,
                                     const model::NetworkModel& network);
    
    // Generate virtual measurements for reference bus
    static void generateReferenceBus(model::TelemetryData& telemetry,
                                    const model::NetworkModel& network);
};

} // namespace measurements
} // namespace sle

#endif // SLE_MEASUREMENTS_VIRTUALMEASUREMENTGENERATOR_H

