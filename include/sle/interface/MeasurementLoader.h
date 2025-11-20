/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_INTERFACE_MEASUREMENTLOADER_H
#define SLE_INTERFACE_MEASUREMENTLOADER_H

#include <sle/Export.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/DeviceModel.h>
#include <string>
#include <memory>
#include <vector>

namespace sle {
namespace interface {

class SLE_API MeasurementLoader {
public:
    // Load telemetry data from file
    static std::unique_ptr<model::TelemetryData> loadTelemetry(
        const std::string& filepath, const model::NetworkModel& network);
    
    // Load from CSV
    static std::unique_ptr<model::TelemetryData> loadFromCSV(
        const std::string& filepath, const model::NetworkModel& network);
    
    // Load from JSON
    static std::unique_ptr<model::TelemetryData> loadFromJSON(
        const std::string& filepath, const model::NetworkModel& network);
    
    // Add virtual measurements (zero injection)
    static void addVirtualMeasurements(model::TelemetryData& telemetry,
                                      const model::NetworkModel& network);
    
    // Add pseudo measurements (load forecasts)
    static void addPseudoMeasurements(model::TelemetryData& telemetry,
                                     const model::NetworkModel& network,
                                     const std::vector<Real>& loadForecasts);
};

} // namespace interface
} // namespace sle

#endif // SLE_INTERFACE_MEASUREMENTLOADER_H

