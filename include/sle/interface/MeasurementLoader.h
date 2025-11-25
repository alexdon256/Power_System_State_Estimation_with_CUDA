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
    
    // Load devices from file (creates devices and links measurements)
    static void loadDevices(const std::string& filepath, 
                           model::TelemetryData& telemetry,
                           const model::NetworkModel& network);
    
    // Load devices from CSV
    static void loadDevicesFromCSV(const std::string& filepath,
                                  model::TelemetryData& telemetry,
                                  const model::NetworkModel& network);
};

} // namespace interface
} // namespace sle

#endif // SLE_INTERFACE_MEASUREMENTLOADER_H

