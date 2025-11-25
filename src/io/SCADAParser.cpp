/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/io/SCADAParser.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/model/MeasurementDevice.h>
#include <sle/Types.h>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace sle {
namespace io {

std::unique_ptr<model::TelemetryData> SCADAParser::parse(const std::string& filepath) {
    auto telemetry = std::make_unique<model::TelemetryData>();
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open SCADA file: " + filepath);
    }
    
    // First pass: collect measurements grouped by device
    struct MeasurementData {
        MeasurementType type;
        Real value;
        Real stdDev;
        BusId busId;
        int64_t timestamp;
    };
    std::unordered_map<std::string, std::vector<MeasurementData>> deviceMeasurements;
    std::unordered_map<std::string, BusId> deviceBusMap;  // Track bus for each device
    
    std::string line;
    while (std::getline(file, line)) {
        // Parse SCADA format (simplified)
        // Format: deviceId,type,busId,value,timestamp
        std::istringstream iss(line);
        std::string deviceId, typeStr, busIdStr, valueStr, timestampStr;
        
        if (std::getline(iss, deviceId, ',') &&
            std::getline(iss, typeStr, ',') &&
            std::getline(iss, busIdStr, ',') &&
            std::getline(iss, valueStr, ',') &&
            std::getline(iss, timestampStr, ',')) {
            
            MeasurementType type = MeasurementType::P_INJECTION;  // Default
            if (typeStr == "P") type = MeasurementType::P_INJECTION;
            else if (typeStr == "Q") type = MeasurementType::Q_INJECTION;
            else if (typeStr == "V") type = MeasurementType::V_MAGNITUDE;
            
            BusId busId = std::stoi(busIdStr);
            Real value = std::stod(valueStr);
            int64_t timestamp = std::stoll(timestampStr);
            
            Real stdDev = 0.01;  // Default uncertainty
            
            MeasurementData data;
            data.type = type;
            data.value = value;
            data.stdDev = stdDev;
            data.busId = busId;
            data.timestamp = timestamp;
            
            deviceMeasurements[deviceId].push_back(data);
            deviceBusMap[deviceId] = busId;  // Store bus ID for device
        }
    }
    
    // Second pass: Create devices from topology, then add measurements
    for (const auto& pair : deviceMeasurements) {
        const std::string& deviceId = pair.first;
        const auto& measurements = pair.second;
        BusId busId = deviceBusMap[deviceId];
        
        // Create voltmeter device for this bus
        auto voltmeter = std::make_unique<model::Voltmeter>(
            deviceId, busId, 1.0, "SCADA Device " + deviceId
        );
        telemetry->addDevice(std::move(voltmeter));
        
        // Add measurements to device
        for (const auto& measData : measurements) {
            auto measurement = std::make_unique<model::MeasurementModel>(
                measData.type, measData.value, measData.stdDev);
            measurement->setTimestamp(measData.timestamp);
            telemetry->addMeasurementToDevice(deviceId, std::move(measurement));
        }
    }
    
    return telemetry;
}

} // namespace io
} // namespace sle

