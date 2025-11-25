/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/io/SCADAParser.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/Types.h>
#include <fstream>
#include <sstream>

namespace sle {
namespace io {

std::unique_ptr<model::TelemetryData> SCADAParser::parse(const std::string& filepath) {
    auto telemetry = std::make_unique<model::TelemetryData>();
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open SCADA file: " + filepath);
    }
    
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
            
            auto measurement = std::make_unique<model::MeasurementModel>(
                type, value, stdDev);
            measurement->setLocation(busId);
            measurement->setTimestamp(timestamp);
            
            // Pass deviceId to addMeasurement for automatic linking
            telemetry->addMeasurement(std::move(measurement), deviceId);
        }
    }
    
    return telemetry;
}

} // namespace io
} // namespace sle

