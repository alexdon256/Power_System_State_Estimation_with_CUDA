/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/interface/MeasurementLoader.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/model/MeasurementDevice.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

using sle::model::Multimeter;
using sle::model::Voltmeter;

namespace sle {
namespace interface {

std::unique_ptr<model::TelemetryData> MeasurementLoader::loadTelemetry(
    const std::string& filepath, const model::NetworkModel& network) {
    
    std::string ext = filepath.substr(filepath.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "csv") {
        return loadFromCSV(filepath, network);
    } else if (ext == "json") {
        return loadFromJSON(filepath, network);
    } else {
        return loadFromCSV(filepath, network);  // Default to CSV
    }
}

std::unique_ptr<model::TelemetryData> MeasurementLoader::loadFromCSV(
    const std::string& filepath, const model::NetworkModel& network) {
    
    auto telemetry = std::make_unique<model::TelemetryData>();
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::string line;
    bool firstLine = true;
    
    // First pass: collect measurement data (we'll create devices and add measurements in second pass)
    struct MeasurementData {
        std::string deviceId;
        MeasurementType type;
        Real value;
        Real stdDev;
        BusId busId;
        BusId fromBus;
        BusId toBus;
    };
    std::vector<MeasurementData> measurementData;
    
    while (std::getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue;  // Skip header
        }
        
        std::istringstream iss(line);
        std::string typeStr, deviceId, busIdStr, fromBusStr, toBusStr, valueStr, stdDevStr;
        
        // Try to parse with fromBus/toBus (for branch measurements)
        std::getline(iss, typeStr, ',');
        std::getline(iss, deviceId, ',');
        std::getline(iss, busIdStr, ',');
        
        // Check if next field is fromBus (branch measurement) or value (bus measurement)
        std::string nextField;
        std::getline(iss, nextField, ',');
        
        MeasurementType type;
        if (typeStr == "P_FLOW") type = MeasurementType::P_FLOW;
        else if (typeStr == "Q_FLOW") type = MeasurementType::Q_FLOW;
        else if (typeStr == "P_INJECTION") type = MeasurementType::P_INJECTION;
        else if (typeStr == "Q_INJECTION") type = MeasurementType::Q_INJECTION;
        else if (typeStr == "V_MAGNITUDE") type = MeasurementType::V_MAGNITUDE;
        else if (typeStr == "I_MAGNITUDE") type = MeasurementType::I_MAGNITUDE;
        // BREAKER_STATUS is no longer a measurement type - use CircuitBreaker component instead
        else continue;
        
        Real value, stdDev;
        BusId busId = -1, fromBus = -1, toBus = -1;
        
        // Check if this is a branch measurement (has fromBus/toBus)
        if (!nextField.empty() && std::all_of(nextField.begin(), nextField.end(), ::isdigit)) {
            // Branch measurement format: type,deviceId,busId,fromBus,toBus,value,stdDev
            fromBusStr = nextField;
            if (!std::getline(iss, toBusStr, ',') || !std::getline(iss, valueStr, ',') || 
                !std::getline(iss, stdDevStr, ',')) {
                continue;  // Skip malformed line
            }
            
            try {
                fromBus = std::stoi(fromBusStr);
                toBus = std::stoi(toBusStr);
            } catch (...) {
                continue;  // Skip invalid bus IDs
            }
        } else {
            // Bus measurement format: type,deviceId,busId,value,stdDev
            valueStr = nextField;
            if (!std::getline(iss, stdDevStr, ',')) {
                continue;  // Skip malformed line
            }
            try {
                busId = std::stoi(busIdStr);
            } catch (...) {
                continue;  // Skip invalid bus ID
            }
        }
        
        // Parse value and stdDev with error handling
        try {
            value = std::stod(valueStr);
            stdDev = std::stod(stdDevStr);
        } catch (...) {
            continue;  // Skip invalid values
        }
        
        MeasurementData data;
        data.deviceId = deviceId;
        data.type = type;
        data.value = value;
        data.stdDev = stdDev;
        data.busId = busId;
        data.fromBus = fromBus;
        data.toBus = toBus;
        measurementData.push_back(data);
    }
    
    // Second pass: Create devices from topology, then add measurements
    // Group measurements by device
    std::unordered_map<std::string, std::vector<MeasurementData>> deviceMeasurements;
    for (const auto& data : measurementData) {
        if (!data.deviceId.empty()) {
            deviceMeasurements[data.deviceId].push_back(data);
        }
    }
    
    // Create devices and add measurements
    for (const auto& pair : deviceMeasurements) {
        const std::string& deviceId = pair.first;
        const auto& measurements = pair.second;
        
        if (measurements.empty()) continue;
        
        // Determine device type and location from first measurement
        const auto& firstMeas = measurements[0];
        bool isBranchMeasurement = (firstMeas.fromBus >= 0 && firstMeas.toBus >= 0);
        
        if (isBranchMeasurement) {
            // Create multimeter device
            const model::Branch* branch = network.getBranchByBuses(firstMeas.fromBus, firstMeas.toBus);
            if (branch) {
                auto multimeter = std::make_unique<model::Multimeter>(
                    deviceId, branch->getId(), firstMeas.fromBus, firstMeas.toBus, 1.0, 1.0, deviceId
                );
                telemetry->addDevice(std::move(multimeter));
                
                // Add measurements to device
                // Note: Location is stored in device (multimeter), not in measurement
                for (const auto& measData : measurements) {
                    auto measurement = std::make_unique<model::MeasurementModel>(
                        measData.type, measData.value, measData.stdDev);
                    telemetry->addMeasurementToDevice(deviceId, std::move(measurement));
                }
            }
        } else {
            // Create voltmeter device
            if (firstMeas.busId >= 0) {
                auto voltmeter = std::make_unique<model::Voltmeter>(
                    deviceId, firstMeas.busId, 1.0, deviceId
                );
                telemetry->addDevice(std::move(voltmeter));
                
                // Add measurements to device
                // Note: Location is stored in device (voltmeter), not in measurement
                for (const auto& measData : measurements) {
                    auto measurement = std::make_unique<model::MeasurementModel>(
                        measData.type, measData.value, measData.stdDev);
                    telemetry->addMeasurementToDevice(deviceId, std::move(measurement));
                }
            }
        }
    }
    
    return telemetry;
}

std::unique_ptr<model::TelemetryData> MeasurementLoader::loadFromJSON(
    const std::string& /* filepath */, const model::NetworkModel& /* network */) {
    
    // Simplified - would use proper JSON library
    auto telemetry = std::make_unique<model::TelemetryData>();
    return telemetry;
}


void MeasurementLoader::loadDevices(const std::string& filepath,
                                   model::TelemetryData& telemetry,
                                   const model::NetworkModel& network) {
    std::string ext = filepath.substr(filepath.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // Only CSV format supported for devices
    loadDevicesFromCSV(filepath, telemetry, network);
}

void MeasurementLoader::loadDevicesFromCSV(const std::string& filepath,
                                          model::TelemetryData& telemetry,
                                          const model::NetworkModel& network) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open devices file: " + filepath);
    }
    
    std::string line;
    bool firstLine = true;
    
    // Load devices from topology first
    while (std::getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue;  // Skip header
        }
        
        std::istringstream iss(line);
        std::string deviceType, deviceId, name, locationStr, ctRatioStr, ptRatioStr, accuracyStr;
        
        // Format: deviceType,deviceId,name,locationId,ctRatio,ptRatio,accuracy
        // For multimeters: locationId is branchId, for voltmeters: locationId is busId
        if (std::getline(iss, deviceType, ',') &&
            std::getline(iss, deviceId, ',') &&
            std::getline(iss, name, ',') &&
            std::getline(iss, locationStr, ',')) {
            
            std::getline(iss, ctRatioStr, ',');
            std::getline(iss, ptRatioStr, ',');
            std::getline(iss, accuracyStr, ',');
            
            Real ctRatio = 1.0;
            Real ptRatio = 1.0;
            Real accuracy = 0.01;
            
            try {
                if (!ctRatioStr.empty()) ctRatio = std::stod(ctRatioStr);
                if (!ptRatioStr.empty()) ptRatio = std::stod(ptRatioStr);
                if (!accuracyStr.empty()) accuracy = std::stod(accuracyStr);
            } catch (...) {
                // Use defaults on parse error
            }
            
            if (deviceType == "MULTIMETER" || deviceType == "MM") {
                // For multimeter, locationStr should be branchId or "fromBus:toBus"
                std::istringstream locStream(locationStr);
                std::string fromBusStr, toBusStr;
                
                if (std::getline(locStream, fromBusStr, ':') &&
                    std::getline(locStream, toBusStr, ':')) {
                    // Format: "fromBus:toBus"
                    BusId fromBus, toBus;
                    try {
                        fromBus = std::stoi(fromBusStr);
                        toBus = std::stoi(toBusStr);
                    } catch (...) {
                        continue;  // Skip invalid location
                    }
                    
                    // Find branch ID
                    const model::Branch* branch = network.getBranchByBuses(fromBus, toBus);
                    if (branch) {
                        auto multimeter = std::make_unique<Multimeter>(
                            deviceId, branch->getId(), fromBus, toBus, ctRatio, ptRatio, name
                        );
                        multimeter->setAccuracy(accuracy);
                        telemetry.addDevice(std::move(multimeter));
                    }
                } else {
                    // Try as branchId
                    BranchId branchId;
                    try {
                        branchId = std::stoi(locationStr);
                    } catch (...) {
                        continue;  // Skip invalid branch ID
                    }
                    const model::Branch* branch = network.getBranch(branchId);
                    if (branch) {
                        auto multimeter = std::make_unique<Multimeter>(
                            deviceId, branchId, branch->getFromBus(), branch->getToBus(),
                            ctRatio, ptRatio, name
                        );
                        multimeter->setAccuracy(accuracy);
                        telemetry.addDevice(std::move(multimeter));
                    }
                }
            } else if (deviceType == "VOLTMETER" || deviceType == "VM") {
                // For voltmeter, locationStr is busId
                BusId busId;
                try {
                    busId = std::stoi(locationStr);
                } catch (...) {
                    continue;  // Skip invalid bus ID
                }
                auto voltmeter = std::make_unique<Voltmeter>(
                    deviceId, busId, ptRatio, name
                );
                voltmeter->setAccuracy(accuracy);
                telemetry.addDevice(std::move(voltmeter));
            }
        }
    }
    // Devices are now created from topology
    // Measurements should be added to devices separately via addMeasurementToDevice
}

} // namespace interface
} // namespace sle

