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
        else if (typeStr == "BREAKER_STATUS") type = MeasurementType::BREAKER_STATUS;
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
        
        auto measurement = std::make_unique<model::MeasurementModel>(type, value, stdDev);
        
        if (fromBus >= 0 && toBus >= 0) {
            measurement->setBranchLocation(fromBus, toBus);
        } else if (busId >= 0) {
            measurement->setLocation(busId);
        }
        
        // Pass deviceId to addMeasurement for automatic linking
        telemetry->addMeasurement(std::move(measurement), deviceId);
    }
    
    return telemetry;
}

std::unique_ptr<model::TelemetryData> MeasurementLoader::loadFromJSON(
    const std::string& /* filepath */, const model::NetworkModel& /* network */) {
    
    // Simplified - would use proper JSON library
    auto telemetry = std::make_unique<model::TelemetryData>();
    return telemetry;
}


void MeasurementLoader::addPseudoMeasurements(model::TelemetryData& telemetry,
                                              const model::NetworkModel& network,
                                              const std::vector<Real>& loadForecasts) {
    auto buses = network.getBuses();
    
    for (size_t i = 0; i < buses.size() && i < loadForecasts.size(); ++i) {
        // Add pseudo load measurements with low weight
        Real stdDev = 0.1;  // High uncertainty for pseudo measurements
        
        auto pMeas = std::make_unique<model::MeasurementModel>(
            MeasurementType::PSEUDO, loadForecasts[i], stdDev);
        pMeas->setLocation(buses[i]->getId());
        telemetry.addMeasurement(std::move(pMeas));
    }
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
            
            model::MeasurementDevice* devicePtr = nullptr;
            
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
                        devicePtr = multimeter.get();
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
                        devicePtr = multimeter.get();
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
                devicePtr = voltmeter.get();
                telemetry.addDevice(std::move(voltmeter));
            }
            
            // Link existing measurements to this device by matching location
            // This handles the case where measurements were loaded before devices
            if (devicePtr) {
                const auto& measurements = telemetry.getMeasurements();
                for (const auto& m : measurements) {
                    if (m && !m->getDevice()) {
                        bool shouldLink = false;
                        
                        // For voltmeters, match by busId
                        const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(devicePtr);
                        if (voltmeter && m->getLocation() == voltmeter->getBusId()) {
                            shouldLink = true;
                        }
                        
                        // For multimeters, match by branch location
                        const Multimeter* multimeter = dynamic_cast<const Multimeter*>(devicePtr);
                        if (multimeter && m->getFromBus() == multimeter->getFromBus() && 
                            m->getToBus() == multimeter->getToBus()) {
                            shouldLink = true;
                        }
                        
                        if (shouldLink) {
                            devicePtr->addMeasurement(m.get());
                        }
                    }
                }
            }
        }
    }
}

} // namespace interface
} // namespace sle

