/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/interface/MeasurementLoader.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <fstream>
#include <sstream>
#include <algorithm>

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
    const std::string& filepath, const model::NetworkModel& /* network */) {
    
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
        std::string typeStr, deviceId, busIdStr, valueStr, stdDevStr;
        
        if (std::getline(iss, typeStr, ',') &&
            std::getline(iss, deviceId, ',') &&
            std::getline(iss, busIdStr, ',') &&
            std::getline(iss, valueStr, ',') &&
            std::getline(iss, stdDevStr, ',')) {
            
            MeasurementType type;
            if (typeStr == "P_FLOW") type = MeasurementType::P_FLOW;
            else if (typeStr == "Q_FLOW") type = MeasurementType::Q_FLOW;
            else if (typeStr == "P_INJECTION") type = MeasurementType::P_INJECTION;
            else if (typeStr == "Q_INJECTION") type = MeasurementType::Q_INJECTION;
            else if (typeStr == "V_MAGNITUDE") type = MeasurementType::V_MAGNITUDE;
            else if (typeStr == "I_MAGNITUDE") type = MeasurementType::I_MAGNITUDE;
            else continue;
            
            BusId busId = std::stoi(busIdStr);
            Real value = std::stod(valueStr);
            Real stdDev = std::stod(stdDevStr);
            
            auto measurement = std::make_unique<model::MeasurementModel>(
                type, value, stdDev, deviceId);
            measurement->setLocation(busId);
            
            telemetry->addMeasurement(std::move(measurement));
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

void MeasurementLoader::addVirtualMeasurements(model::TelemetryData& telemetry,
                                               const model::NetworkModel& network) {
    auto buses = network.getBuses();
    
    for (auto* bus : buses) {
        if (bus->isZeroInjection()) {
            // Add zero injection constraint: P = 0, Q = 0
            auto pMeas = std::make_unique<model::MeasurementModel>(
                MeasurementType::VIRTUAL, 0.0, 1e-6, "VIRTUAL_P");
            pMeas->setLocation(bus->getId());
            telemetry.addMeasurement(std::move(pMeas));
            
            auto qMeas = std::make_unique<model::MeasurementModel>(
                MeasurementType::VIRTUAL, 0.0, 1e-6, "VIRTUAL_Q");
            qMeas->setLocation(bus->getId());
            telemetry.addMeasurement(std::move(qMeas));
        }
    }
}

void MeasurementLoader::addPseudoMeasurements(model::TelemetryData& telemetry,
                                              const model::NetworkModel& network,
                                              const std::vector<Real>& loadForecasts) {
    auto buses = network.getBuses();
    
    for (size_t i = 0; i < buses.size() && i < loadForecasts.size(); ++i) {
        // Add pseudo load measurements with low weight
        Real stdDev = 0.1;  // High uncertainty for pseudo measurements
        
        auto pMeas = std::make_unique<model::MeasurementModel>(
            MeasurementType::PSEUDO, loadForecasts[i], stdDev, "PSEUDO_P");
        pMeas->setLocation(buses[i]->getId());
        telemetry.addMeasurement(std::move(pMeas));
    }
}

} // namespace interface
} // namespace sle

