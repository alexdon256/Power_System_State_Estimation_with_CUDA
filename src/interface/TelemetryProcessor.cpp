/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/interface/TelemetryProcessor.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>

namespace sle {
namespace interface {

TelemetryProcessor::TelemetryProcessor() 
    : telemetry_(nullptr), network_(nullptr), latestTimestamp_(0) {
}

TelemetryProcessor::~TelemetryProcessor() {
}

void TelemetryProcessor::setTelemetryData(model::TelemetryData* telemetry) {
    telemetry_ = telemetry;
}

void TelemetryProcessor::setNetworkModel(model::NetworkModel* network) {
    network_ = network;
}

void TelemetryProcessor::setTopologyChangeCallback(std::function<void()> callback) {
    onTopologyChange_ = callback;
}

void TelemetryProcessor::updateMeasurement(const TelemetryUpdate& update) {
    applyUpdate(update);
}

void TelemetryProcessor::addMeasurement(const TelemetryUpdate& update) {
    updateMeasurement(update);
}

void TelemetryProcessor::removeAllMeasurementsFromDevice(const std::string& deviceId) {
    if (!telemetry_) return;
    
    telemetry_->removeAllMeasurementsFromDevice(deviceId);
}

void TelemetryProcessor::updateMeasurements(const std::vector<TelemetryUpdate>& updates) {
    for (const auto& update : updates) {
        applyUpdate(update);
    }
}

void TelemetryProcessor::applyUpdate(const TelemetryUpdate& update) {
    // Handle BREAKER_STATUS separately (updates topology, not telemetry)
    if (update.type == MeasurementType::BREAKER_STATUS) {
        if (network_) {
            model::Branch* branch = nullptr;
            if (update.fromBus >= 0 && update.toBus >= 0) {
                branch = network_->getBranchByBuses(update.fromBus, update.toBus);
            } else if (update.busId >= 0) {
                branch = network_->getBranch(update.busId);
            }
            
            if (branch) {
                bool newStatus = (update.value > 0.5);
                if (branch->getStatus() != newStatus) {
                    branch->setStatus(newStatus);
                    if (onTopologyChange_) {
                        onTopologyChange_();
                    }
                }
            }
        }
        return;
    }

    if (!telemetry_) return;
    
    // Find device
    model::MeasurementDevice* device = nullptr;
    if (!update.deviceId.empty()) {
        const auto& devices = telemetry_->getDevices();
        auto it = devices.find(update.deviceId);
        if (it != devices.end()) {
            device = it->second.get();
        }
    }
    
    // Try updating existing measurement
    if (device && telemetry_->updateMeasurement(update.deviceId, update.value, update.stdDev, update.timestamp)) {
        latestTimestamp_ = update.timestamp;
        return;
    }
    
    // Create new measurement
    auto measurement = std::make_unique<model::MeasurementModel>(
        update.type, update.value, update.stdDev);
    
    if (device) {
        measurement->setDevice(device);
    }
    if (update.busId >= 0) {
        measurement->setLocation(update.busId);
    }
    if (update.fromBus >= 0 && update.toBus >= 0) {
        measurement->setBranchLocation(update.fromBus, update.toBus);
    }
    measurement->setTimestamp(update.timestamp);
    latestTimestamp_ = update.timestamp;
    
    telemetry_->addMeasurement(std::move(measurement));
}

} // namespace interface
} // namespace sle

