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

void TelemetryProcessor::removeMeasurement(const std::string& deviceId) {
    if (!telemetry_) return;
    
    telemetry_->removeMeasurement(deviceId);
}

void TelemetryProcessor::updateMeasurements(const std::vector<TelemetryUpdate>& updates) {
    for (const auto& update : updates) {
        applyUpdate(update);
    }
}

void TelemetryProcessor::applyUpdate(const TelemetryUpdate& update) {
    
    // Special handling for BREAKER_STATUS
    if (update.type == MeasurementType::BREAKER_STATUS) {
        if (network_) {
            model::Branch* branch = nullptr;
            
            // OPTIMIZED: O(1) lookup by (fromBus, toBus) pair instead of O(avg_degree) linear search
            if (update.fromBus >= 0 && update.toBus >= 0) {
                branch = network_->getBranchByBuses(update.fromBus, update.toBus);
            } 
            // Fallback: Try interpreting busId as BranchId if provided
            else if (update.busId >= 0) {
                branch = network_->getBranch(update.busId);
            }
            
            if (branch) {
                bool newStatus = (update.value > 0.5); // > 0.5 considered Closed
                if (branch->getStatus() != newStatus) {
                    branch->setStatus(newStatus);
                    // Trigger topology change
                    if (onTopologyChange_) {
                        onTopologyChange_();
                    }
                }
            }
        }
        return; // Don't add breaker status to telemetry measurements
    }

    if (!telemetry_) return;
    
    // Try to update existing measurement by device ID
    if (!update.deviceId.empty() && 
        telemetry_->updateMeasurement(update.deviceId, update.value, update.stdDev, update.timestamp)) {
        latestTimestamp_ = update.timestamp;
        return;  // Successfully updated existing measurement
    }
    
    // Create new measurement if update failed (device ID not found or empty)
    auto measurement = std::make_unique<model::MeasurementModel>(
        update.type, update.value, update.stdDev, update.deviceId);
    
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

