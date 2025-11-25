/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementDevice.h>
#include <sle/model/NetworkModel.h>
#include <algorithm>
#include <unordered_set>

namespace sle {
namespace model {

TelemetryData::TelemetryData() 
    : network_(nullptr), latestTimestamp_(0) {
}

void TelemetryData::addMeasurement(std::unique_ptr<MeasurementModel> measurement, const std::string& deviceId) {
    if (!measurement) return;
    
    // Handle BREAKER_STATUS separately (updates topology, not just telemetry)
    if (measurement->getType() == MeasurementType::BREAKER_STATUS) {
        if (network_) {
            Branch* branch = nullptr;
            if (measurement->getFromBus() >= 0 && measurement->getToBus() >= 0) {
                branch = network_->getBranchByBuses(measurement->getFromBus(), measurement->getToBus());
            } else if (measurement->getLocation() >= 0) {
                branch = network_->getBranch(measurement->getLocation());
            }
            
            if (branch) {
                bool newStatus = (measurement->getValue() > 0.5);
                if (branch->getStatus() != newStatus) {
                    branch->setStatus(newStatus);
                    if (onTopologyChange_) {
                        onTopologyChange_();
                    }
                }
            }
        }
    }
    
    // Link to device: prefer deviceId parameter, fallback to measurement's device pointer
    MeasurementDevice* device = nullptr;
    if (!deviceId.empty()) {
        auto deviceIt = devices_.find(deviceId);
        if (deviceIt != devices_.end() && deviceIt->second) {
            device = deviceIt->second.get();
        }
    } else {
        // Fallback to device pointer if already set in measurement
        device = measurement->getDevice();
    }
    
    if (device) {
        measurement->setDevice(device);
        device->addMeasurement(measurement.get());
    }
    
    measurements_.push_back(std::move(measurement));
}

void TelemetryData::getMeasurementVector(std::vector<Real>& z) const {
    z.clear();
    z.reserve(measurements_.size());
    for (const auto& m : measurements_) {
        z.push_back(m->getValue());
    }
}

void TelemetryData::getWeightMatrix(std::vector<Real>& weights) const {
    weights.clear();
    weights.reserve(measurements_.size());
    for (const auto& m : measurements_) {
        weights.push_back(m->getWeight());
    }
}

bool TelemetryData::removeMeasurement(const std::string& deviceId, MeasurementType type) {
    if (deviceId.empty()) return false;
    
    // Find device
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end() || !deviceIt->second) {
        return false;
    }
    
    // Find measurement with matching type
    const auto& deviceMeasurements = deviceIt->second->getMeasurements();
    for (MeasurementModel* m : deviceMeasurements) {
        if (m && m->getType() == type) {
            // Unlink from device
            deviceIt->second->removeMeasurement(m);
            
            // Remove from vector
            auto it = std::find_if(measurements_.begin(), measurements_.end(),
                [m](const std::unique_ptr<MeasurementModel>& ptr) {
                    return ptr.get() == m;
                });
            if (it != measurements_.end()) {
                measurements_.erase(it);
                return true;
            }
        }
    }
    
    return false;
}

size_t TelemetryData::removeAllMeasurementsFromDevice(const std::string& deviceId) {
    if (deviceId.empty()) return 0;
    
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end() || !deviceIt->second) {
        return 0;
    }
    
    // Get all measurements from device (copy since we'll modify)
    const auto& deviceMeasurements = deviceIt->second->getMeasurements();
    if (deviceMeasurements.empty()) {
        return 0;
    }
    
    // Create a set of pointers for O(1) lookup
    std::unordered_set<MeasurementModel*> toRemoveSet(deviceMeasurements.begin(), deviceMeasurements.end());
    
    // Remove measurements using erase-remove idiom (more efficient than multiple erase calls)
    size_t initialSize = measurements_.size();
    measurements_.erase(
        std::remove_if(measurements_.begin(), measurements_.end(),
            [&toRemoveSet, deviceIt](const std::unique_ptr<MeasurementModel>& ptr) {
                if (toRemoveSet.find(ptr.get()) != toRemoveSet.end()) {
                    // Unlink from device before removing
                    deviceIt->second->removeMeasurement(ptr.get());
                    return true;
                }
                return false;
            }),
        measurements_.end()
    );
    
    return initialSize - measurements_.size();
}

bool TelemetryData::updateMeasurement(const std::string& deviceId, MeasurementType type, Real value, Real stdDev, int64_t timestamp) {
    if (deviceId.empty()) return false;
    
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end() || !deviceIt->second) {
        return false;
    }
    
    // Find measurement with matching type
    const auto& deviceMeasurements = deviceIt->second->getMeasurements();
    for (MeasurementModel* m : deviceMeasurements) {
        if (m && m->getType() == type) {
            m->setValue(value);
            m->setStdDev(stdDev);
            if (timestamp >= 0) {
                m->setTimestamp(timestamp);
            }
            
            // Handle BREAKER_STATUS separately (updates topology)
            if (type == MeasurementType::BREAKER_STATUS && network_) {
                Branch* branch = nullptr;
                if (m->getFromBus() >= 0 && m->getToBus() >= 0) {
                    branch = network_->getBranchByBuses(m->getFromBus(), m->getToBus());
                } else if (m->getLocation() >= 0) {
                    branch = network_->getBranch(m->getLocation());
                }
                
                if (branch) {
                    bool newStatus = (value > 0.5);
                    if (branch->getStatus() != newStatus) {
                        branch->setStatus(newStatus);
                        if (onTopologyChange_) {
                            onTopologyChange_();
                        }
                    }
                }
            }
            
            return true;
        }
    }
    
    return false;
}


void TelemetryData::addDevice(std::unique_ptr<MeasurementDevice> device) {
    if (!device) return;
    
    const std::string& deviceId = device->getId();
    if (deviceId.empty() || devices_.find(deviceId) != devices_.end()) {
        return;
    }
    
    MeasurementDevice* devicePtr = device.get();
    devices_[deviceId] = std::move(device);
    
    // Link existing measurements to this device
    // Measurements were created with deviceId but not linked yet
    for (auto& m : measurements_) {
        if (m && !m->getDevice()) {
            // Check if measurement's location matches device location
            bool shouldLink = false;
            
            // For voltmeters, check busId match
            const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(devicePtr);
            if (voltmeter && m->getLocation() == voltmeter->getBusId()) {
                shouldLink = true;
            }
            
            // For multimeters, check branch location match
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


std::vector<const MeasurementDevice*> TelemetryData::getDevicesByBus(BusId busId) const {
    std::vector<const MeasurementDevice*> result;
    for (const auto& pair : devices_) {
        if (!pair.second) continue;
        
        const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(pair.second.get());
        if (voltmeter && voltmeter->getBusId() == busId) {
            result.push_back(pair.second.get());
        }
    }
    return result;
}

std::vector<const MeasurementDevice*> TelemetryData::getDevicesByBranch(BusId fromBus, BusId toBus) const {
    std::vector<const MeasurementDevice*> result;
    for (const auto& pair : devices_) {
        if (!pair.second) continue;
        
        const Multimeter* multimeter = dynamic_cast<const Multimeter*>(pair.second.get());
        if (multimeter && 
            ((multimeter->getFromBus() == fromBus && multimeter->getToBus() == toBus) ||
             (multimeter->getFromBus() == toBus && multimeter->getToBus() == fromBus))) {
            result.push_back(pair.second.get());
        }
    }
    return result;
}

void TelemetryData::setNetworkModel(NetworkModel* network) {
    network_ = network;
}

void TelemetryData::setTopologyChangeCallback(std::function<void()> callback) {
    onTopologyChange_ = callback;
}

void TelemetryData::updateMeasurement(const TelemetryUpdate& update) {
    applyUpdate(update);
}

void TelemetryData::addMeasurement(const TelemetryUpdate& update) {
    applyUpdate(update);
}

void TelemetryData::updateMeasurements(const std::vector<TelemetryUpdate>& updates) {
    for (const auto& update : updates) {
        applyUpdate(update);
    }
}

void TelemetryData::applyUpdate(const TelemetryUpdate& update) {
    // Handle BREAKER_STATUS separately (updates topology, not telemetry)
    if (update.type == MeasurementType::BREAKER_STATUS) {
        if (network_) {
            Branch* branch = nullptr;
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

    // Find device
    MeasurementDevice* device = nullptr;
    if (!update.deviceId.empty()) {
        auto it = devices_.find(update.deviceId);
        if (it != devices_.end()) {
            device = it->second.get();
        }
    }
    
    // Try updating existing measurement
    if (device && updateMeasurement(update.deviceId, update.type, update.value, update.stdDev, update.timestamp)) {
        latestTimestamp_ = update.timestamp;
        return;
    }
    
    // Create new measurement
    auto measurement = std::make_unique<MeasurementModel>(
        update.type, update.value, update.stdDev);
    
    if (update.busId >= 0) {
        measurement->setLocation(update.busId);
    }
    if (update.fromBus >= 0 && update.toBus >= 0) {
        measurement->setBranchLocation(update.fromBus, update.toBus);
    }
    measurement->setTimestamp(update.timestamp);
    latestTimestamp_ = update.timestamp;
    
    // Pass deviceId to addMeasurement for automatic linking
    addMeasurement(std::move(measurement), update.deviceId);
}

void TelemetryData::clear() {
    // Unlink measurements from devices
    for (auto& m : measurements_) {
        if (m && m->getDevice()) {
            m->getDevice()->removeMeasurement(m.get());
        }
    }
    
    measurements_.clear();
    devices_.clear();
    latestTimestamp_ = 0;
}

} // namespace model
} // namespace sle

