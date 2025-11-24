/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementDevice.h>
#include <algorithm>

namespace sle {
namespace model {

TelemetryData::TelemetryData() {
}

void TelemetryData::addMeasurement(std::unique_ptr<MeasurementModel> measurement) {
    if (!measurement) return;
    
    const std::string& deviceId = measurement->getDeviceId();
    if (!deviceId.empty()) {
        // Update device ID index
        deviceIdIndex_[deviceId] = measurements_.size();
        
        // Link measurement to device if device exists
        auto deviceIt = deviceIndex_.find(deviceId);
        if (deviceIt != deviceIndex_.end() && deviceIt->second < devices_.size()) {
            devices_[deviceIt->second]->addMeasurement(measurement.get());
        }
    }
    measurements_.push_back(std::move(measurement));
}

std::vector<const MeasurementModel*> TelemetryData::getMeasurementsByType(MeasurementType type) const {
    std::vector<const MeasurementModel*> result;
    for (const auto& m : measurements_) {
        if (m->getType() == type) {
            result.push_back(m.get());
        }
    }
    return result;
}

std::vector<const MeasurementModel*> TelemetryData::getMeasurementsByBus(BusId busId) const {
    std::vector<const MeasurementModel*> result;
    for (const auto& m : measurements_) {
        if (m->getLocation() == busId) {
            result.push_back(m.get());
        }
    }
    return result;
}

std::vector<const MeasurementModel*> TelemetryData::getMeasurementsByBranch(BusId fromBus, BusId toBus) const {
    std::vector<const MeasurementModel*> result;
    for (const auto& m : measurements_) {
        if ((m->getFromBus() == fromBus && m->getToBus() == toBus) ||
            (m->getFromBus() == toBus && m->getToBus() == fromBus)) {
            result.push_back(m.get());
        }
    }
    return result;
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

MeasurementModel* TelemetryData::findMeasurementByDeviceId(const std::string& deviceId) {
    if (deviceId.empty()) return nullptr;
    
    auto it = deviceIdIndex_.find(deviceId);
    if (it != deviceIdIndex_.end() && it->second < measurements_.size()) {
        return measurements_[it->second].get();
    }
    return nullptr;
}

const MeasurementModel* TelemetryData::findMeasurementByDeviceId(const std::string& deviceId) const {
    if (deviceId.empty()) return nullptr;
    
    auto it = deviceIdIndex_.find(deviceId);
    if (it != deviceIdIndex_.end() && it->second < measurements_.size()) {
        return measurements_[it->second].get();
    }
    return nullptr;
}

bool TelemetryData::removeMeasurement(const std::string& deviceId) {
    if (deviceId.empty()) return false;
    
    auto it = deviceIdIndex_.find(deviceId);
    if (it == deviceIdIndex_.end()) return false;
    
    size_t idx = it->second;
    
    // Remove measurement
    measurements_.erase(measurements_.begin() + idx);
    deviceIdIndex_.erase(it);
    
    // Update indices for all measurements after the removed one (indices shifted down by 1)
    for (size_t i = idx; i < measurements_.size(); ++i) {
        const std::string& deviceId = measurements_[i]->getDeviceId();
        if (!deviceId.empty()) {
            deviceIdIndex_[deviceId] = i;
        }
    }
    
    return true;
}

bool TelemetryData::updateMeasurement(const std::string& deviceId, Real value, Real stdDev, int64_t timestamp) {
    MeasurementModel* m = findMeasurementByDeviceId(deviceId);
    if (!m) return false;
    
    m->setValue(value);
    m->setStdDev(stdDev);
    if (timestamp >= 0) {
        m->setTimestamp(timestamp);
    }
    return true;
}


void TelemetryData::addDevice(std::unique_ptr<MeasurementDevice> device) {
    if (!device) return;
    
    const std::string& deviceId = device->getId();
    if (deviceId.empty()) return;
    
    auto it = deviceIndex_.find(deviceId);
    if (it != deviceIndex_.end()) {
        // Device already exists, update it
        devices_[it->second] = std::move(device);
        return;
    }
    
    deviceIndex_[deviceId] = devices_.size();
    devices_.push_back(std::move(device));
    
    // Link existing measurements to this device
    for (auto& measurement : measurements_) {
        if (measurement->getDeviceId() == deviceId) {
            devices_.back()->addMeasurement(measurement.get());
        }
    }
}

MeasurementDevice* TelemetryData::getDevice(const DeviceId& deviceId) {
    if (deviceId.empty()) return nullptr;
    
    auto it = deviceIndex_.find(deviceId);
    if (it != deviceIndex_.end() && it->second < devices_.size()) {
        return devices_[it->second].get();
    }
    return nullptr;
}

const MeasurementDevice* TelemetryData::getDevice(const DeviceId& deviceId) const {
    if (deviceId.empty()) return nullptr;
    
    auto it = deviceIndex_.find(deviceId);
    if (it != deviceIndex_.end() && it->second < devices_.size()) {
        return devices_[it->second].get();
    }
    return nullptr;
}

std::vector<const MeasurementDevice*> TelemetryData::getDevices() const {
    std::vector<const MeasurementDevice*> result;
    result.reserve(devices_.size());
    for (const auto& device : devices_) {
        result.push_back(device.get());
    }
    return result;
}

std::vector<const MeasurementDevice*> TelemetryData::getDevicesByBus(BusId busId) const {
    std::vector<const MeasurementDevice*> result;
    for (const auto& device : devices_) {
        // Check if it's a voltmeter on this bus
        const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(device.get());
        if (voltmeter && voltmeter->getBusId() == busId) {
            result.push_back(device.get());
        }
    }
    return result;
}

std::vector<const MeasurementDevice*> TelemetryData::getDevicesByBranch(BusId fromBus, BusId toBus) const {
    std::vector<const MeasurementDevice*> result;
    for (const auto& device : devices_) {
        // Check if it's a multimeter on this branch
        const Multimeter* multimeter = dynamic_cast<const Multimeter*>(device.get());
        if (multimeter && 
            ((multimeter->getFromBus() == fromBus && multimeter->getToBus() == toBus) ||
             (multimeter->getFromBus() == toBus && multimeter->getToBus() == fromBus))) {
            result.push_back(device.get());
        }
    }
    return result;
}

void TelemetryData::clear() {
    measurements_.clear();
    deviceIdIndex_.clear();
    devices_.clear();
    deviceIndex_.clear();
}

} // namespace model
} // namespace sle

