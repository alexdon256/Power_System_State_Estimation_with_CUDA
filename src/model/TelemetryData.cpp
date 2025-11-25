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
    
    // Link to device if device pointer is set
    MeasurementDevice* device = measurement->getDevice();
    if (device) {
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

bool TelemetryData::removeMeasurement(MeasurementModel* measurement) {
    if (!measurement) return false;
    
    // Unlink from device
    if (MeasurementDevice* device = measurement->getDevice()) {
        device->removeMeasurement(measurement);
    }
    
    // Remove from vector
    auto it = std::find_if(measurements_.begin(), measurements_.end(),
        [measurement](const std::unique_ptr<MeasurementModel>& ptr) {
            return ptr.get() == measurement;
        });
    if (it != measurements_.end()) {
        measurements_.erase(it);
        return true;
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
    std::vector<MeasurementModel*> toRemove(deviceMeasurements.begin(), deviceMeasurements.end());
    
    // Remove each measurement
    size_t removedCount = 0;
    for (MeasurementModel* m : toRemove) {
        if (removeMeasurement(m)) {
            removedCount++;
        }
    }
    
    return removedCount;
}

bool TelemetryData::updateMeasurement(const std::string& deviceId, Real value, Real stdDev, int64_t timestamp) {
    if (deviceId.empty()) return false;
    
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end() || !deviceIt->second) {
        return false;
    }
    
    // Update first measurement from device (most common case: one measurement per device)
    const auto& deviceMeasurements = deviceIt->second->getMeasurements();
    if (deviceMeasurements.empty()) {
        return false;
    }
    
    MeasurementModel* m = deviceMeasurements[0];
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
    if (deviceId.empty() || devices_.find(deviceId) != devices_.end()) {
        return;
    }
    
    devices_[deviceId] = std::move(device);
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

void TelemetryData::clear() {
    // Unlink measurements from devices
    for (auto& m : measurements_) {
        if (m && m->getDevice()) {
            m->getDevice()->removeMeasurement(m.get());
        }
    }
    
    measurements_.clear();
    devices_.clear();
}

} // namespace model
} // namespace sle

