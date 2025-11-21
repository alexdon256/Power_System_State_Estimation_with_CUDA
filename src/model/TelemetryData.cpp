/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/TelemetryData.h>

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


void TelemetryData::clear() {
    measurements_.clear();
    deviceIdIndex_.clear();
}

} // namespace model
} // namespace sle

