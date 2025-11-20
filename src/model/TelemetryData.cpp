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

void TelemetryData::clear() {
    measurements_.clear();
}

} // namespace model
} // namespace sle

