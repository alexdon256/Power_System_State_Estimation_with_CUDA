/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/MeasurementDevice.h>
#include <sle/model/MeasurementModel.h>
#include <algorithm>

namespace sle {
namespace model {

MeasurementDevice::MeasurementDevice(DeviceId id, const std::string& name)
    : id_(id), name_(name), status_(DeviceStatus::OPERATIONAL), accuracy_(0.01) {
}

MeasurementModel* MeasurementDevice::addMeasurement(std::unique_ptr<MeasurementModel> measurement) {
    if (!measurement) return nullptr;
    
    MeasurementType type = measurement->getType();
    
    // Check if measurement type already exists (one measurement per type per device)
    auto mapIt = measurementMap_.find(type);
    if (mapIt != measurementMap_.end()) {
        // Replace existing measurement of same type
        MeasurementModel* oldMeas = mapIt->second;
        oldMeas->setDevice(nullptr);
        
        // Find and replace in vector
        auto vecIt = std::find_if(measurements_.begin(), measurements_.end(),
            [oldMeas](const std::unique_ptr<MeasurementModel>& m) {
                return m.get() == oldMeas;
            });
        if (vecIt != measurements_.end()) {
            measurement->setDevice(this);
            MeasurementModel* newMeas = measurement.get();
            *vecIt = std::move(measurement);
            measurementMap_[type] = newMeas;
            return newMeas;
        }
    }
    
    // Add new measurement
    measurement->setDevice(this);
    MeasurementModel* measPtr = measurement.get();
    measurements_.push_back(std::move(measurement));
    measurementMap_[type] = measPtr;
    return measPtr;
}

bool MeasurementDevice::removeMeasurement(MeasurementType type) {
    auto mapIt = measurementMap_.find(type);
    if (mapIt == measurementMap_.end()) {
        return false;
    }
    
    MeasurementModel* measToRemove = mapIt->second;
    measToRemove->setDevice(nullptr);
    
    // Remove from vector
    auto vecIt = std::find_if(measurements_.begin(), measurements_.end(),
        [measToRemove](const std::unique_ptr<MeasurementModel>& m) {
            return m.get() == measToRemove;
        });
    if (vecIt != measurements_.end()) {
        measurements_.erase(vecIt);
    }
    
    // Remove from map
    measurementMap_.erase(mapIt);
    return true;
}

MeasurementModel* MeasurementDevice::getMeasurement(MeasurementType type) {
    auto it = measurementMap_.find(type);
    return (it != measurementMap_.end()) ? it->second : nullptr;
}

const MeasurementModel* MeasurementDevice::getMeasurement(MeasurementType type) const {
    auto it = measurementMap_.find(type);
    return (it != measurementMap_.end()) ? it->second : nullptr;
}

Multimeter::Multimeter(DeviceId id, BranchId branchId, BusId fromBus, BusId toBus,
                       Real ctRatio, Real ptRatio, const std::string& name)
    : MeasurementDevice(id, name),
      fromBus_(fromBus), toBus_(toBus), ctRatio_(ctRatio), ptRatio_(ptRatio) {
}


Voltmeter::Voltmeter(DeviceId id, BusId busId, Real ptRatio, const std::string& name)
    : MeasurementDevice(id, name), busId_(busId), ptRatio_(ptRatio) {
}


} // namespace model
} // namespace sle

