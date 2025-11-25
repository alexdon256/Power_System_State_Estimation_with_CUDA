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

void MeasurementDevice::addMeasurement(MeasurementModel* measurement) {
    if (!measurement) return;
    
    // Avoid duplicates
    if (std::find(measurements_.begin(), measurements_.end(), measurement) == measurements_.end()) {
        measurements_.push_back(measurement);
        measurement->setDevice(this);
    }
}

void MeasurementDevice::removeMeasurement(MeasurementModel* measurement) {
    if (!measurement) return;
    
    auto it = std::find(measurements_.begin(), measurements_.end(), measurement);
    if (it != measurements_.end()) {
        measurements_.erase(it);
        measurement->setDevice(nullptr);
    }
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

