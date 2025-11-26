/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/MeasurementModel.h>
#include <sle/model/MeasurementDevice.h>

namespace sle {
namespace model {

MeasurementModel::MeasurementModel(MeasurementType type, Real value, Real stdDev)
    : type_(type), value_(value), stdDev_(stdDev), device_(nullptr), timestamp_(0), globalIndex_(-1),
      cachedWeight_(1.0 / (stdDev * stdDev)), weightDirty_(false) {
}

std::string MeasurementModel::getDeviceId() const {
    return device_ ? device_->getId() : "";
}

BusId MeasurementModel::getLocation() const {
    if (!device_) return -1;
    const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(device_);
    if (voltmeter) {
        return voltmeter->getBusId();
    }
    return -1;  // Branch measurement, no single bus location
}

BusId MeasurementModel::getFromBus() const {
    if (!device_) return -1;
    const Multimeter* multimeter = dynamic_cast<const Multimeter*>(device_);
    if (multimeter) {
        return multimeter->getFromBus();
    }
    return -1;  // Bus measurement, no from bus
}

BusId MeasurementModel::getToBus() const {
    if (!device_) return -1;
    const Multimeter* multimeter = dynamic_cast<const Multimeter*>(device_);
    if (multimeter) {
        return multimeter->getToBus();
    }
    return -1;  // Bus measurement, no to bus
}

} // namespace model
} // namespace sle

