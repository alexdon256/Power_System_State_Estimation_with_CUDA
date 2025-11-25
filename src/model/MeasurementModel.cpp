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
    : type_(type), value_(value), stdDev_(stdDev), device_(nullptr),
      busId_(-1), fromBus_(-1), toBus_(-1), timestamp_(0) {
}

std::string MeasurementModel::getDeviceId() const {
    return device_ ? device_->getId() : "";
}

void MeasurementModel::setBranchLocation(BusId fromBus, BusId toBus) {
    fromBus_ = fromBus;
    toBus_ = toBus;
}

} // namespace model
} // namespace sle

