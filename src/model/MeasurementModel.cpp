/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/MeasurementModel.h>

namespace sle {
namespace model {

MeasurementModel::MeasurementModel(MeasurementType type, Real value, Real stdDev, DeviceId deviceId)
    : type_(type), value_(value), stdDev_(stdDev), deviceId_(deviceId),
      busId_(-1), fromBus_(-1), toBus_(-1),
      status_(MeasurementStatus::VALID), timestamp_(0) {
}

void MeasurementModel::setBranchLocation(BusId fromBus, BusId toBus) {
    fromBus_ = fromBus;
    toBus_ = toBus;
}

} // namespace model
} // namespace sle

