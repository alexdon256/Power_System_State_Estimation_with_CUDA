/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/DeviceModel.h>

namespace sle {
namespace model {

DeviceModel::DeviceModel(DeviceId id, DeviceType type, const std::string& name)
    : id_(id), name_(name), type_(type),
      busId_(-1), fromBus_(-1), toBus_(-1),
      stdDev_(0.01), enabled_(true), timestamp_(0) {
}

void DeviceModel::setBranchLocation(BusId fromBus, BusId toBus) {
    fromBus_ = fromBus;
    toBus_ = toBus;
}

} // namespace model
} // namespace sle

