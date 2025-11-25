/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/CircuitBreaker.h>

namespace sle {
namespace model {

CircuitBreaker::CircuitBreaker(const std::string& id, BranchId branchId, BusId fromBus, BusId toBus, const std::string& name)
    : id_(id), name_(name), branchId_(branchId), fromBus_(fromBus), toBus_(toBus), status_(true) {
    // Default to closed (allows flow)
}

} // namespace model
} // namespace sle

