/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/StateVector.h>
#include <sle/model/NetworkModel.h>
#include <cmath>
#include <algorithm>

namespace sle {
namespace model {

StateVector::StateVector() : nBuses_(0) {
}

StateVector::StateVector(size_t nBuses) : nBuses_(nBuses) {
    resize(nBuses);
}

void StateVector::resize(size_t nBuses) {
    nBuses_ = nBuses;
    
    // Reserve capacity to avoid reallocations
    state_.reserve(2 * nBuses);
    angles_.reserve(nBuses);
    magnitudes_.reserve(nBuses);
    
    state_.resize(2 * nBuses, 0.0);
    angles_.resize(nBuses, 0.0);
    magnitudes_.resize(nBuses, 1.0);
    
    // Initialize magnitudes to 1.0 p.u. (vectorized by compiler)
    std::fill(magnitudes_.begin(), magnitudes_.end(), 1.0);
}

Real StateVector::getVoltageMagnitude(Index busIdx) const {
    if (busIdx >= 0 && static_cast<size_t>(busIdx) < nBuses_) {
        return magnitudes_[busIdx];
    }
    return 1.0;
}

Real StateVector::getVoltageAngle(Index busIdx) const {
    if (busIdx >= 0 && static_cast<size_t>(busIdx) < nBuses_) {
        return angles_[busIdx];
    }
    return 0.0;
}

void StateVector::setVoltageMagnitude(Index busIdx, Real v) {
    if (busIdx >= 0 && static_cast<size_t>(busIdx) < nBuses_) {
        magnitudes_[busIdx] = v;
        state_[nBuses_ + busIdx] = v;
    }
}

void StateVector::setVoltageAngle(Index busIdx, Real angle) {
    if (busIdx >= 0 && static_cast<size_t>(busIdx) < nBuses_) {
        angles_[busIdx] = angle;
        state_[busIdx] = angle;
    }
}

void StateVector::updateFromStateVector() {
    // Vectorized update (compiler can optimize)
    const size_t n = nBuses_;
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        angles_[i] = state_[i];
        magnitudes_[i] = state_[n + i];
    }
}

void StateVector::updateStateVector() {
    // Vectorized update (compiler can optimize)
    const size_t n = nBuses_;
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        state_[i] = angles_[i];
        state_[n + i] = magnitudes_[i];
    }
}

void StateVector::initializeFromNetwork(const NetworkModel& network) {
    size_t n = network.getBusCount();
    resize(n);
    
    auto buses = network.getBuses();
    for (size_t i = 0; i < buses.size(); ++i) {
        magnitudes_[i] = buses[i]->getVoltageMagnitude();
        angles_[i] = buses[i]->getVoltageAngle();
    }
    
    updateStateVector();
}

void StateVector::add(const StateVector& other) {
    if (other.nBuses_ != nBuses_) return;
    
    // Add state vectors (vectorized by compiler with SIMD)
    const size_t n = state_.size();
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        state_[i] += other.state_[i];
    }
    
    updateFromStateVector();
}

void StateVector::scale(Real factor) {
    // Vectorized scaling (compiler can optimize)
    const size_t n = state_.size();
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        state_[i] *= factor;
    }
    updateFromStateVector();
}

Real StateVector::norm() const {
    return std::sqrt(normSquared());
}

Real StateVector::normSquared() const {
    // Vectorized norm squared computation (compiler can optimize)
    Real sum = 0.0;
    const size_t n = state_.size();
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        sum += state_[i] * state_[i];
    }
    return sum;
}

} // namespace model
} // namespace sle

