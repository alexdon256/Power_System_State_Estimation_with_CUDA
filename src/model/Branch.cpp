/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/Branch.h>
#include <cmath>

namespace sle {
namespace model {

Branch::Branch(BranchId id, BusId fromBus, BusId toBus)
    : id_(id), fromBus_(fromBus), toBus_(toBus),
      r_(0.0), x_(0.0), b_(0.0), mvaRating_(0.0),
      tapRatio_(1.0), phaseShift_(0.0) {
}

void Branch::setImpedance(Real r, Real x) {
    r_ = r;
    x_ = x;
}

Complex Branch::getAdmittance() const {
    Real z2 = r_ * r_ + x_ * x_;
    if (z2 < 1e-12) {
        return Complex(0.0, 0.0);
    }
    // Optimized: compute reciprocal once
    Real inv_z2 = 1.0 / z2;
    return Complex(r_ * inv_z2, -x_ * inv_z2);
}

} // namespace model
} // namespace sle

