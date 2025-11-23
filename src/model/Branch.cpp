/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/Branch.h>

namespace sle {
namespace model {

Branch::Branch(BranchId id, BusId fromBus, BusId toBus)
    : id_(id), fromBus_(fromBus), toBus_(toBus),
      r_(0.0), x_(0.0), b_(0.0), mvaRating_(0.0),
      tapRatio_(1.0), phaseShift_(0.0),
      pFlow_(0.0), qFlow_(0.0), pMW_(0.0), qMVAR_(0.0), iAmps_(0.0), iPU_(0.0) {
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

void Branch::setPowerFlow(Real p, Real q, Real pMW, Real qMVAR, Real iAmps, Real iPU) {
    pFlow_ = p;
    qFlow_ = q;
    pMW_ = pMW;
    qMVAR_ = qMVAR;
    iAmps_ = iAmps;
    iPU_ = iPU;
}

Branch& Branch::operator=(const Branch& other) {
    if (this != &other) {
        // id_, fromBus_, toBus_ are not copied (they identify the branch)
        r_ = other.r_;
        x_ = other.x_;
        b_ = other.b_;
        mvaRating_ = other.mvaRating_;
        tapRatio_ = other.tapRatio_;
        phaseShift_ = other.phaseShift_;
        // Computed values are not copied (they're recalculated)
        pFlow_ = other.pFlow_;
        qFlow_ = other.qFlow_;
        pMW_ = other.pMW_;
        qMVAR_ = other.qMVAR_;
        iAmps_ = other.iAmps_;
        iPU_ = other.iPU_;
    }
    return *this;
}

} // namespace model
} // namespace sle

