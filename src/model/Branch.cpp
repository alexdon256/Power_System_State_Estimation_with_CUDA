/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/Branch.h>
#include <sle/model/StateVector.h>
#include <cmath>

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

void Branch::computePowerFlow(const StateVector& state, Index fromBusIdx, Index toBusIdx,
                             Real& pFlow, Real& qFlow) const {
    // Get voltages and angles from state
    Real vFrom = state.getVoltageMagnitude(fromBusIdx);
    Real vTo = state.getVoltageMagnitude(toBusIdx);
    Real thetaFrom = state.getVoltageAngle(fromBusIdx);
    Real thetaTo = state.getVoltageAngle(toBusIdx);
    
    // Compute branch admittance
    Real z2 = r_ * r_ + x_ * x_;
    if (z2 < 1e-12) {
        pFlow = 0.0;
        qFlow = 0.0;
        return;
    }
    
    Real g = r_ / z2;
    Real b_series = -x_ / z2;
    Real tap = tapRatio_;
    Real phase = phaseShift_;
    Real thetaDiff = thetaFrom - thetaTo - phase;
    
    Real cosDiff = std::cos(thetaDiff);
    Real sinDiff = std::sin(thetaDiff);
    
    // Power flow from -> to
    Real tap2 = tap * tap;
    pFlow = (vFrom * vFrom * g / tap2) - (vFrom * vTo * (g * cosDiff + b_series * sinDiff) / tap);
    qFlow = (-vFrom * vFrom * (b_series + b_ * 0.5) / tap2) - 
            (vFrom * vTo * (g * sinDiff - b_series * cosDiff) / tap);
}

void Branch::computePowerFlowMW(const StateVector& state, Index fromBusIdx, Index toBusIdx,
                                Real baseMVA, Real& pMW, Real& qMVAR) const {
    Real pFlow, qFlow;
    computePowerFlow(state, fromBusIdx, toBusIdx, pFlow, qFlow);
    pMW = pFlow * baseMVA;
    qMVAR = qFlow * baseMVA;
}

Real Branch::computeCurrentMagnitude(Real pFlow, Real qFlow, Real vFrom) const {
    // I = |S| / V = sqrt(P² + Q²) / V
    Real sMag = std::sqrt(pFlow * pFlow + qFlow * qFlow);
    return sMag / vFrom;  // Current in p.u.
}

void Branch::setPowerFlow(Real p, Real q, Real pMW, Real qMVAR, Real iAmps, Real iPU) {
    pFlow_ = p;
    qFlow_ = q;
    pMW_ = pMW;
    qMVAR_ = qMVAR;
    iAmps_ = iAmps;
    iPU_ = iPU;
}

} // namespace model
} // namespace sle

