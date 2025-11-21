/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/Branch.h>
#include <sle/model/StateVector.h>
#include <cmath>

#ifdef _MSC_VER
#include <intrin.h>  // For _sincos on MSVC
#endif

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
    
    // Check for zero tap ratio (division by zero protection)
    if (std::abs(tap) < 1e-12) {
        pFlow = 0.0;
        qFlow = 0.0;
        return;
    }
    
    Real phase = phaseShift_;
    Real thetaDiff = thetaFrom - thetaTo - phase;
    
    // Use sincos for simultaneous sin/cos computation (more efficient)
    // This is similar to the GPU version and reduces function call overhead
    Real sinDiff, cosDiff;
#ifdef _MSC_VER
    _sincos(thetaDiff, &sinDiff, &cosDiff);
#elif defined(__GNUC__) || defined(__clang__)
    sincos(thetaDiff, &sinDiff, &cosDiff);
#else
    // Fallback: separate calls (compiler may optimize)
    sinDiff = std::sin(thetaDiff);
    cosDiff = std::cos(thetaDiff);
#endif
    
    // Power flow from -> to
    // Optimized computation with reduced operations and better numerical stability
    Real tap2 = tap * tap;
    Real inv_tap2 = 1.0 / tap2;
    Real inv_tap = 1.0 / tap;
    Real vFrom2 = vFrom * vFrom;
    Real vFrom_vTo = vFrom * vTo;
    
    // Compute terms efficiently (reduces redundant multiplications)
    Real term1_p = vFrom2 * g * inv_tap2;
    Real term2_p = vFrom_vTo * (g * cosDiff + b_series * sinDiff) * inv_tap;
    pFlow = term1_p - term2_p;
    
    Real term1_q = -vFrom2 * (b_series + b_ * 0.5) * inv_tap2;
    Real term2_q = vFrom_vTo * (g * sinDiff - b_series * cosDiff) * inv_tap;
    qFlow = term1_q - term2_q;
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
    // Optimized: use hypot for better numerical stability and potentially faster
    // Check for zero voltage to avoid division by zero
    if (std::abs(vFrom) < 1e-12) {
        return 0.0;  // Return zero current for zero voltage
    }
    Real sMag = std::hypot(pFlow, qFlow);  // More accurate than sqrt(p²+q²)
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

