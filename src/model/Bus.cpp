/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/Bus.h>
#include <cmath>

namespace sle {
namespace model {

Bus::Bus(BusId id, const std::string& name)
    : id_(id), name_(name), type_(BusType::PQ),
      baseKV_(0.0), voltageMag_(1.0), voltageAngle_(0.0),
      pLoad_(0.0), qLoad_(0.0), pGen_(0.0), qGen_(0.0),
      gShunt_(0.0), bShunt_(0.0), vMin_(0.9), vMax_(1.1),
      vPU_(0.0), vKV_(0.0), thetaRad_(0.0), thetaDeg_(0.0),
      pInjection_(0.0), qInjection_(0.0), pInjectionMW_(0.0), qInjectionMVAR_(0.0) {
}

void Bus::setVoltage(Real magnitude, Real angle) {
    voltageMag_ = magnitude;
    voltageAngle_ = angle;
}

void Bus::setLoad(Real pLoad, Real qLoad) {
    pLoad_ = pLoad;
    qLoad_ = qLoad;
}

void Bus::setGeneration(Real pGen, Real qGen) {
    pGen_ = pGen;
    qGen_ = qGen;
}

void Bus::setShunt(Real gShunt, Real bShunt) {
    gShunt_ = gShunt;
    bShunt_ = bShunt;
}

void Bus::setVoltageLimits(Real vMin, Real vMax) {
    vMin_ = vMin;
    vMax_ = vMax;
}

bool Bus::isZeroInjection() const {
    // Zero injection if no load and no generation
    const Real tolerance = 1e-6;
    return std::abs(pLoad_) < tolerance && std::abs(qLoad_) < tolerance &&
           std::abs(pGen_) < tolerance && std::abs(qGen_) < tolerance;
}

void Bus::setVoltEstimates(Real vPU, Real vKV, Real thetaRad, Real thetaDeg) {
    vPU_ = vPU;
    vKV_ = vKV;
    thetaRad_ = thetaRad;
    thetaDeg_ = thetaDeg;
}

void Bus::setPowerInjections(Real pInj, Real qInj, Real pMW, Real qMVAR) {
    pInjection_ = pInj;
    qInjection_ = qInj;
    pInjectionMW_ = pMW;
    qInjectionMVAR_ = qMVAR;
}

} // namespace model
} // namespace sle

