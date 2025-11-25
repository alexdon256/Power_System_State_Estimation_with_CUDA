/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/Bus.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementDevice.h>
#include <cmath>
#include <limits>

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

Bus& Bus::operator=(const Bus& other) {
    if (this != &other) {
        // id_ and name_ are not copied (they identify the bus)
        type_ = other.type_;
        baseKV_ = other.baseKV_;
        voltageMag_ = other.voltageMag_;
        voltageAngle_ = other.voltageAngle_;
        pLoad_ = other.pLoad_;
        qLoad_ = other.qLoad_;
        pGen_ = other.pGen_;
        qGen_ = other.qGen_;
        gShunt_ = other.gShunt_;
        bShunt_ = other.bShunt_;
        vMin_ = other.vMin_;
        vMax_ = other.vMax_;
        // Computed values are not copied (they're recalculated)
        vPU_ = other.vPU_;
        vKV_ = other.vKV_;
        thetaRad_ = other.thetaRad_;
        thetaDeg_ = other.thetaDeg_;
        pInjection_ = other.pInjection_;
        qInjection_ = other.qInjection_;
        pInjectionMW_ = other.pInjectionMW_;
        qInjectionMVAR_ = other.qInjectionMVAR_;
    }
    return *this;
}

void Bus::addAssociatedDevice(MeasurementDevice* device) {
    if (!device) return;
    // Check for duplicates
    for (auto* d : associatedDevices_) {
        if (d == device) return;
    }
    associatedDevices_.push_back(device);
}

void Bus::removeAssociatedDevice(MeasurementDevice* device) {
    if (!device) return;
    for (auto it = associatedDevices_.begin(); it != associatedDevices_.end(); ++it) {
        if (*it == device) {
            associatedDevices_.erase(it);
            return;
        }
    }
}

std::vector<const MeasurementDevice*> Bus::getAssociatedDevices(const TelemetryData& telemetry) const {
    // Legacy support - construct vector from local storage
    std::vector<const MeasurementDevice*> result;
    result.reserve(associatedDevices_.size());
    for (auto* device : associatedDevices_) {
        result.push_back(device);
    }
    return result;
}

std::vector<const MeasurementModel*> Bus::getMeasurementsFromDevices(const TelemetryData& telemetry) const {
    // OPTIMIZATION: Use local associatedDevices_ (no lookup)
    // Telemetry argument ignored but kept for API compatibility
    std::vector<const MeasurementModel*> result;
    size_t totalMeasurements = 0;
    for (const auto* device : associatedDevices_) {
        totalMeasurements += device->size();
    }
    result.reserve(totalMeasurements);
    
    for (const auto* device : associatedDevices_) {
        for (auto it = device->begin(); it != device->end(); ++it) {
            result.push_back(it->get());
        }
    }
    return result;
}

std::vector<const MeasurementModel*> Bus::getMeasurementsFromDevices(const TelemetryData& telemetry, MeasurementType type) const {
    // OPTIMIZATION: Use local associatedDevices_ (no lookup)
    std::vector<const MeasurementModel*> result;
    result.reserve(associatedDevices_.size());
    
    for (const auto* device : associatedDevices_) {
        const MeasurementModel* meas = device->getMeasurement(type);
        if (meas) {
            result.push_back(meas);
        }
    }
    return result;
}

Real Bus::getCurrentVoltageMeasurement(const TelemetryData& telemetry) const {
    // OPTIMIZATION: Use local associatedDevices_ (no lookup)
    for (const auto* device : associatedDevices_) {
        const MeasurementModel* meas = device->getMeasurement(MeasurementType::V_MAGNITUDE);
        if (meas) {
            return meas->getValue();
        }
    }
    return std::numeric_limits<Real>::quiet_NaN();
}

bool Bus::getCurrentPowerInjections(const TelemetryData& telemetry, Real& pInjection, Real& qInjection) const {
    // OPTIMIZATION: Use local associatedDevices_ (no lookup)
    bool foundP = false, foundQ = false;
    
    for (const auto* device : associatedDevices_) {
        if (!foundP) {
            const MeasurementModel* pMeas = device->getMeasurement(MeasurementType::P_INJECTION);
            if (pMeas) {
                pInjection = pMeas->getValue();
                foundP = true;
            }
        }
        if (!foundQ) {
            const MeasurementModel* qMeas = device->getMeasurement(MeasurementType::Q_INJECTION);
            if (qMeas) {
                qInjection = qMeas->getValue();
                foundQ = true;
            }
        }
        if (foundP && foundQ) break;
    }
    
    if (!foundP) pInjection = 0.0;
    if (!foundQ) qInjection = 0.0;
    
    return foundP || foundQ;
}

} // namespace model
} // namespace sle

