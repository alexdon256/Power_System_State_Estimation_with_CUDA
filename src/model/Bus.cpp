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

std::vector<const MeasurementDevice*> Bus::getAssociatedDevices(const TelemetryData& telemetry) const {
    return telemetry.getDevicesByBus(id_);
}

std::vector<const MeasurementModel*> Bus::getMeasurementsFromDevices(const TelemetryData& telemetry) const {
    std::vector<const MeasurementModel*> result;
    auto devices = getAssociatedDevices(telemetry);
    // Pre-allocate capacity for efficiency
    size_t totalMeasurements = 0;
    for (const auto* device : devices) {
        totalMeasurements += device->getMeasurements().size();
    }
    result.reserve(totalMeasurements);
    
    for (const auto* device : devices) {
        const auto& measurements = device->getMeasurements();
        result.insert(result.end(), measurements.begin(), measurements.end());
    }
    return result;
}

std::vector<const MeasurementModel*> Bus::getMeasurementsFromDevices(const TelemetryData& telemetry, MeasurementType type) const {
    std::vector<const MeasurementModel*> result;
    auto devices = getAssociatedDevices(telemetry);
    // Pre-allocate capacity estimate
    result.reserve(devices.size());
    
    for (const auto* device : devices) {
        const auto& measurements = device->getMeasurements();
        for (const auto* measurement : measurements) {
            if (measurement->getType() == type) {
                result.push_back(measurement);
            }
        }
    }
    return result;
}

Real Bus::getCurrentVoltageMeasurement(const TelemetryData& telemetry) const {
    auto measurements = getMeasurementsFromDevices(telemetry, MeasurementType::V_MAGNITUDE);
    if (!measurements.empty()) {
        // Return the most recent measurement (if multiple devices)
        // In practice, there's usually one voltmeter per bus
        return measurements[0]->getValue();
    }
    return std::numeric_limits<Real>::quiet_NaN();
}

bool Bus::getCurrentPowerInjections(const TelemetryData& telemetry, Real& pInjection, Real& qInjection) const {
    auto pMeasurements = getMeasurementsFromDevices(telemetry, MeasurementType::P_INJECTION);
    auto qMeasurements = getMeasurementsFromDevices(telemetry, MeasurementType::Q_INJECTION);
    
    bool found = false;
    if (!pMeasurements.empty()) {
        pInjection = pMeasurements[0]->getValue();
        found = true;
    } else {
        pInjection = 0.0;
    }
    
    if (!qMeasurements.empty()) {
        qInjection = qMeasurements[0]->getValue();
        found = true;
    } else {
        qInjection = 0.0;
    }
    
    return found;
}

} // namespace model
} // namespace sle

