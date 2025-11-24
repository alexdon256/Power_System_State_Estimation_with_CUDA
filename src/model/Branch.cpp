/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/Branch.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementDevice.h>
#include <limits>

namespace sle {
namespace model {

Branch::Branch(BranchId id, BusId fromBus, BusId toBus)
    : id_(id), fromBus_(fromBus), toBus_(toBus),
      r_(0.0), x_(0.0), b_(0.0), mvaRating_(0.0),
      tapRatio_(1.0), phaseShift_(0.0),
      status_(true), // Default to Closed/In Service
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
        status_ = other.status_;
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

std::vector<const MeasurementDevice*> Branch::getAssociatedDevices(const TelemetryData& telemetry) const {
    return telemetry.getDevicesByBranch(fromBus_, toBus_);
}

std::vector<const MeasurementModel*> Branch::getMeasurementsFromDevices(const TelemetryData& telemetry) const {
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

std::vector<const MeasurementModel*> Branch::getMeasurementsFromDevices(const TelemetryData& telemetry, MeasurementType type) const {
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

bool Branch::getCurrentPowerFlow(const TelemetryData& telemetry, Real& pFlow, Real& qFlow) const {
    auto pMeasurements = getMeasurementsFromDevices(telemetry, MeasurementType::P_FLOW);
    auto qMeasurements = getMeasurementsFromDevices(telemetry, MeasurementType::Q_FLOW);
    
    bool found = false;
    if (!pMeasurements.empty()) {
        pFlow = pMeasurements[0]->getValue();
        found = true;
    } else {
        pFlow = 0.0;
    }
    
    if (!qMeasurements.empty()) {
        qFlow = qMeasurements[0]->getValue();
        found = true;
    } else {
        qFlow = 0.0;
    }
    
    return found;
}

Real Branch::getCurrentCurrentMeasurement(const TelemetryData& telemetry) const {
    auto measurements = getMeasurementsFromDevices(telemetry, MeasurementType::I_MAGNITUDE);
    if (!measurements.empty()) {
        // Return the most recent measurement (if multiple devices)
        return measurements[0]->getValue();
    }
    return std::numeric_limits<Real>::quiet_NaN();
}

} // namespace model
} // namespace sle

