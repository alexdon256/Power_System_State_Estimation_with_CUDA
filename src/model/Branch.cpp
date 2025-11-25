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

void Branch::addAssociatedDevice(MeasurementDevice* device) {
    if (!device) return;
    // Check for duplicates
    for (auto* d : associatedDevices_) {
        if (d == device) return;
    }
    associatedDevices_.push_back(device);
}

void Branch::removeAssociatedDevice(MeasurementDevice* device) {
    if (!device) return;
    for (auto it = associatedDevices_.begin(); it != associatedDevices_.end(); ++it) {
        if (*it == device) {
            associatedDevices_.erase(it);
            return;
        }
    }
}

std::vector<const MeasurementDevice*> Branch::getAssociatedDevices(const TelemetryData& telemetry) const {
    // Legacy support - construct vector from local storage
    std::vector<const MeasurementDevice*> result;
    result.reserve(associatedDevices_.size());
    for (auto* device : associatedDevices_) {
        result.push_back(device);
    }
    return result;
}

std::vector<const MeasurementModel*> Branch::getMeasurementsFromDevices(const TelemetryData& telemetry) const {
    // OPTIMIZATION: Use local associatedDevices_ (no lookup)
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

std::vector<const MeasurementModel*> Branch::getMeasurementsFromDevices(const TelemetryData& telemetry, MeasurementType type) const {
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

bool Branch::getCurrentPowerFlow(const TelemetryData& telemetry, Real& pFlow, Real& qFlow) const {
    // OPTIMIZATION: Use local associatedDevices_ (no lookup)
    bool foundP = false, foundQ = false;
    
    for (const auto* device : associatedDevices_) {
        if (!foundP) {
            const MeasurementModel* pMeas = device->getMeasurement(MeasurementType::P_FLOW);
            if (pMeas) {
                pFlow = pMeas->getValue();
                foundP = true;
            }
        }
        if (!foundQ) {
            const MeasurementModel* qMeas = device->getMeasurement(MeasurementType::Q_FLOW);
            if (qMeas) {
                qFlow = qMeas->getValue();
                foundQ = true;
            }
        }
        if (foundP && foundQ) break;
    }
    
    if (!foundP) pFlow = 0.0;
    if (!foundQ) qFlow = 0.0;
    
    return foundP || foundQ;
}

Real Branch::getCurrentCurrentMeasurement(const TelemetryData& telemetry) const {
    // OPTIMIZATION: Use local associatedDevices_ (no lookup)
    for (const auto* device : associatedDevices_) {
        const MeasurementModel* meas = device->getMeasurement(MeasurementType::I_MAGNITUDE);
        if (meas) {
            return meas->getValue();
        }
    }
    return std::numeric_limits<Real>::quiet_NaN();
}

} // namespace model
} // namespace sle

