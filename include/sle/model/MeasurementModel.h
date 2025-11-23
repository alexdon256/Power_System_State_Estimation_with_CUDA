/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_MEASUREMENTMODEL_H
#define SLE_MODEL_MEASUREMENTMODEL_H

#include <sle/Types.h>
#include <memory>
#include <string>

namespace sle {
namespace model {

class MeasurementModel {
public:
    MeasurementModel(MeasurementType type, Real value, Real stdDev, 
                     DeviceId deviceId = "");
    
    MeasurementType getType() const { return type_; }
    Real getValue() const { return value_; }
    void setValue(Real value) { value_ = value; }
    
    Real getStdDev() const { return stdDev_; }
    void setStdDev(Real stdDev) { stdDev_ = stdDev; }
    Real getVariance() const { return stdDev_ * stdDev_; }
    Real getWeight() const { return 1.0 / getVariance(); }
    
    DeviceId getDeviceId() const { return deviceId_; }
    void setDeviceId(DeviceId id) { deviceId_ = id; }
    
    void setLocation(BusId busId) { busId_ = busId; }
    BusId getLocation() const { return busId_; }
    
    void setBranchLocation(BusId fromBus, BusId toBus);
    BusId getFromBus() const { return fromBus_; }
    BusId getToBus() const { return toBus_; }
    
    void setStatus(MeasurementStatus status) { status_ = status; }
    MeasurementStatus getStatus() const { return status_; }
    
    void setTimestamp(int64_t timestamp) { timestamp_ = timestamp; }
    int64_t getTimestamp() const { return timestamp_; }
    
    bool isPseudo() const { return type_ == MeasurementType::PSEUDO; }
    
private:
    MeasurementType type_;
    Real value_;
    Real stdDev_;
    DeviceId deviceId_;
    
    BusId busId_;      // For bus measurements
    BusId fromBus_;    // For branch measurements
    BusId toBus_;      // For branch measurements
    
    MeasurementStatus status_;
    int64_t timestamp_;
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_MEASUREMENTMODEL_H

