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

// Forward declaration
class MeasurementDevice;

class MeasurementModel {
public:
    MeasurementModel(MeasurementType type, Real value, Real stdDev);
    
    MeasurementType getType() const { return type_; }
    Real getValue() const { return value_; }
    void setValue(Real value) { value_ = value; }
    
    Real getStdDev() const { return stdDev_; }
    void setStdDev(Real stdDev) { stdDev_ = stdDev; }
    Real getWeight() const { return 1.0 / (stdDev_ * stdDev_); }
    
    MeasurementDevice* getDevice() const { return device_; }
    void setDevice(MeasurementDevice* device) { device_ = device; }
    std::string getDeviceId() const;  // Get device ID from device pointer
    
    void setLocation(BusId busId) { busId_ = busId; }
    BusId getLocation() const { return busId_; }
    
    void setBranchLocation(BusId fromBus, BusId toBus);
    BusId getFromBus() const { return fromBus_; }
    BusId getToBus() const { return toBus_; }
    
    void setTimestamp(int64_t timestamp) { timestamp_ = timestamp; }
    int64_t getTimestamp() const { return timestamp_; }
    
private:
    MeasurementType type_;
    Real value_;
    Real stdDev_;
    MeasurementDevice* device_;  // Pointer to device that produced this measurement
    
    BusId busId_;      // For bus measurements
    BusId fromBus_;    // For branch measurements
    BusId toBus_;      // For branch measurements
    
    int64_t timestamp_;
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_MEASUREMENTMODEL_H

