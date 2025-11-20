/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_DEVICEMODEL_H
#define SLE_MODEL_DEVICEMODEL_H

#include <sle/Types.h>
#include <string>
#include <memory>

namespace sle {
namespace model {

class DeviceModel {
public:
    DeviceModel(DeviceId id, DeviceType type, const std::string& name = "");
    
    DeviceId getId() const { return id_; }
    const std::string& getName() const { return name_; }
    DeviceType getType() const { return type_; }
    
    void setLocation(BusId busId) { busId_ = busId; }
    BusId getLocation() const { return busId_; }
    
    void setBranchLocation(BusId fromBus, BusId toBus);
    BusId getFromBus() const { return fromBus_; }
    BusId getToBus() const { return toBus_; }
    
    void setAccuracy(Real stdDev) { stdDev_ = stdDev; }
    Real getStdDev() const { return stdDev_; }
    Real getVariance() const { return stdDev_ * stdDev_; }
    Real getWeight() const { return 1.0 / getVariance(); }
    
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    
    void setTimestamp(int64_t timestamp) { timestamp_ = timestamp; }
    int64_t getTimestamp() const { return timestamp_; }
    
private:
    DeviceId id_;
    std::string name_;
    DeviceType type_;
    
    BusId busId_;      // For bus measurements
    BusId fromBus_;    // For branch measurements
    BusId toBus_;      // For branch measurements
    
    Real stdDev_;      // Standard deviation of measurement error
    bool enabled_;
    int64_t timestamp_;
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_DEVICEMODEL_H

