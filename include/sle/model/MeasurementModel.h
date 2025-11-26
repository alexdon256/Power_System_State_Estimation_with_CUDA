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
    void setStdDev(Real stdDev) { 
        stdDev_ = stdDev; 
        weightDirty_ = true;  // Mark weight as dirty
    }
    Real getWeight() const { 
        if (weightDirty_) {
            cachedWeight_ = 1.0 / (stdDev_ * stdDev_);
            weightDirty_ = false;
        }
        return cachedWeight_;
    }
    
    MeasurementDevice* getDevice() const { return device_; }
    void setDevice(MeasurementDevice* device) { device_ = device; }
    std::string getDeviceId() const;  // Get device ID from device pointer
    
    // Location information is accessed through the device, not stored redundantly here
    // Convenience methods to get location from device
    BusId getLocation() const;  // Get bus ID (for voltmeter measurements) or -1 if branch measurement
    BusId getFromBus() const;   // Get from bus ID (for multimeter measurements) or -1 if bus measurement
    BusId getToBus() const;     // Get to bus ID (for multimeter measurements) or -1 if bus measurement
    
    void setTimestamp(int64_t timestamp) { timestamp_ = timestamp; }
    int64_t getTimestamp() const { return timestamp_; }
    
    void setGlobalIndex(Index index) { globalIndex_ = index; }
    Index getGlobalIndex() const { return globalIndex_; }
    
private:
    MeasurementType type_;
    Real value_;
    Real stdDev_;
    mutable Real cachedWeight_;      // Cached weight = 1.0 / (stdDev^2)
    mutable bool weightDirty_;       // Flag to track if weight needs recomputation
    MeasurementDevice* device_;      // Pointer to device that produced this measurement
    // Device knows its location - no need to store redundantly
    
    int64_t timestamp_;
    Index globalIndex_ = -1;  // Global index in the measurement vector (for optimization)
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_MEASUREMENTMODEL_H

