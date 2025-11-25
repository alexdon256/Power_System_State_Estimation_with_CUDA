/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_MEASUREMENTDEVICE_H
#define SLE_MODEL_MEASUREMENTDEVICE_H

#include <sle/Types.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace sle {
namespace model {

// Forward declaration
class MeasurementModel;

// Device status
enum class DeviceStatus {
    OPERATIONAL,    // Device is working normally
    CALIBRATING,    // Device is being calibrated
    MAINTENANCE,    // Device is under maintenance
    FAILED,         // Device has failed
    OFFLINE         // Device is offline/disabled
};

// Base class for measurement devices
class MeasurementDevice {
public:
    MeasurementDevice(DeviceId id, const std::string& name = "");
    virtual ~MeasurementDevice() = default;
    
    DeviceId getId() const { return id_; }
    const std::string& getName() const { return name_; }
    
    DeviceStatus getStatus() const { return status_; }
    void setStatus(DeviceStatus status) { status_ = status; }
    
    bool isOperational() const { return status_ == DeviceStatus::OPERATIONAL; }
    
    // Device accuracy (standard deviation multiplier, e.g., 0.01 = 1% accuracy)
    Real getAccuracy() const { return accuracy_; }
    void setAccuracy(Real accuracy) { accuracy_ = accuracy; }
    
    // Get all measurements produced by this device (returns iterators for efficient access)
    // Use begin()/end() for iteration, or getMeasurement(type) for direct lookup
    auto begin() const { return measurements_.begin(); }
    auto end() const { return measurements_.end(); }
    size_t size() const { return measurements_.size(); }
    
    // Add measurement produced by this device (takes ownership)
    // Returns pointer to added measurement for direct access
    MeasurementModel* addMeasurement(std::unique_ptr<MeasurementModel> measurement);
    
    // Remove measurement by type
    bool removeMeasurement(MeasurementType type);
    
    // Get measurement by type (O(1) lookup)
    MeasurementModel* getMeasurement(MeasurementType type);
    const MeasurementModel* getMeasurement(MeasurementType type) const;
    
    // Get device type as string (for debugging/logging)
    virtual std::string getDeviceType() const = 0;
    
protected:
    DeviceId id_;
    std::string name_;
    DeviceStatus status_;
    Real accuracy_;  // Standard deviation multiplier (0.01 = 1% accuracy)
    
    // Measurements owned by this device
    std::vector<std::unique_ptr<MeasurementModel>> measurements_;
    // Fast O(1) lookup by measurement type
    std::unordered_map<MeasurementType, MeasurementModel*> measurementMap_;
};

// Multimeter: Measures power flow on branches using CT and PT
class Multimeter : public MeasurementDevice {
public:
    Multimeter(DeviceId id, BranchId branchId, BusId fromBus, BusId toBus,
               Real ctRatio = 1.0, Real ptRatio = 1.0, const std::string& name = "");
    
    BusId getFromBus() const { return fromBus_; }
    BusId getToBus() const { return toBus_; }
    
    std::string getDeviceType() const override { return "Multimeter"; }
    
private:
    BusId fromBus_;
    BusId toBus_;
    Real ctRatio_;   // Current Transformer ratio (e.g., 100:1 = 100.0)
    Real ptRatio_;   // Potential Transformer ratio (e.g., 1000:1 = 1000.0)
};

// Voltmeter: Measures voltage magnitude at a bus
class Voltmeter : public MeasurementDevice {
public:
    Voltmeter(DeviceId id, BusId busId, Real ptRatio = 1.0, const std::string& name = "");
    
    BusId getBusId() const { return busId_; }
    
    std::string getDeviceType() const override { return "Voltmeter"; }
    
private:
    BusId busId_;
    Real ptRatio_;   // Potential Transformer ratio (e.g., 1000:1 = 1000.0)
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_MEASUREMENTDEVICE_H

