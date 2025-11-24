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
    void setName(const std::string& name) { name_ = name; }
    
    DeviceStatus getStatus() const { return status_; }
    void setStatus(DeviceStatus status) { status_ = status; }
    
    bool isOperational() const { return status_ == DeviceStatus::OPERATIONAL; }
    
    // Device accuracy (standard deviation multiplier, e.g., 0.01 = 1% accuracy)
    Real getAccuracy() const { return accuracy_; }
    void setAccuracy(Real accuracy) { accuracy_ = accuracy; }
    
    // Get all measurements produced by this device
    const std::vector<MeasurementModel*>& getMeasurements() const { return measurements_; }
    
    // Add measurement produced by this device
    void addMeasurement(MeasurementModel* measurement);
    
    // Remove measurement
    void removeMeasurement(MeasurementModel* measurement);
    
    // Get device type as string (for debugging/logging)
    virtual std::string getDeviceType() const = 0;
    
protected:
    DeviceId id_;
    std::string name_;
    DeviceStatus status_;
    Real accuracy_;  // Standard deviation multiplier (0.01 = 1% accuracy)
    std::vector<MeasurementModel*> measurements_;  // Measurements produced by this device
};

// Multimeter: Measures power flow on branches using CT and PT
class Multimeter : public MeasurementDevice {
public:
    Multimeter(DeviceId id, BranchId branchId, BusId fromBus, BusId toBus,
               Real ctRatio = 1.0, Real ptRatio = 1.0, const std::string& name = "");
    
    BranchId getBranchId() const { return branchId_; }
    BusId getFromBus() const { return fromBus_; }
    BusId getToBus() const { return toBus_; }
    
    // Current Transformer (CT) ratio
    Real getCTRatio() const { return ctRatio_; }
    void setCTRatio(Real ratio) { ctRatio_ = ratio; }
    
    // Potential Transformer (PT) ratio
    Real getPTRatio() const { return ptRatio_; }
    void setPTRatio(Real ratio) { ptRatio_ = ratio; }
    
    // Get combined transformer ratio (CT * PT)
    Real getTransformerRatio() const { return ctRatio_ * ptRatio_; }
    
    // Apply transformer ratios to raw measurement value
    Real applyTransformerRatio(Real rawValue) const;
    
    // Reverse transformer ratios (for converting measured value back to raw)
    Real reverseTransformerRatio(Real measuredValue) const;
    
    std::string getDeviceType() const override { return "Multimeter"; }
    
private:
    BranchId branchId_;
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
    
    // Potential Transformer (PT) ratio
    Real getPTRatio() const { return ptRatio_; }
    void setPTRatio(Real ratio) { ptRatio_ = ratio; }
    
    // Apply PT ratio to raw measurement value
    Real applyPTRatio(Real rawValue) const;
    
    // Reverse PT ratio
    Real reversePTRatio(Real measuredValue) const;
    
    std::string getDeviceType() const override { return "Voltmeter"; }
    
private:
    BusId busId_;
    Real ptRatio_;   // Potential Transformer ratio (e.g., 1000:1 = 1000.0)
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_MEASUREMENTDEVICE_H

