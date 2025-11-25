/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_BUS_H
#define SLE_MODEL_BUS_H

#include <sle/Types.h>
#include <string>
#include <vector>
#include <memory>

// Forward declarations
namespace sle {
namespace model {
    class TelemetryData;
    class MeasurementDevice;
    class MeasurementModel;
}
}

namespace sle {
namespace model {

class Bus {
public:
    Bus(BusId id, const std::string& name = "");
    
    // Copy assignment operator
    Bus& operator=(const Bus& other);
    
    BusId getId() const { return id_; }
    const std::string& getName() const { return name_; }
    
    void setType(BusType type) { type_ = type; }
    BusType getType() const { return type_; }
    
    void setBaseKV(Real baseKV) { baseKV_ = baseKV; }
    Real getBaseKV() const { return baseKV_; }
    
    void setVoltage(Real magnitude, Real angle = 0.0);
    Real getVoltageMagnitude() const { return voltageMag_; }
    Real getVoltageAngle() const { return voltageAngle_; }
    
    void setLoad(Real pLoad, Real qLoad);
    Real getPLoad() const { return pLoad_; }
    Real getQLoad() const { return qLoad_; }
    
    void setGeneration(Real pGen, Real qGen);
    Real getPGeneration() const { return pGen_; }
    Real getQGeneration() const { return qGen_; }
    
    void setShunt(Real gShunt, Real bShunt);
    Real getGShunt() const { return gShunt_; }
    Real getBShunt() const { return bShunt_; }
    
    bool isZeroInjection() const;
    
    // Getters for computed voltage estimates (set by Solver::storeComputedValues)
    Real getVPU() const { return vPU_; }              // Voltage in per-unit
    Real getVKV() const { return vKV_; }              // Voltage in kV
    Real getThetaRad() const { return thetaRad_; }     // Angle in radians
    Real getThetaDeg() const { return thetaDeg_; }     // Angle in degrees
    
    // Getters for computed power injections (set by computePowerInjections or Solver::storeComputedValues)
    Real getPInjection() const { return pInjection_; }        // P injection in p.u.
    Real getQInjection() const { return qInjection_; }        // Q injection in p.u.
    Real getPInjectionMW() const { return pInjectionMW_; }    // P injection in MW
    Real getQInjectionMVAR() const { return qInjectionMVAR_; } // Q injection in MVAR
    
    // Get telemetry from associated devices (requires TelemetryData reference)
    // Returns all devices (voltmeters) associated with this bus
    std::vector<const MeasurementDevice*> getAssociatedDevices(const TelemetryData& telemetry) const;
    
    // Get all measurements from devices associated with this bus
    // NOTE: Always queries latest values from telemetry - reflects real-time updates
    std::vector<const MeasurementModel*> getMeasurementsFromDevices(const TelemetryData& telemetry) const;
    
    // Get specific measurement type from associated devices
    // NOTE: Always queries latest values from telemetry - reflects real-time updates
    std::vector<const MeasurementModel*> getMeasurementsFromDevices(const TelemetryData& telemetry, MeasurementType type) const;
    
    // Convenience methods: Get current measurement values directly
    // These methods query telemetry each time, so they always return the latest values
    
    // Get current voltage magnitude measurement (from voltmeter)
    // Returns measurement value if found, NaN if no measurement available
    Real getCurrentVoltageMeasurement(const TelemetryData& telemetry) const;
    
    // Get current power injection measurements (P and Q)
    // Returns true if measurements found, false otherwise
    bool getCurrentPowerInjections(const TelemetryData& telemetry, Real& pInjection, Real& qInjection) const;
    
private:
    // Internal setters (used by NetworkModel)
    friend class NetworkModel;
    void setVoltEstimates(Real vPU, Real vKV, Real thetaRad, Real thetaDeg);
    void setPowerInjections(Real pInj, Real qInj, Real pMW, Real qMVAR);
    BusId id_;
    std::string name_;
    BusType type_;
    Real baseKV_;
    
    Real voltageMag_;
    Real voltageAngle_;
    
    Real pLoad_;
    Real qLoad_;
    Real pGen_;
    Real qGen_;
    
    Real gShunt_;
    Real bShunt_;
    
    // Computed values (set by NetworkModel compute methods)
    Real vPU_;              // Voltage in per-unit (from state estimation)
    Real vKV_;              // Voltage in kV
    Real thetaRad_;         // Angle in radians
    Real thetaDeg_;         // Angle in degrees
    
    Real pInjection_;       // P injection in p.u.
    Real qInjection_;       // Q injection in p.u.
    Real pInjectionMW_;     // P injection in MW
    Real qInjectionMVAR_;   // Q injection in MVAR
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_BUS_H

