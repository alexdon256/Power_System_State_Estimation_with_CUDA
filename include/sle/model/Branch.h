/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_BRANCH_H
#define SLE_MODEL_BRANCH_H

#include <sle/Types.h>
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

class StateVector;

class Branch {
public:
    Branch(BranchId id, BusId fromBus, BusId toBus);
    
    // Copy assignment operator
    Branch& operator=(const Branch& other);
    
    BranchId getId() const { return id_; }
    BusId getFromBus() const { return fromBus_; }
    BusId getToBus() const { return toBus_; }
    
    void setImpedance(Real r, Real x);
    Real getR() const { return r_; }
    Real getX() const { return x_; }
    
    void setCharging(Real b) { b_ = b; }
    Real getB() const { return b_; }
    
    void setRating(Real mvaRating) { mvaRating_ = mvaRating; }
    Real getRating() const { return mvaRating_; }
    
    void setTapRatio(Real tap) { tapRatio_ = tap; }
    Real getTapRatio() const { return tapRatio_; }
    
    void setPhaseShift(Real shift) { phaseShift_ = shift; }
    Real getPhaseShift() const { return phaseShift_; }
    
    bool isTransformer() const { return std::abs(tapRatio_ - 1.0) > 1e-6 || std::abs(phaseShift_) > 1e-6; }
    
    Complex getAdmittance() const;
    
    // Getters for computed power flows (set by Solver::storeComputedValues)
    Real getPFlow() const { return pFlow_; }           // P flow in p.u.
    Real getQFlow() const { return qFlow_; }          // Q flow in p.u.
    Real getPMW() const { return pMW_; }              // P flow in MW
    Real getQMVAR() const { return qMVAR_; }          // Q flow in MVAR
    Real getIAmps() const { return iAmps_; }          // Current in Amperes
    Real getIPU() const { return iPU_; }             // Current in per-unit
    
    // Status management
    void setStatus(bool closed) { status_ = closed; }
    bool getStatus() const { return status_; }
    bool isOn() const { return status_; }
    
    // Get telemetry from associated devices (requires TelemetryData reference)
    // Returns all devices (multimeters) associated with this branch
    std::vector<const MeasurementDevice*> getAssociatedDevices(const TelemetryData& telemetry) const;
    
    // Get all measurements from devices associated with this branch
    // NOTE: Always queries latest values from telemetry - reflects real-time updates
    std::vector<const MeasurementModel*> getMeasurementsFromDevices(const TelemetryData& telemetry) const;
    
    // Get specific measurement type from associated devices
    // NOTE: Always queries latest values from telemetry - reflects real-time updates
    std::vector<const MeasurementModel*> getMeasurementsFromDevices(const TelemetryData& telemetry, MeasurementType type) const;
    
    // Convenience methods: Get current measurement values directly
    // These methods query telemetry each time, so they always return the latest values
    
    // Get current power flow measurements (P and Q)
    // Returns true if measurements found, false otherwise
    bool getCurrentPowerFlow(const TelemetryData& telemetry, Real& pFlow, Real& qFlow) const;
    
    // Get current current magnitude measurement
    // Returns measurement value if found, NaN if no measurement available
    Real getCurrentCurrentMeasurement(const TelemetryData& telemetry) const;
    
private:
    // Internal setters (used by NetworkModel)
    friend class NetworkModel;
    void setPowerFlow(Real p, Real q, Real pMW, Real qMVAR, Real iAmps, Real iPU);
    BranchId id_;
    BusId fromBus_;
    BusId toBus_;
    
    Real r_;           // Resistance (p.u.)
    Real x_;           // Reactance (p.u.)
    Real b_;           // Charging susceptance (p.u.)
    Real mvaRating_;   // MVA rating
    
    Real tapRatio_;    // Transformer tap ratio (1.0 for lines)
    Real phaseShift_;  // Phase shift angle (radians)
    
    bool status_;      // true = Closed (In Service), false = Open (Out of Service)
    
    // Computed values (set by Solver::storeComputedValues)
    Real pFlow_;       // P flow in p.u.
    Real qFlow_;       // Q flow in p.u.
    Real pMW_;         // P flow in MW
    Real qMVAR_;       // Q flow in MVAR
    Real iAmps_;       // Current in Amperes
    Real iPU_;         // Current in per-unit
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_BRANCH_H

