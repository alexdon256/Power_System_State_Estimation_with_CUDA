/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_BRANCH_H
#define SLE_MODEL_BRANCH_H

#include <sle/Types.h>
#include <memory>

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
    
    // Compute power flow from estimated state
    // Returns power flow from fromBus to toBus
    void computePowerFlow(const StateVector& state, Index fromBusIdx, Index toBusIdx,
                         Real& pFlow, Real& qFlow) const;
    
    // Get power flow in physical units (MW, MVAR)
    void computePowerFlowMW(const StateVector& state, Index fromBusIdx, Index toBusIdx,
                           Real baseMVA, Real& pMW, Real& qMVAR) const;
    
    // Compute current magnitude from power flow
    Real computeCurrentMagnitude(Real pFlow, Real qFlow, Real vFrom) const;
    
    // Getters for computed power flows (set by NetworkModel::computePowerFlows)
    Real getPFlow() const { return pFlow_; }           // P flow in p.u.
    Real getQFlow() const { return qFlow_; }          // Q flow in p.u.
    Real getPMW() const { return pMW_; }              // P flow in MW
    Real getQMVAR() const { return qMVAR_; }          // Q flow in MVAR
    Real getIAmps() const { return iAmps_; }          // Current in Amperes
    Real getIPU() const { return iPU_; }             // Current in per-unit
    
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
    
    // Computed values (set by NetworkModel computePowerFlows)
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

