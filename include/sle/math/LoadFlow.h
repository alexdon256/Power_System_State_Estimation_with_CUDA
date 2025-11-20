/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_LOADFLOW_H
#define SLE_MATH_LOADFLOW_H

#include <sle/Export.h>
#include <sle/Types.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <memory>

namespace sle {
namespace math {

struct SLE_API LoadFlowConfig {
    Real tolerance = 1e-6;
    Index maxIterations = 100;
    Real accelerationFactor = 1.0;
    bool useFastDecoupled = false;
    bool useGPU = true;
};

struct SLE_API LoadFlowResult {
    bool converged;
    Index iterations;
    Real finalMismatch;
    std::unique_ptr<StateVector> state;
    std::vector<Real> busMismatches;  // P and Q mismatches per bus
    std::string message;
};

class SLE_API LoadFlow {
public:
    LoadFlow();
    explicit LoadFlow(const LoadFlowConfig& config);
    
    void setConfig(const LoadFlowConfig& config) { config_ = config; }
    const LoadFlowConfig& getConfig() const { return config_; }
    
    // Run load flow
    LoadFlowResult solve(const NetworkModel& network);
    
    // Run load flow with initial state
    LoadFlowResult solve(const NetworkModel& network, const StateVector& initialState);
    
    // Compute power mismatches
    void computeMismatches(const NetworkModel& network, const StateVector& state,
                          std::vector<Real>& pMismatch, std::vector<Real>& qMismatch) const;
    
private:
    LoadFlowConfig config_;
    
    // Newton-Raphson load flow
    LoadFlowResult solveNewtonRaphson(const NetworkModel& network, const StateVector* initialState);
    
    // Fast decoupled load flow
    LoadFlowResult solveFastDecoupled(const NetworkModel& network, const StateVector* initialState);
    
    // Build power flow Jacobian
    void buildPowerFlowJacobian(const NetworkModel& network, const StateVector& state,
                               std::vector<Complex>& J, std::vector<Index>& rowPtr,
                               std::vector<Index>& colInd) const;
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_LOADFLOW_H

