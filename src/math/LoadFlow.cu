/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/LoadFlow.h>
#include <sle/math/SparseMatrix.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/Bus.h>
#include <sle/Types.h>
#include <cmath>
#include <algorithm>
#include <complex>

namespace sle {
namespace math {

LoadFlow::LoadFlow() {
    config_.tolerance = 1e-6;
    config_.maxIterations = 100;
    config_.useFastDecoupled = false;
}

LoadFlow::LoadFlow(const LoadFlowConfig& config) : config_(config) {
}

using model::NetworkModel;
using model::StateVector;
using model::Bus;

LoadFlowResult LoadFlow::solve(const NetworkModel& network) {
    return solve(network, StateVector());
}

LoadFlowResult LoadFlow::solve(const NetworkModel& network, const StateVector& initialState) {
    if (config_.useFastDecoupled) {
        return solveFastDecoupled(&network, &initialState);
    } else {
        return solveNewtonRaphson(&network, &initialState);
    }
}

LoadFlowResult LoadFlow::solveNewtonRaphson(const NetworkModel& network,
                                           const StateVector* initialState) {
    LoadFlowResult result;
    result.converged = false;
    result.iterations = 0;
    
    size_t nBuses = network.getBusCount();
    StateVector state(nBuses);
    
    if (initialState && initialState->size() == nBuses) {
        state = *initialState;
    } else {
        state.initializeFromNetwork(network);
    }
    
    std::vector<Real> pMismatch(nBuses);
    std::vector<Real> qMismatch(nBuses);
    
    for (Index iter = 0; iter < config_.maxIterations; ++iter) {
        // Compute power mismatches
        computeMismatches(network, state, pMismatch, qMismatch);
        
        // Check convergence
        Real maxMismatch = 0.0;
        for (size_t i = 0; i < nBuses; ++i) {
            maxMismatch = std::max(maxMismatch, std::abs(pMismatch[i]));
            maxMismatch = std::max(maxMismatch, std::abs(qMismatch[i]));
        }
        
        result.finalMismatch = maxMismatch;
        result.busMismatches = pMismatch;
        result.busMismatches.insert(result.busMismatches.end(),
                                    qMismatch.begin(), qMismatch.end());
        
        if (maxMismatch < config_.tolerance) {
            result.converged = true;
            result.iterations = iter + 1;
            result.message = "Converged";
            break;
        }
        
        // Build Jacobian and solve for state update
        std::vector<Complex> J;
        std::vector<Index> rowPtr, colInd;
        buildPowerFlowJacobian(network, state, J, rowPtr, colInd);
        
        // Solve linear system (simplified - would use cuSOLVER)
        // For now, use simplified update
        const auto& angles = state.getAngles();
        const auto& magnitudes = state.getMagnitudes();
        
        for (size_t i = 0; i < nBuses; ++i) {
            // Get bus by index (simplified - would use proper bus ID mapping)
            auto buses = network.getBuses();
            if (i < buses.size()) {
                auto* bus = buses[i];
                if (bus && bus->getType() != BusType::Slack) {
                    // Update angle
                    Real deltaAngle = -pMismatch[i] * 0.1;  // Simplified
                    state.setVoltageAngle(i, angles[i] + deltaAngle);
                    
                    if (bus->getType() == BusType::PQ) {
                        // Update magnitude
                        Real deltaV = -qMismatch[i] * 0.1;  // Simplified
                        state.setVoltageMagnitude(i, magnitudes[i] + deltaV);
                    }
                }
            }
        }
    }
    
    if (!result.converged) {
        result.iterations = config_.maxIterations;
        result.message = "Maximum iterations reached";
    }
    
    result.state = std::make_unique<StateVector>(state);
    
    return result;
}

LoadFlowResult LoadFlow::solveFastDecoupled(const NetworkModel& network,
                                            const StateVector* initialState) {
    // Fast decoupled load flow implementation
    // Simplified version - full implementation would separate P-θ and Q-V updates
    return solveNewtonRaphson(&network, initialState);
}

void LoadFlow::computeMismatches(const NetworkModel& network, const StateVector& state,
                                std::vector<Real>& pMismatch,
                                std::vector<Real>& qMismatch) const {
    size_t nBuses = network.getBusCount();
    
    // Handle empty network
    if (nBuses == 0) {
        pMismatch.clear();
        qMismatch.clear();
        return;
    }
    
    // Resize mismatch vectors if needed
    if (pMismatch.size() != nBuses) {
        pMismatch.resize(nBuses);
    }
    if (qMismatch.size() != nBuses) {
        qMismatch.resize(nBuses);
    }
    
    // Compute power injections once and store directly in bus objects
    model::NetworkModel& modNetwork = const_cast<model::NetworkModel&>(network);
    modNetwork.computePowerInjections(state, config_.useGPU);
    
    // Compute mismatches: P_mismatch = P_gen - P_load - P_injection
    //                     Q_mismatch = Q_gen - Q_load - Q_injection
    auto buses = network.getBuses();
    for (size_t i = 0; i < nBuses && i < buses.size(); ++i) {
        const Bus* bus = buses[i];
        if (!bus) {
            pMismatch[i] = 0.0;
            qMismatch[i] = 0.0;
            continue;
        }
        pMismatch[i] = bus->getPGeneration() - bus->getPLoad() - bus->getPInjection();
        qMismatch[i] = bus->getQGeneration() - bus->getQLoad() - bus->getQInjection();
    }
}

void LoadFlow::buildPowerFlowJacobian(const NetworkModel& network, const StateVector& state,
                                     std::vector<Complex>& J,
                                     std::vector<Index>& rowPtr,
                                     std::vector<Index>& colInd) const {
    // Build power flow Jacobian matrix
    // This is a simplified version - full implementation would properly construct
    // the Jacobian for P-θ and Q-V relationships
    
    size_t nBuses = network.getBusCount();
    size_t nStates = 2 * nBuses;
    
    // Only resize rowPtr if size changed (avoid unnecessary reallocation)
    if (rowPtr.size() != nStates + 1) {
        rowPtr.clear();
        rowPtr.resize(nStates + 1, 0);
    } else {
        // Zero-initialize existing vector (faster than resize)
        std::fill(rowPtr.begin(), rowPtr.end(), 0);
    }
    colInd.clear();
    J.clear();
    
    // Simplified structure - would need proper Jacobian computation
    for (size_t i = 0; i < nStates; ++i) {
        rowPtr[i + 1] = rowPtr[i] + nStates;
        for (size_t j = 0; j < nStates; ++j) {
            colInd.push_back(j);
            J.push_back(Complex(0.0, 0.0));
        }
    }
}

} // namespace math
} // namespace sle

