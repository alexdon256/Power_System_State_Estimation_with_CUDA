/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/LoadFlow.h>
#include <sle/math/Solver.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/model/MeasurementDevice.h>
#include <sle/model/Bus.h>
#include <sle/Types.h>
#include <cmath>
#include <memory>
#include <iostream>

namespace sle {
namespace math {

LoadFlow::LoadFlow() : solver_(std::make_unique<Solver>()) {
    // Default config
    config_.tolerance = 1e-6;
    config_.maxIterations = 50;
    config_.useGPU = true;
}

LoadFlow::LoadFlow(const LoadFlowConfig& config) : config_(config), solver_(std::make_unique<Solver>()) {
}

LoadFlowResult LoadFlow::solve(const model::NetworkModel& network) {
    model::StateVector state(network.getBusCount());
    state.initializeFromNetwork(network);
    return solve(network, state);
}

LoadFlowResult LoadFlow::solve(const model::NetworkModel& network, const model::StateVector& initialState) {
    // Use Solver to solve Power Flow as a specialized State Estimation problem
    // Create pseudo-measurements from bus specifications
    
    model::TelemetryData telemetry;
    const auto& buses = network.getBuses();
    
    // High weight for power flow constraints (essentially hard constraints)
    // Use stdDev = 1e-4 (variance = 1e-8, weight = 1e8)
    const Real pfStdDev = 1e-4;
    
    for (const auto* bus : buses) {
        if (!bus) continue;
        
        // Create pseudo device for load flow measurements
        std::string deviceId = "PF_" + std::to_string(bus->getId());
        auto voltmeter = std::make_unique<model::Voltmeter>(
            deviceId, bus->getId(), 1.0, "Load Flow Pseudo Device Bus " + std::to_string(bus->getId())
        );
        telemetry.addDevice(std::move(voltmeter));
        
        if (bus->getType() == model::BusType::Slack) {
            // Slack Bus: Fixed V and Theta
            // V_mag measurement
            auto measV = std::make_unique<model::MeasurementModel>(
                model::MeasurementType::V_MAGNITUDE, bus->getVoltageMagnitude(), pfStdDev);
            telemetry.addMeasurementToDevice(deviceId, std::move(measV));
            
            // Angle measurement (V_ANGLE) to fix reference
            // Assuming 0 degrees for Slack unless specified otherwise (usually 0 in NetworkModel)
            Real angle = bus->getVoltageAngle(); // Usually 0
            auto measAng = std::make_unique<model::MeasurementModel>(
                model::MeasurementType::V_ANGLE, angle, pfStdDev);
            telemetry.addMeasurementToDevice(deviceId, std::move(measAng));
            
        } else if (bus->getType() == model::BusType::PV) {
            // PV Bus: Fixed P and V
            // P_injection = P_gen - P_load
            Real pInj = bus->getPGeneration() - bus->getPLoad();
            auto measP = std::make_unique<model::MeasurementModel>(
                model::MeasurementType::P_INJECTION, pInj, pfStdDev);
            telemetry.addMeasurementToDevice(deviceId, std::move(measP));
            
            // V_mag measurement
            auto measV = std::make_unique<model::MeasurementModel>(
                model::MeasurementType::V_MAGNITUDE, bus->getVoltageMagnitude(), pfStdDev);
            telemetry.addMeasurementToDevice(deviceId, std::move(measV));
            
        } else {
            // PQ Bus: Fixed P and Q
            // P_injection = P_gen - P_load
            Real pInj = bus->getPGeneration() - bus->getPLoad();
            auto measP = std::make_unique<model::MeasurementModel>(
                model::MeasurementType::P_INJECTION, pInj, pfStdDev);
            telemetry.addMeasurementToDevice(deviceId, std::move(measP));
            
            // Q_injection = Q_gen - Q_load
            Real qInj = bus->getQGeneration() - bus->getQLoad();
            auto measQ = std::make_unique<model::MeasurementModel>(
                model::MeasurementType::Q_INJECTION, qInj, pfStdDev);
            telemetry.addMeasurementToDevice(deviceId, std::move(measQ));
        }
    }
    
    // Configure Solver
    SolverConfig solverConfig;
    solverConfig.maxIterations = config_.maxIterations;
    solverConfig.tolerance = config_.tolerance;
    solverConfig.useGPU = config_.useGPU;
    // No regularization for pure Newton-Raphson simulation
    
    // Run Solver (reusing persistent instance)
    if (!solver_) {
        solver_ = std::make_unique<Solver>();
    }
    solver_->setConfig(solverConfig);
    
    // Initialize state
    model::StateVector state = initialState;
    
    SolverResult solverResult = solver_->solve(state, network, telemetry);
    
    LoadFlowResult result;
    result.converged = solverResult.converged;
    result.iterations = solverResult.iterations;
    result.finalMismatch = solverResult.finalCost; // Approximation of mismatch
    result.state = std::make_unique<model::StateVector>(state);
    
    if (result.converged) {
        result.message = "Converged";
    } else {
        result.message = "Did not converge (Solver status: " + 
                        std::string(solverResult.converged ? "Success" : "Failed") + ")";
    }
    
    // Compute actual mismatches for report
    std::vector<Real> pMis, qMis;
    computeMismatches(network, state, pMis, qMis);
    result.busMismatches.clear();
    result.busMismatches.reserve(pMis.size() + qMis.size());
    result.busMismatches.insert(result.busMismatches.end(), pMis.begin(), pMis.end());
    result.busMismatches.insert(result.busMismatches.end(), qMis.begin(), qMis.end());
    
    // Update finalMismatch to be the max absolute mismatch
    Real maxMis = 0.0;
    for (Real val : result.busMismatches) {
        maxMis = std::max(maxMis, std::abs(val));
    }
    result.finalMismatch = maxMis;

    return result;
}

void LoadFlow::computeMismatches(const model::NetworkModel& network, const model::StateVector& state,
                                std::vector<Real>& pMismatch,
                                std::vector<Real>& qMismatch) const {
    size_t nBuses = network.getBusCount();
    
    if (nBuses == 0) {
        pMismatch.clear();
        qMismatch.clear();
        return;
    }
    
    pMismatch.resize(nBuses);
    qMismatch.resize(nBuses);
    
    // Use NetworkModel to compute injections (GPU or CPU)
    // But NetworkModel::computePowerInjections updates internal state.
    // We can use that.
    model::NetworkModel& modNetwork = const_cast<model::NetworkModel&>(network);
    modNetwork.computePowerInjections(state);
    
    const auto& buses = network.getBuses();
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < nBuses; ++i) {
        const auto* bus = buses[i];
        if (!bus) continue;
        
        // Mismatch = Specified - Calculated
        // P_spec = P_gen - P_load
        Real pSpec = bus->getPGeneration() - bus->getPLoad();
        Real qSpec = bus->getQGeneration() - bus->getQLoad();
        
        // P_calc = P_injection (computed by model)
        Real pCalc = bus->getPInjection();
        Real qCalc = bus->getQInjection();
        
        pMismatch[i] = pSpec - pCalc;
        qMismatch[i] = qSpec - qCalc;
    }
}

// Forward to main solve method
LoadFlowResult LoadFlow::solveNewtonRaphson(const model::NetworkModel* network,
                                           const model::StateVector* initialState) {
    if (!network) return LoadFlowResult{false, 0, 0.0, nullptr, {}, "Invalid network"};
    if (initialState) {
        return solve(*network, *initialState);
    } else {
        return solve(*network);
    }
}

LoadFlowResult LoadFlow::solveFastDecoupled(const model::NetworkModel* network,
                                            const model::StateVector* initialState) {
    // Fallback to Newton-Raphson (Solver)
    return solveNewtonRaphson(network, initialState);
}

} // namespace math
} // namespace sle



