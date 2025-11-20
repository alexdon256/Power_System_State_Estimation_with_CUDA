/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_INTERFACE_STATEESTIMATOR_H
#define SLE_INTERFACE_STATEESTIMATOR_H

#include <sle/model/NetworkModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/math/Solver.h>
#include <sle/interface/TelemetryProcessor.h>
#include <memory>
#include <mutex>
#include <atomic>

namespace sle {
namespace interface {

struct StateEstimationResult {
    bool converged;
    int iterations;
    double finalNorm;
    double objectiveValue;
    std::unique_ptr<model::StateVector> state;
    std::string message;
    int64_t timestamp;
};

class StateEstimator {
public:
    StateEstimator();
    ~StateEstimator();
    
    // Set network model (can be updated in real-time)
    void setNetwork(std::shared_ptr<model::NetworkModel> network);
    std::shared_ptr<model::NetworkModel> getNetwork() const;
    
    // Set telemetry data (can be updated in real-time)
    void setTelemetryData(std::shared_ptr<model::TelemetryData> telemetry);
    std::shared_ptr<model::TelemetryData> getTelemetryData() const;
    
    // Get telemetry processor for real-time updates
    TelemetryProcessor& getTelemetryProcessor() { return telemetryProcessor_; }
    
    // Run state estimation
    StateEstimationResult estimate();
    
    // Run state estimation with current state as initial guess (faster for real-time)
    StateEstimationResult estimateIncremental();
    
    // Check if model or measurements have been updated
    bool isModelUpdated() const { return modelUpdated_.load(); }
    bool isTelemetryUpdated() const { return telemetryUpdated_.load(); }
    
    // Mark model/telemetry as updated (called automatically on updates)
    void markModelUpdated() { modelUpdated_.store(true); }
    void markTelemetryUpdated() { telemetryUpdated_.store(true); }
    
    // Get current state estimate
    std::shared_ptr<model::StateVector> getCurrentState() const;
    
    // Configuration
    void setSolverConfig(const math::SolverConfig& config);
    const math::SolverConfig& getSolverConfig() const;
    
    // Real-time update methods
    void updateNetworkModel(std::shared_ptr<model::NetworkModel> network);
    void updateTelemetryData(std::shared_ptr<model::TelemetryData> telemetry);
    
    // Convenience methods for easier usage
    // Configure for real-time operation (fast, relaxed tolerance)
    void configureForRealTime(Real tolerance = 1e-5, int maxIterations = 15, bool useGPU = true);
    
    // Configure for offline analysis (accurate, tight tolerance)
    void configureForOffline(Real tolerance = 1e-8, int maxIterations = 50, bool useGPU = true);
    
    // Quick setup: load network and telemetry from files
    bool loadFromFiles(const std::string& networkFile, const std::string& telemetryFile);
    
    // Get voltage at a specific bus (convenience wrapper)
    Real getVoltageMagnitude(BusId busId) const;
    Real getVoltageAngle(BusId busId) const;
    
    // Check if estimation is ready (has network and telemetry)
    bool isReady() const;
    
private:
    std::shared_ptr<model::NetworkModel> network_;
    std::shared_ptr<model::TelemetryData> telemetry_;
    std::shared_ptr<model::StateVector> currentState_;
    
    std::unique_ptr<math::Solver> solver_;
    TelemetryProcessor telemetryProcessor_;
    
    std::mutex estimationMutex_;
    std::atomic<bool> modelUpdated_;
    std::atomic<bool> telemetryUpdated_;
    
    math::SolverConfig solverConfig_;
    
    void initializeState();
};

} // namespace interface
} // namespace sle

#endif // SLE_INTERFACE_STATEESTIMATOR_H

