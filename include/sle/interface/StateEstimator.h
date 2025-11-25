/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_INTERFACE_STATEESTIMATOR_H
#define SLE_INTERFACE_STATEESTIMATOR_H

#include <sle/Export.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/math/Solver.h>
#include <sle/baddata/BadDataDetector.h> // Added for bad data result type
#include <memory>
#include <atomic>

namespace sle {
namespace interface {

struct SLE_API StateEstimationResult {
    bool converged;
    int iterations;
    double finalNorm;
    double objectiveValue;
    std::unique_ptr<model::StateVector> state;
    std::string message;
    int64_t timestamp;
};

class SLE_API StateEstimator {
public:
    StateEstimator();
    ~StateEstimator();
    
    // Set network model (can be updated in real-time)
    void setNetwork(std::shared_ptr<model::NetworkModel> network);
    std::shared_ptr<model::NetworkModel> getNetwork() const;
    
    // Set telemetry data (can be updated in real-time)
    void setTelemetryData(std::shared_ptr<model::TelemetryData> telemetry);
    std::shared_ptr<model::TelemetryData> getTelemetryData() const;
    
    // Run state estimation
    StateEstimationResult estimate();
    
    // Run state estimation with current state as initial guess (faster for real-time)
    StateEstimationResult estimateIncremental();
    
    // Detect bad data using results from last estimation
    // Uses GPU data from the solver to avoid re-evaluation
    baddata::BadDataResult detectBadData();
    
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
    
    // Convenience methods for easier usage
    // Configure for real-time operation (fast, relaxed tolerance)
    // CUDA-EXCLUSIVE: All operations use GPU
    void configureForRealTime(Real tolerance = 1e-5, int maxIterations = 15);
    
    // Configure for offline analysis (accurate, tight tolerance)
    // CUDA-EXCLUSIVE: All operations use GPU
    void configureForOffline(Real tolerance = 1e-8, int maxIterations = 50);
    
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
    
    std::atomic<bool> modelUpdated_;
    std::atomic<bool> telemetryUpdated_;
    
    math::SolverConfig solverConfig_;
    
    void initializeState();
    
    // Internal estimation helper
    StateEstimationResult estimateInternal(const math::SolverConfig& config);
};

} // namespace interface
} // namespace sle

#endif // SLE_INTERFACE_STATEESTIMATOR_H
