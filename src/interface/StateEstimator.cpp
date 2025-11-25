/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/interface/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/math/Solver.h>
#include <sle/baddata/BadDataDetector.h>
#include <chrono>

namespace sle {
namespace interface {

StateEstimator::StateEstimator() 
    : solver_(std::make_unique<math::Solver>()),
      modelUpdated_(true), telemetryUpdated_(true) {
}

StateEstimator::~StateEstimator() = default;

void StateEstimator::setNetwork(std::shared_ptr<model::NetworkModel> network) {
    network_ = network;
    
    // Pass network to telemetry data for topology updates
    if (telemetry_) {
        telemetry_->setNetworkModel(network_.get());
    }
    
    markModelUpdated();
    
    // Reinitialize state if needed
    if (!currentState_ || currentState_->size() != network_->getBusCount()) {
        initializeState();
    }
}

std::shared_ptr<model::NetworkModel> StateEstimator::getNetwork() const {
    return network_;
}

void StateEstimator::setTelemetryData(std::shared_ptr<model::TelemetryData> telemetry) {
    telemetry_ = telemetry;
    
    // Set up topology change callback
    if (telemetry_) {
        telemetry_->setTopologyChangeCallback([this]() {
            markModelUpdated();
        });
        
        // Pass network if already set
        if (network_) {
            telemetry_->setNetworkModel(network_.get());
        }
    }
    
    telemetryUpdated_.store(true);
}

std::shared_ptr<model::TelemetryData> StateEstimator::getTelemetryData() const {
    return telemetry_;
}


StateEstimationResult StateEstimator::estimate() {
    return estimateInternal(solverConfig_);
}

StateEstimationResult StateEstimator::estimateIncremental() {
    // Use relaxed tolerance and fewer iterations for incremental updates (faster convergence)
    math::SolverConfig incrementalConfig = solverConfig_;
    incrementalConfig.tolerance = solverConfig_.tolerance * 10.0;  // Relaxed for speed
    incrementalConfig.maxIterations = std::min(solverConfig_.maxIterations, 10);
    
    return estimateInternal(incrementalConfig);
}

baddata::BadDataResult StateEstimator::detectBadData() {
    baddata::BadDataResult result;
    result.hasBadData = false;
    
    if (!network_ || !telemetry_ || !currentState_) {
        return result;
    }
    
    // Perform bad data detection using the solver's measurement functions
    // This reuses the GPU data (topology, state) from the last estimation
    // OPTIMIZATION: Pass residuals from last solve to avoid re-calculation
    std::vector<Real> residuals;
    solver_->getLastResiduals(residuals);
    
    baddata::BadDataDetector detector;
    result = detector.detectBadData(*telemetry_, *currentState_, *network_, 
                                   &solver_->getMeasurementFunctions(),
                                   residuals.empty() ? nullptr : &residuals);
    
    return result;
}

StateEstimationResult StateEstimator::estimateInternal(const math::SolverConfig& config) {
    if (!network_ || !telemetry_) {
        StateEstimationResult result;
        result.converged = false;
        result.message = "Network or telemetry data not set";
        return result;
    }
    
    // Initialize state if needed
    if (!currentState_ || currentState_->size() != network_->getBusCount()) {
        initializeState();
    }
    
    // Run estimation
    solver_->setConfig(config);
    
    // Check if we can reuse Jacobian structure (topology not updated)
    bool reuseStructure = !modelUpdated_.load() && !telemetryUpdated_.load();
    
    math::SolverResult solverResult = solver_->solve(*currentState_, *network_, *telemetry_, reuseStructure);
    
    StateEstimationResult result;
    result.converged = solverResult.converged;
    result.iterations = solverResult.iterations;
    result.finalNorm = solverResult.finalNorm;
    result.objectiveValue = solverResult.objectiveValue;
    result.message = solverResult.message;
    result.state = std::make_unique<model::StateVector>(*currentState_);
    result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Update all computed values from GPU (optimized - reuses GPU data from solver)
    if (network_) {
        solver_->storeComputedValues(*currentState_, *network_);
    }
    
    modelUpdated_.store(false);
    telemetryUpdated_.store(false);
    
    return result;
}

std::shared_ptr<model::StateVector> StateEstimator::getCurrentState() const {
    if (currentState_) {
        return std::make_shared<model::StateVector>(*currentState_);
    }
    return nullptr;
}

void StateEstimator::setSolverConfig(const math::SolverConfig& config) {
    solverConfig_ = config;
}

const math::SolverConfig& StateEstimator::getSolverConfig() const {
    return solverConfig_;
}

void StateEstimator::initializeState() {
    if (network_) {
        currentState_ = std::make_shared<model::StateVector>(network_->getBusCount());
        currentState_->initializeFromNetwork(*network_);
    }
}

void StateEstimator::configureForRealTime(Real tolerance, int maxIterations) {
    // CUDA-EXCLUSIVE: All operations use GPU
    math::SolverConfig config;
    config.tolerance = tolerance;
    config.maxIterations = maxIterations;
    setSolverConfig(config);
}

void StateEstimator::configureForOffline(Real tolerance, int maxIterations) {
    // CUDA-EXCLUSIVE: All operations use GPU
    math::SolverConfig config;
    config.tolerance = tolerance;
    config.maxIterations = maxIterations;
    setSolverConfig(config);
}

bool StateEstimator::loadFromFiles(const std::string& networkFile, const std::string& telemetryFile) {
    try {
        // Load network model
        auto network = ModelLoader::loadFromIEEE(networkFile);
        if (!network) {
            return false;
        }
        // NetworkModel is non-copyable (contains unique_ptr vectors), so convert to shared_ptr
        setNetwork(std::shared_ptr<model::NetworkModel>(network.release()));
        
        // Load telemetry data (use the network from the estimator)
        auto telemetry = MeasurementLoader::loadTelemetry(telemetryFile, *getNetwork());
        if (!telemetry) {
            return false;
        }
        // Convert unique_ptr to shared_ptr
        setTelemetryData(std::shared_ptr<model::TelemetryData>(telemetry.release()));
        
        return true;
    } catch (...) {
        return false;
    }
}

Real StateEstimator::getVoltageMagnitude(BusId busId) const {
    auto state = getCurrentState();
    if (!state || !network_) {
        return 0.0;
    }
    
    // Use efficient bus index lookup instead of linear search
    Index busIdx = network_->getBusIndex(busId);
    if (busIdx >= 0 && static_cast<size_t>(busIdx) < state->size()) {
        return state->getVoltageMagnitude(busIdx);
    }
    return 0.0;
}

Real StateEstimator::getVoltageAngle(BusId busId) const {
    auto state = getCurrentState();
    if (!state || !network_) {
        return 0.0;
    }
    
    // Use efficient bus index lookup instead of linear search
    Index busIdx = network_->getBusIndex(busId);
    if (busIdx >= 0 && static_cast<size_t>(busIdx) < state->size()) {
        return state->getVoltageAngle(busIdx);
    }
    return 0.0;
}

bool StateEstimator::isReady() const {
    return network_ != nullptr && telemetry_ != nullptr;
}

} // namespace interface
} // namespace sle
