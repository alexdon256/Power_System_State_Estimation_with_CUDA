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
#include <chrono>

namespace sle {
namespace interface {

StateEstimator::StateEstimator() 
    : solver_(std::make_unique<math::Solver>()),
      modelUpdated_(true), telemetryUpdated_(true) {
    telemetryProcessor_.setTelemetryData(telemetry_.get());
}

StateEstimator::~StateEstimator() {
    telemetryProcessor_.stopRealTimeProcessing();
}

void StateEstimator::setNetwork(std::shared_ptr<model::NetworkModel> network) {
    network_ = network;
    modelUpdated_.store(true);
    
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
    telemetryProcessor_.setTelemetryData(telemetry_.get());
    telemetryUpdated_.store(true);
}

std::shared_ptr<model::TelemetryData> StateEstimator::getTelemetryData() const {
    return telemetry_;
}

void StateEstimator::updateNetworkModel(std::shared_ptr<model::NetworkModel> network) {
    setNetwork(network);
}

void StateEstimator::updateTelemetryData(std::shared_ptr<model::TelemetryData> telemetry) {
    setTelemetryData(telemetry);
}

StateEstimationResult StateEstimator::estimate() {
    if (!network_ || !telemetry_) {
        StateEstimationResult result;
        result.converged = false;
        result.message = "Network or telemetry data not set";
        return result;
    }
    
    // Process any pending telemetry updates
    telemetryProcessor_.processUpdateQueue();
    
    // Initialize state if needed
    if (!currentState_ || currentState_->size() != network_->getBusCount()) {
        initializeState();
    }
    
    // Run estimation
    solver_->setConfig(solverConfig_);
    math::SolverResult solverResult = solver_->solve(*currentState_, *network_, *telemetry_);
    
    StateEstimationResult result;
    result.converged = solverResult.converged;
    result.iterations = solverResult.iterations;
    result.finalNorm = solverResult.finalNorm;
    result.objectiveValue = solverResult.objectiveValue;
    result.message = solverResult.message;
    result.state = std::make_unique<model::StateVector>(*currentState_);
    result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Update bus voltage estimates so downstream consumers don't need to call explicitly
    if (network_) {
        network_->computeVoltEstimates(*currentState_, solverConfig_.useGPU);
    }
    
    modelUpdated_.store(false);
    telemetryUpdated_.store(false);
    
    return result;
}

StateEstimationResult StateEstimator::estimateIncremental() {
    // Use current state as initial guess for faster convergence
    // This is optimized for real-time updates where state changes are small
    if (!network_ || !telemetry_) {
        StateEstimationResult result;
        result.converged = false;
        result.message = "Network or telemetry data not set";
        return result;
    }
    
    // Process pending updates
    telemetryProcessor_.processUpdateQueue();
    
    // Use relaxed tolerance and fewer iterations for incremental updates (faster convergence)
    math::SolverConfig incrementalConfig = solverConfig_;
    incrementalConfig.tolerance = solverConfig_.tolerance * 10.0;  // Relaxed for speed
    incrementalConfig.maxIterations = std::min(solverConfig_.maxIterations, 10);
    
    solver_->setConfig(incrementalConfig);
    math::SolverResult solverResult = solver_->solve(*currentState_, *network_, *telemetry_);
    
    StateEstimationResult result;
    result.converged = solverResult.converged;
    result.iterations = solverResult.iterations;
    result.finalNorm = solverResult.finalNorm;
    result.objectiveValue = solverResult.objectiveValue;
    result.message = solverResult.message;
    result.state = std::make_unique<model::StateVector>(*currentState_);
    result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    if (network_) {
        network_->computeVoltEstimates(*currentState_, incrementalConfig.useGPU);
    }
    
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

void StateEstimator::configureForRealTime(Real tolerance, int maxIterations, bool useGPU) {
    math::SolverConfig config;
    config.tolerance = tolerance;
    config.maxIterations = maxIterations;
    config.useGPU = useGPU;
    setSolverConfig(config);
}

void StateEstimator::configureForOffline(Real tolerance, int maxIterations, bool useGPU) {
    math::SolverConfig config;
    config.tolerance = tolerance;
    config.maxIterations = maxIterations;
    config.useGPU = useGPU;
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

