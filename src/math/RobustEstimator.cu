/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/RobustEstimator.h>
#include <sle/math/Solver.h>
#include <sle/math/MeasurementFunctions.h>
#include <sle/math/JacobianMatrix.h>
#include <cmath>
#include <algorithm>

namespace sle {
namespace math {

using model::StateVector;
using model::NetworkModel;
using model::TelemetryData;
using model::MeasurementModel;

RobustEstimator::RobustEstimator() {
    config_.weightFunction = RobustWeightFunction::HUBER;
    config_.tuningConstant = 1.345;
}

RobustEstimator::RobustEstimator(const RobustEstimatorConfig& config)
    : config_(config) {
}

RobustEstimator::RobustResult RobustEstimator::estimate(
    StateVector& state, const NetworkModel& network,
    const TelemetryData& telemetry) {
    
    return solveIRLS(state, network, telemetry);
}

RobustEstimator::RobustResult RobustEstimator::solveIRLS(
    StateVector& state, const NetworkModel& network,
    const TelemetryData& telemetry) {
    
    RobustResult result;
    result.converged = false;
    result.iterations = 0;
    
    const auto& measurements = telemetry.getMeasurements();
    size_t nMeas = measurements.size();
    
    // Initial weights (standard WLS)
    std::vector<Real> weights(nMeas);
    for (size_t i = 0; i < nMeas; ++i) {
        weights[i] = measurements[i]->getWeight();
    }
    
    MeasurementFunctions measFuncs;
    JacobianMatrix jacobian;
    Solver solver;
    
    SolverConfig solverConfig;
    solverConfig.tolerance = config_.tolerance;
    solverConfig.maxIterations = 10;  // Fewer iterations per IRLS step
    // CUDA-EXCLUSIVE: All operations use GPU
    solver.setConfig(solverConfig);
    
    // IRLS iterations
    for (Index irlsIter = 0; irlsIter < config_.maxIterations; ++irlsIter) {
        // Create weighted telemetry with current robust weights
        auto weightedTelemetry = std::make_unique<TelemetryData>();
        for (size_t i = 0; i < measurements.size(); ++i) {
            Real robustWeight = weights[i];
            Real effectiveStdDev = measurements[i]->getStdDev() / sqrt(robustWeight);
            
            auto meas = std::make_unique<MeasurementModel>(
                measurements[i]->getType(),
                measurements[i]->getValue(),
                effectiveStdDev,
                measurements[i]->getDeviceId());
            
            if (measurements[i]->getLocation() >= 0) {
                meas->setLocation(measurements[i]->getLocation());
            }
            if (measurements[i]->getFromBus() >= 0) {
                meas->setBranchLocation(measurements[i]->getFromBus(),
                                       measurements[i]->getToBus());
            }
            
            weightedTelemetry->addMeasurement(std::move(meas));
        }
        
        // Solve weighted least squares
        SolverResult solverResult = solver.solve(state, network, *weightedTelemetry);
        
        if (!solverResult.converged) {
            result.message = "Solver did not converge in IRLS iteration " + 
                           std::to_string(irlsIter);
            break;
        }
        
        // CUDA-EXCLUSIVE: Compute residuals
        std::vector<Real> hx;
        measFuncs.evaluate(state, network, telemetry, hx);
        
        std::vector<Real> residuals(nMeas);
        std::vector<Real> stdDevs(nMeas);
        for (size_t i = 0; i < nMeas && i < hx.size(); ++i) {
            residuals[i] = measurements[i]->getValue() - hx[i];
            stdDevs[i] = measurements[i]->getStdDev();
        }
        
        // Update robust weights
        std::vector<Real> newWeights(nMeas);
        computeRobustWeights(residuals, stdDevs, newWeights);
        
        // Check convergence (weights stabilized)
        if (irlsIter > 0) {
            Real weightChange = 0.0;
            for (size_t i = 0; i < newWeights.size() && i < result.weights.size(); ++i) {
                weightChange += std::abs(newWeights[i] - result.weights[i]);
            }
            
            if (weightChange < config_.tolerance * nMeas) {
                result.converged = true;
                result.iterations = irlsIter + 1;
                result.message = "Converged";
                break;
            }
        }
        
        weights = newWeights;
        result.weights = weights;
        result.finalNorm = solverResult.finalNorm;
        result.objectiveValue = solverResult.objectiveValue;
    }
    
    if (!result.converged) {
        result.iterations = config_.maxIterations;
        result.message = "Maximum IRLS iterations reached";
    }
    
    result.state = std::make_unique<StateVector>(state);
    
    return result;
}

void RobustEstimator::computeRobustWeights(const std::vector<Real>& residuals,
                                          const std::vector<Real>& stdDevs,
                                          std::vector<Real>& weights) const {
    auto weightFunc = getWeightFunction();
    
    for (size_t i = 0; i < residuals.size() && i < stdDevs.size(); ++i) {
        Real normalizedResidual = std::abs(residuals[i]) / stdDevs[i];
        weights[i] = weightFunc(normalizedResidual);
    }
}

Real RobustEstimator::huberWeight(Real normalizedResidual) const {
    Real c = config_.tuningConstant;
    if (std::abs(normalizedResidual) <= c) {
        return 1.0;
    } else {
        return c / std::abs(normalizedResidual);
    }
}

Real RobustEstimator::bisquareWeight(Real normalizedResidual) const {
    Real c = config_.tuningConstant;
    Real u = normalizedResidual / c;
    
    if (std::abs(u) >= 1.0) {
        return 0.0;
    } else {
        Real t = 1.0 - u * u;
        return t * t;
    }
}

Real RobustEstimator::cauchyWeight(Real normalizedResidual) const {
    Real c = config_.tuningConstant;
    Real u = normalizedResidual / c;
    return 1.0 / (1.0 + u * u);
}

Real RobustEstimator::welschWeight(Real normalizedResidual) const {
    Real c = config_.tuningConstant;
    Real u = normalizedResidual / c;
    return std::exp(-u * u);
}

std::function<Real(Real)> RobustEstimator::getWeightFunction() const {
    switch (config_.weightFunction) {
        case RobustWeightFunction::HUBER:
            return [this](Real r) { return this->huberWeight(r); };
        case RobustWeightFunction::BISQUARE:
            return [this](Real r) { return this->bisquareWeight(r); };
        case RobustWeightFunction::CAUCHY:
            return [this](Real r) { return this->cauchyWeight(r); };
        case RobustWeightFunction::WELSCH:
            return [this](Real r) { return this->welschWeight(r); };
        default:
            return [this](Real r) { return this->huberWeight(r); };
    }
}

} // namespace math
} // namespace sle

