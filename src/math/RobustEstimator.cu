/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/RobustEstimator.h>
#include <sle/math/Solver.h>
#include <sle/math/MeasurementFunctions.h>
#include <cmath>
#include <algorithm>

namespace sle {
namespace math {

using model::StateVector;
using model::NetworkModel;
using model::TelemetryData;
using model::MeasurementModel;

RobustEstimator::RobustEstimator() : solver_(std::make_unique<Solver>()) {
    config_.weightFunction = RobustWeightFunction::HUBER;
    config_.tuningConstant = 1.345;
}

RobustEstimator::RobustEstimator(const RobustEstimatorConfig& config)
    : config_(config), solver_(std::make_unique<Solver>()) {
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
    
    auto measurements = telemetry.getMeasurements();
    size_t nMeas = measurements.size();
    
    // OPTIMIZATION: Pre-compute base weights and stdDevs once (loop invariants)
    std::vector<Real> baseWeights(nMeas);
    std::vector<Real> stdDevs(nMeas);
    for (size_t i = 0; i < nMeas; ++i) {
        baseWeights[i] = measurements[i]->getWeight();
        stdDevs[i] = measurements[i]->getStdDev();
    }
    
    // Robust scaling factors (0.0 to 1.0)
    // Initialize to 1.0 (standard WLS)
    std::vector<Real> robustScaling(nMeas, 1.0);
    result.weights = robustScaling; // Initial weights in result
    
    MeasurementFunctions measFuncs;
    measFuncs.setDataManager(solver_->getDataManager());
    
    SolverConfig solverConfig;
    solverConfig.tolerance = config_.tolerance;
    solverConfig.maxIterations = 10;  // Fewer iterations per IRLS step
    solver_->setConfig(solverConfig);
    
    // IRLS iterations
    for (Index irlsIter = 0; irlsIter < config_.maxIterations; ++irlsIter) {
        // OPTIMIZATION: Calculate absolute weights for Solver using pre-computed base weights
        std::vector<Real> solverWeights(nMeas);
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < nMeas; ++i) {
            solverWeights[i] = baseWeights[i] * robustScaling[i];
        }
        
        // Solve weighted least squares with override weights
        // Reuse structure for subsequent iterations
        bool reuseStructure = (irlsIter > 0);
        SolverResult solverResult = solver_->solve(state, network, telemetry, solverWeights, reuseStructure);
        
        if (!solverResult.converged && irlsIter == 0) {
            // If first iteration (standard WLS) fails, likely bad data or observability issue
            result.message = "Initial WLS solution failed to converge";
            break;
        }
        
        // CUDA-EXCLUSIVE: Compute residuals using Solver's optimized method
        // This reuses the GPU buffers (d_z, d_hx) from the solver
        std::vector<Real> residuals(nMeas);
        solver_->getLastResiduals(residuals); // Efficiently copies r = z - h(x) from GPU
        
        // OPTIMIZATION: stdDevs already pre-computed outside loop
        
        // Update robust weights (scaling factors)
        std::vector<Real> newRobustScaling(nMeas);
        computeRobustWeights(residuals, stdDevs, newRobustScaling);
        
        // Check convergence (weights stabilized)
        if (irlsIter > 0) {
            Real weightChange = 0.0;
            // OPTIMIZATION: OpenMP for weight difference reduction
            #ifdef USE_OPENMP
            #pragma omp parallel for reduction(+:weightChange)
            #endif
            for (size_t i = 0; i < nMeas; ++i) {
                weightChange += std::abs(newRobustScaling[i] - robustScaling[i]);
            }
            
            if (weightChange < config_.tolerance * nMeas) {
                result.converged = true;
                result.iterations = irlsIter + 1;
                result.message = "Converged";
                robustScaling = newRobustScaling;
                break;
            }
        }
        
        robustScaling = newRobustScaling;
        result.weights = robustScaling;
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
    
    // OPTIMIZATION: OpenMP parallelization for independent weight calculations
    size_t n = residuals.size();
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < n; ++i) {
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
