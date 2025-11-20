/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_ROBUSTESTIMATOR_H
#define SLE_MATH_ROBUSTESTIMATOR_H

#include <sle/Types.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <functional>

namespace sle {
namespace math {

// Robust M-estimator weight functions
enum class RobustWeightFunction {
    HUBER,      // Huber's function
    BISQUARE,   // Bi-square (Tukey's biweight)
    CAUCHY,     // Cauchy function
    WELSCH      // Welsch function
};

// Robust estimator configuration
struct RobustEstimatorConfig {
    RobustWeightFunction weightFunction = RobustWeightFunction::HUBER;
    Real tuningConstant = 1.345;  // Default for Huber
    Real tolerance = 1e-6;
    Index maxIterations = 50;
    bool useGPU = true;
};

class RobustEstimator {
public:
    RobustEstimator();
    explicit RobustEstimator(const RobustEstimatorConfig& config);
    
    // Set configuration
    void setConfig(const RobustEstimatorConfig& config) { config_ = config; }
    const RobustEstimatorConfig& getConfig() const { return config_; }
    
    // Robust state estimation using M-estimators
    struct RobustResult {
        bool converged;
        Index iterations;
        Real finalNorm;
        Real objectiveValue;
        std::vector<Real> weights;  // Final robust weights
        std::unique_ptr<StateVector> state;
        std::string message;
    };
    
    RobustResult estimate(StateVector& state, const NetworkModel& network,
                         const TelemetryData& telemetry);
    
    // Compute robust weights based on residuals
    void computeRobustWeights(const std::vector<Real>& residuals,
                             const std::vector<Real>& stdDevs,
                             std::vector<Real>& weights) const;
    
private:
    RobustEstimatorConfig config_;
    
    // Weight functions
    Real huberWeight(Real normalizedResidual) const;
    Real bisquareWeight(Real normalizedResidual) const;
    Real cauchyWeight(Real normalizedResidual) const;
    Real welschWeight(Real normalizedResidual) const;
    
    // Select weight function
    std::function<Real(Real)> getWeightFunction() const;
    
    // Iteratively reweighted least squares (IRLS)
    RobustResult solveIRLS(StateVector& state, const NetworkModel& network,
                          const TelemetryData& telemetry);
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_ROBUSTESTIMATOR_H

