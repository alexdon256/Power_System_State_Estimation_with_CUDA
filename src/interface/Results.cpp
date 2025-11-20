/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/interface/Results.h>
#include <sle/interface/StateEstimator.h>
#include <sle/model/StateVector.h>
#include <sstream>
#include <iomanip>

namespace sle {
namespace interface {

Results::Results() 
    : converged_(false), iterations_(0), finalNorm_(0.0), 
      objectiveValue_(0.0), timestamp_(0) {
}

Results::Results(const StateEstimationResult& result)
    : state_(result.state ? std::make_unique<model::StateVector>(*result.state) : nullptr),
      converged_(result.converged), iterations_(result.iterations),
      finalNorm_(result.finalNorm), objectiveValue_(result.objectiveValue),
      message_(result.message), timestamp_(result.timestamp) {
}

const model::StateVector& Results::getState() const {
    if (!state_) {
        throw std::runtime_error("State vector not available");
    }
    return *state_;
}

std::vector<Real> Results::getVoltages() const {
    if (!state_) {
        return {};
    }
    return state_->getMagnitudes();
}

std::vector<Real> Results::getAngles() const {
    if (!state_) {
        return {};
    }
    return state_->getAngles();
}

std::string Results::toJSON() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{\n";
    oss << "  \"converged\": " << (converged_ ? "true" : "false") << ",\n";
    oss << "  \"iterations\": " << iterations_ << ",\n";
    oss << "  \"finalNorm\": " << finalNorm_ << ",\n";
    oss << "  \"objectiveValue\": " << objectiveValue_ << ",\n";
    oss << "  \"message\": \"" << message_ << "\",\n";
    oss << "  \"timestamp\": " << timestamp_ << ",\n";
    
    if (state_) {
        auto voltages = getVoltages();
        auto angles = getAngles();
        oss << "  \"voltages\": [";
        for (size_t i = 0; i < voltages.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << voltages[i];
        }
        oss << "],\n";
        oss << "  \"angles\": [";
        for (size_t i = 0; i < angles.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << angles[i];
        }
        oss << "]\n";
    }
    oss << "}";
    return oss.str();
}

std::string Results::toCSV() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Bus,Voltage,Angle\n";
    
    if (state_) {
        auto voltages = getVoltages();
        auto angles = getAngles();
        for (size_t i = 0; i < voltages.size(); ++i) {
            oss << i << "," << voltages[i] << "," << angles[i] << "\n";
        }
    }
    return oss.str();
}

std::string Results::toString() const {
    std::ostringstream oss;
    oss << "State Estimation Result:\n";
    oss << "  Converged: " << (converged_ ? "Yes" : "No") << "\n";
    oss << "  Iterations: " << iterations_ << "\n";
    oss << "  Final Norm: " << finalNorm_ << "\n";
    oss << "  Objective Value: " << objectiveValue_ << "\n";
    oss << "  Message: " << message_ << "\n";
    return oss.str();
}

} // namespace interface
} // namespace sle

