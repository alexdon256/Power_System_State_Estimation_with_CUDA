/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_INTERFACE_RESULTS_H
#define SLE_INTERFACE_RESULTS_H

#include <sle/Export.h>
#include <sle/model/StateVector.h>
#include <sle/Types.h>
#include <string>
#include <vector>
#include <memory>

namespace sle {
namespace interface {

struct StateEstimationResult;

class SLE_API Results {
public:
    Results();
    explicit Results(const StateEstimationResult& result);
    
    // Get state vector
    const model::StateVector& getState() const;
    
    // Get voltages and angles
    std::vector<Real> getVoltages() const;
    std::vector<Real> getAngles() const;
    
    // Get convergence info
    bool converged() const { return converged_; }
    int getIterations() const { return iterations_; }
    double getFinalNorm() const { return finalNorm_; }
    double getObjectiveValue() const { return objectiveValue_; }
    const std::string& getMessage() const { return message_; }
    
    // Export to different formats
    std::string toJSON() const;
    std::string toCSV() const;
    std::string toString() const;
    
private:
    std::unique_ptr<model::StateVector> state_;
    bool converged_;
    int iterations_;
    double finalNorm_;
    double objectiveValue_;
    std::string message_;
    int64_t timestamp_;
};

} // namespace interface
} // namespace sle

#endif // SLE_INTERFACE_RESULTS_H

