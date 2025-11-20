/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_STATEVECTOR_H
#define SLE_MODEL_STATEVECTOR_H

#include <sle/Types.h>
#include <vector>
#include <memory>

namespace sle {
namespace model {

class NetworkModel;

class StateVector {
public:
    StateVector();
    explicit StateVector(size_t nBuses);
    
    void resize(size_t nBuses);
    size_t size() const { return nBuses_; }
    
    // Access voltage magnitudes and angles
    Real getVoltageMagnitude(Index busIdx) const;
    Real getVoltageAngle(Index busIdx) const;
    void setVoltageMagnitude(Index busIdx, Real v);
    void setVoltageAngle(Index busIdx, Real angle);
    
    // Get state vector [θ₁, ..., θₙ, V₁, ..., Vₙ]ᵀ
    const std::vector<Real>& getStateVector() const { return state_; }
    std::vector<Real>& getStateVector() { return state_; }
    
    // Get angles and magnitudes separately
    const std::vector<Real>& getAngles() const { return angles_; }
    const std::vector<Real>& getMagnitudes() const { return magnitudes_; }
    
    // Update from state vector
    void updateFromStateVector();
    
    // Initialize from network model
    void initializeFromNetwork(const NetworkModel& network);
    
    // State vector operations
    void add(const StateVector& other);
    void scale(Real factor);
    Real norm() const;
    Real normSquared() const;
    
private:
    size_t nBuses_;
    std::vector<Real> state_;      // [θ₁, ..., θₙ, V₁, ..., Vₙ]ᵀ
    std::vector<Real> angles_;      // [θ₁, ..., θₙ]
    std::vector<Real> magnitudes_;  // [V₁, ..., Vₙ]
    
    void updateStateVector();
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_STATEVECTOR_H

