/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_JACOBIANMATRIX_H
#define SLE_MATH_JACOBIANMATRIX_H

#include <sle/Types.h>
#include <sle/math/SparseMatrix.h>
#include <vector>

// Forward declarations
namespace sle {
namespace model {
    class StateVector;
    class NetworkModel;
    class TelemetryData;
}
}

namespace sle {
namespace math {

class JacobianMatrix {
public:
    JacobianMatrix();
    ~JacobianMatrix();
    
    // Build Jacobian matrix H = ∂h/∂x
    void build(const model::StateVector& state, const model::NetworkModel& network,
               const model::TelemetryData& telemetry);
    
    // Get sparse matrix representation
    const SparseMatrix& getMatrix() const { return matrix_; }
    SparseMatrix& getMatrix() { return matrix_; }
    
    // Get dimensions
    Index getNRows() const { return nRows_; }
    Index getNCols() const { return nCols_; }
    
    // Build on GPU
    void buildGPU(const model::StateVector& state, const model::NetworkModel& network,
                  const model::TelemetryData& telemetry);
    
private:
    SparseMatrix matrix_;
    Index nRows_;
    Index nCols_;
    
    void buildStructure(const model::NetworkModel& network, const model::TelemetryData& telemetry);
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_JACOBIANMATRIX_H

