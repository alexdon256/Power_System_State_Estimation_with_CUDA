/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_SOLVER_H
#define SLE_MATH_SOLVER_H

#include <sle/Types.h>
#include <sle/math/JacobianMatrix.h>
#include <sle/math/MeasurementFunctions.h>
#include <sle/math/SparseMatrix.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

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

struct SolverConfig {
    Real tolerance = 1e-6;
    Index maxIterations = 50;
    Real dampingFactor = 1.0;
    bool useGPU = true;
    bool verbose = false;
};

struct SolverResult {
    bool converged;
    Index iterations;
    Real finalNorm;
    Real objectiveValue;
    std::string message;
};

class Solver {
public:
    Solver();
    ~Solver();
    
    void setConfig(const SolverConfig& config) { config_ = config; }
    const SolverConfig& getConfig() const { return config_; }
    
    // Solve WLS state estimation: minimize J(x) = [z - h(x)]^T R^-1 [z - h(x)]
    SolverResult solve(model::StateVector& state, const model::NetworkModel& network,
                      const model::TelemetryData& telemetry);
    
private:
    SolverConfig config_;
    std::unique_ptr<JacobianMatrix> jacobian_;
    std::unique_ptr<MeasurementFunctions> measFuncs_;
    
    // CUDA memory pool for performance (reuse allocations across iterations)
    struct CudaMemoryPool {
        Real* d_residual = nullptr;
        Real* d_weights = nullptr;
        Real* d_weightedResidual = nullptr;
        Real* d_rhs = nullptr;
        size_t residualSize = 0;
        size_t weightsSize = 0;
        size_t weightedResidualSize = 0;
        size_t rhsSize = 0;
        
        ~CudaMemoryPool() {
            if (d_residual) cudaFree(d_residual);
            if (d_weights) cudaFree(d_weights);
            if (d_weightedResidual) cudaFree(d_weightedResidual);
            if (d_rhs) cudaFree(d_rhs);
        }
        
        void ensureCapacity(size_t nMeas, size_t nStates) {
            // Add error checking for cudaMalloc failures
            cudaError_t err;
            if (residualSize < nMeas) {
                if (d_residual) cudaFree(d_residual);
                err = cudaMalloc(&d_residual, nMeas * sizeof(Real));
                if (err != cudaSuccess) {
                    d_residual = nullptr;
                    residualSize = 0;
                    return;  // Allocation failed
                }
                residualSize = nMeas;
            }
            if (weightsSize < nMeas) {
                if (d_weights) cudaFree(d_weights);
                err = cudaMalloc(&d_weights, nMeas * sizeof(Real));
                if (err != cudaSuccess) {
                    d_weights = nullptr;
                    weightsSize = 0;
                    return;
                }
                weightsSize = nMeas;
            }
            if (weightedResidualSize < nMeas) {
                if (d_weightedResidual) cudaFree(d_weightedResidual);
                err = cudaMalloc(&d_weightedResidual, nMeas * sizeof(Real));
                if (err != cudaSuccess) {
                    d_weightedResidual = nullptr;
                    weightedResidualSize = 0;
                    return;
                }
                weightedResidualSize = nMeas;
            }
            if (rhsSize < nStates) {
                if (d_rhs) cudaFree(d_rhs);
                err = cudaMalloc(&d_rhs, nStates * sizeof(Real));
                if (err != cudaSuccess) {
                    d_rhs = nullptr;
                    rhsSize = 0;
                    return;
                }
                rhsSize = nStates;
            }
        }
    };
    mutable CudaMemoryPool memoryPool_;
    
    // Compute gain matrix G = H^T R^-1 H
    void computeGainMatrix(const JacobianMatrix& H, const std::vector<Real>& weights,
                          SparseMatrix& G);
    
    // Solve linear system G * Δx = H^T R^-1 r
    void solveLinearSystem(const SparseMatrix& G, const std::vector<Real>& rhs,
                          std::vector<Real>& deltaX);
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_SOLVER_H

