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
#include <sle/cuda/CudaDataManager.h>
#include <memory>
#include <vector>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
using cudaError_t = int;
#define cudaSuccess 0
inline cudaError_t cudaMalloc(void**, size_t) { return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return cudaSuccess; }
inline cudaError_t cudaMemset(void*, int, size_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
#define cudaMemcpyHostToDevice 0
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA disabled"; }
#endif

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
    bool verbose = false;
    // Note: All operations are CUDA-exclusive (GPU required)
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
    
    // Store computed values from GPU into Bus/Branch objects (optimized - reuses GPU data)
    // This method copies power injections/flows already computed during solver iterations
    // instead of recomputing them, eliminating redundant GPU computations
    void storeComputedValues(model::StateVector& state, model::NetworkModel& network);
    
private:
    SolverConfig config_;
    std::unique_ptr<JacobianMatrix> jacobian_;
    std::unique_ptr<MeasurementFunctions> measFuncs_;
    
    // Shared CudaDataManager for both MeasurementFunctions and JacobianMatrix
    std::unique_ptr<cuda::CudaDataManager> sharedDataManager_;
    
    // CUDA streams for overlapping operations (memory transfers + kernel execution)
    cudaStream_t computeStream_;      // Main compute stream
    cudaStream_t transferStream_;    // Dedicated stream for memory transfers
    bool streamInitialized_;
    
    // Pinned memory buffers for frequently transferred data (faster transfers)
    Real* h_pinned_z_ = nullptr;           // Pinned: measurement vector
    Real* h_pinned_weights_ = nullptr;      // Pinned: weight vector
    Real* h_pinned_state_ = nullptr;       // Pinned: state vector
    size_t pinned_z_size_ = 0;
    size_t pinned_weights_size_ = 0;
    size_t pinned_state_size_ = 0;
    
    // CUDA memory pool for performance (reuse allocations across iterations)
    struct CudaMemoryPool {
        Real* d_residual = nullptr;
        Real* d_weights = nullptr;
        Real* d_weightedResidual = nullptr;
        Real* d_rhs = nullptr;
        Real* d_z = nullptr;              // Measurement vector
        Real* d_hx = nullptr;              // Measurement function values
        Real* d_deltaX = nullptr;          // State correction
        Real* d_x_old = nullptr;           // Previous state
        Real* d_x_new = nullptr;           // Updated state
        Real* d_partial = nullptr;         // Partial reduction buffer (for norms, weighted sums)
        Real* d_pMW = nullptr;             // Pooled: MW values (computed on GPU)
        Real* d_qMVAR = nullptr;           // Pooled: MVAR values (computed on GPU)
        Real* d_iPU = nullptr;            // Pooled: Current in p.u. (computed on GPU)
        Real* d_iAmps = nullptr;          // Pooled: Current in Amperes (computed on GPU)
        size_t residualSize = 0;
        size_t weightsSize = 0;
        size_t weightedResidualSize = 0;
        size_t rhsSize = 0;
        size_t zSize = 0;
        size_t hxSize = 0;
        size_t deltaXSize = 0;
        size_t stateSize = 0;
        size_t partialSize = 0;            // Max grid size for partial reductions
        size_t derivedQuantitiesSize = 0;  // Size of pooled derived quantity buffers
        
        ~CudaMemoryPool() {
            if (d_residual) cudaFree(d_residual);
            if (d_weights) cudaFree(d_weights);
            if (d_weightedResidual) cudaFree(d_weightedResidual);
            if (d_rhs) cudaFree(d_rhs);
            if (d_z) cudaFree(d_z);
            if (d_hx) cudaFree(d_hx);
            if (d_deltaX) cudaFree(d_deltaX);
            if (d_x_old) cudaFree(d_x_old);
            if (d_x_new) cudaFree(d_x_new);
            if (d_partial) cudaFree(d_partial);
            if (d_WH_values) cudaFree(d_WH_values);
            if (d_pMW) cudaFree(d_pMW);
            if (d_qMVAR) cudaFree(d_qMVAR);
            if (d_iPU) cudaFree(d_iPU);
            if (d_iAmps) cudaFree(d_iAmps);
        }
        
        void ensureCapacity(size_t nMeas, size_t nStates, size_t nBranches = 0) {
            // Add error checking for cudaMalloc failures
            cudaError_t err;
            
            // Ensure partial reduction buffer (for max grid size)
            // This buffer is reused across iterations for norm computations
            constexpr size_t blockSize = 256;
            size_t maxGridSize = (std::max(nMeas, nStates) + blockSize - 1) / blockSize;
            if (partialSize < maxGridSize) {
                if (d_partial) cudaFree(d_partial);
                err = cudaMalloc(reinterpret_cast<void**>(&d_partial), maxGridSize * sizeof(Real));
                if (err != cudaSuccess) {
                    d_partial = nullptr;
                    partialSize = 0;
                } else {
                    partialSize = maxGridSize;
                }
            }
            if (residualSize < nMeas) {
                if (d_residual) cudaFree(d_residual);
                err = cudaMalloc(reinterpret_cast<void**>(&d_residual), nMeas * sizeof(Real));
                if (err != cudaSuccess) {
                    d_residual = nullptr;
                    residualSize = 0;
                    return;
                }
                residualSize = nMeas;
            }
            if (weightsSize < nMeas) {
                if (d_weights) cudaFree(d_weights);
                err = cudaMalloc(reinterpret_cast<void**>(&d_weights), nMeas * sizeof(Real));
                if (err != cudaSuccess) {
                    d_weights = nullptr;
                    weightsSize = 0;
                    return;
                }
                weightsSize = nMeas;
            }
            if (weightedResidualSize < nMeas) {
                if (d_weightedResidual) cudaFree(d_weightedResidual);
                err = cudaMalloc(reinterpret_cast<void**>(&d_weightedResidual), nMeas * sizeof(Real));
                if (err != cudaSuccess) {
                    d_weightedResidual = nullptr;
                    weightedResidualSize = 0;
                    return;
                }
                weightedResidualSize = nMeas;
            }
            if (rhsSize < nStates) {
                if (d_rhs) cudaFree(d_rhs);
                err = cudaMalloc(reinterpret_cast<void**>(&d_rhs), nStates * sizeof(Real));
                if (err != cudaSuccess) {
                    d_rhs = nullptr;
                    rhsSize = 0;
                    return;
                }
                rhsSize = nStates;
            }
            if (zSize < nMeas) {
                if (d_z) cudaFree(d_z);
                err = cudaMalloc(reinterpret_cast<void**>(&d_z), nMeas * sizeof(Real));
                if (err != cudaSuccess) {
                    d_z = nullptr;
                    zSize = 0;
                } else {
                    zSize = nMeas;
                }
            }
            if (hxSize < nMeas) {
                if (d_hx) cudaFree(d_hx);
                err = cudaMalloc(reinterpret_cast<void**>(&d_hx), nMeas * sizeof(Real));
                if (err != cudaSuccess) {
                    d_hx = nullptr;
                    hxSize = 0;
                } else {
                    hxSize = nMeas;
                }
            }
            if (deltaXSize < nStates) {
                if (d_deltaX) cudaFree(d_deltaX);
                err = cudaMalloc(reinterpret_cast<void**>(&d_deltaX), nStates * sizeof(Real));
                if (err != cudaSuccess) {
                    d_deltaX = nullptr;
                    deltaXSize = 0;
                } else {
                    deltaXSize = nStates;
                }
            }
            if (stateSize < nStates) {
                if (d_x_old) cudaFree(d_x_old);
                if (d_x_new) cudaFree(d_x_new);
                err = cudaMalloc(reinterpret_cast<void**>(&d_x_old), nStates * sizeof(Real));
                if (err == cudaSuccess) {
                    err = cudaMalloc(reinterpret_cast<void**>(&d_x_new), nStates * sizeof(Real));
                    if (err != cudaSuccess) {
                        cudaFree(d_x_old);
                        d_x_old = nullptr;
                        d_x_new = nullptr;
                        stateSize = 0;
                    } else {
                        stateSize = nStates;
                    }
                } else {
                    d_x_old = nullptr;
                    d_x_new = nullptr;
                    stateSize = 0;
                }
            }
            // Ensure WH_values buffer for gain matrix computation (pooled to avoid per-iteration allocation)
            // This buffer is used in computeGainMatrixGPU for W*H computation
            if (nMeas > 0) {
                // Estimate max nnz (worst case: all measurements have all states)
                // Conservative estimate: assume each measurement has at most nStates non-zeros
                size_t maxH_nnz = nMeas * nStates;  // Conservative upper bound
                if (WH_valuesSize < maxH_nnz) {
                    if (d_WH_values) cudaFree(d_WH_values);
                    err = cudaMalloc(reinterpret_cast<void**>(&d_WH_values), maxH_nnz * sizeof(Real));
                    if (err != cudaSuccess) {
                        d_WH_values = nullptr;
                        WH_valuesSize = 0;
                    } else {
                        WH_valuesSize = maxH_nnz;
                    }
                }
            }
            
            // Ensure derived quantity buffers (for power flow MW/MVAR/I_PU/I_Amps)
            if (nBranches > 0 && derivedQuantitiesSize < nBranches) {
                if (d_pMW) cudaFree(d_pMW);
                if (d_qMVAR) cudaFree(d_qMVAR);
                if (d_iPU) cudaFree(d_iPU);
                if (d_iAmps) cudaFree(d_iAmps);
                
                err = cudaMalloc(reinterpret_cast<void**>(&d_pMW), nBranches * sizeof(Real));
                if (err == cudaSuccess) {
                    err = cudaMalloc(reinterpret_cast<void**>(&d_qMVAR), nBranches * sizeof(Real));
                    if (err == cudaSuccess) {
                        err = cudaMalloc(reinterpret_cast<void**>(&d_iPU), nBranches * sizeof(Real));
                        if (err == cudaSuccess) {
                            err = cudaMalloc(reinterpret_cast<void**>(&d_iAmps), nBranches * sizeof(Real));
                            if (err == cudaSuccess) {
                                derivedQuantitiesSize = nBranches;
                            } else {
                                // Free previously allocated buffers on failure
                                cudaFree(d_pMW);
                                cudaFree(d_qMVAR);
                                cudaFree(d_iPU);
                                d_pMW = nullptr;
                                d_qMVAR = nullptr;
                                d_iPU = nullptr;
                                d_iAmps = nullptr;
                                derivedQuantitiesSize = 0;
                            }
                        } else {
                            cudaFree(d_pMW);
                            cudaFree(d_qMVAR);
                            d_pMW = nullptr;
                            d_qMVAR = nullptr;
                            d_iPU = nullptr;
                            d_iAmps = nullptr;
                            derivedQuantitiesSize = 0;
                        }
                    } else {
                        cudaFree(d_pMW);
                        d_pMW = nullptr;
                        d_qMVAR = nullptr;
                        d_iPU = nullptr;
                        d_iAmps = nullptr;
                        derivedQuantitiesSize = 0;
                    }
                } else {
                    d_pMW = nullptr;
                    d_qMVAR = nullptr;
                    d_iPU = nullptr;
                    d_iAmps = nullptr;
                    derivedQuantitiesSize = 0;
                }
            }
        }
    };
    mutable CudaMemoryPool memoryPool_;
    
    // CUDA-EXCLUSIVE: Compute gain matrix G = H^T R^-1 H on GPU using cuSPARSE
    void computeGainMatrixGPU(const JacobianMatrix& H, const Real* d_weights,
                             SparseMatrix& G, cusparseHandle_t cusparseHandle);
    
    // CUDA-EXCLUSIVE: Solve linear system G * Δx = rhs entirely on GPU
    void solveLinearSystemGPU(const SparseMatrix& G, Real* d_rhs, Real* d_deltaX, Index nStates);
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_SOLVER_H

