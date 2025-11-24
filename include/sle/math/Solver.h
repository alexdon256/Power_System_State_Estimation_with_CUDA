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
#include <sle/cuda/CudaUtils.h>
#include <sle/math/CuSOLVERIntegration.h>
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
using cudaStream_t = void*;
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
    // reuseStructure: If true, assumes Jacobian structure matches previous call (static topology)
    SolverResult solve(model::StateVector& state, const model::NetworkModel& network,
                      const model::TelemetryData& telemetry, bool reuseStructure = false);
                      
    // Solve with custom weights (useful for Robust Estimator / IRLS)
    // If weights is empty, uses weights from telemetry
    // reuseStructure: If true, assumes Jacobian structure matches previous call (static topology)
    SolverResult solve(model::StateVector& state, const model::NetworkModel& network,
                      const model::TelemetryData& telemetry, const std::vector<Real>& weights,
                      bool reuseStructure = false);
    
    // Store computed values from GPU into Bus/Branch objects (optimized - reuses GPU data)
    // This method copies power injections/flows already computed during solver iterations
    // instead of recomputing them, eliminating redundant GPU computations
    void storeComputedValues(model::StateVector& state, model::NetworkModel& network);
    
    void setUseUnifiedPool(bool use) { /* Deprecated */ }
    bool getUseUnifiedPool() const { return true; }
    
    // Get data manager (useful for sharing with other components)
    cuda::CudaDataManager* getDataManager() const { return sharedDataManager_.get(); }

    // Get measurement functions (useful for sharing with BadDataDetector)
    MeasurementFunctions& getMeasurementFunctions() { return *measFuncs_; }
    const MeasurementFunctions& getMeasurementFunctions() const { return *measFuncs_; }
    
private:
    SolverConfig config_;
    std::unique_ptr<JacobianMatrix> jacobian_;
    std::unique_ptr<MeasurementFunctions> measFuncs_;
    
    // Shared CudaDataManager for both MeasurementFunctions and JacobianMatrix
    std::unique_ptr<cuda::CudaDataManager> sharedDataManager_;
    
    // CUDA streams for overlapping operations (memory transfers + kernel execution)
    cudaStream_t computeStream_;      // Main compute stream
    cudaStream_t transferStream_;    // Dedicated stream for memory transfers
    
    // Persistent handles and objects to avoid re-initialization
    cusparseHandle_t cusparseHandle_;
    std::unique_ptr<CuSOLVERIntegration> cusolver_;
    SparseMatrix gainMatrix_;         // Reused gain matrix (preserves allocation)
    
    // Pinned memory buffers for frequently transferred data (faster transfers)
    Real* h_pinned_z_ = nullptr;           // Pinned: measurement vector
    Real* h_pinned_weights_ = nullptr;      // Pinned: weight vector
    Real* h_pinned_state_ = nullptr;       // Pinned: state vector
    size_t pinned_z_size_ = 0;
    size_t pinned_weights_size_ = 0;
    size_t pinned_state_size_ = 0;
    
    // CudaMemoryPool stores large GPU buffers to avoid re-allocation overhead.
    // These buffers are only resized when necessary.
    struct CudaMemoryPool {
        Real* d_residual = nullptr;
        Real* d_weights = nullptr;
        Real* d_weightedResidual = nullptr;
        Real* d_rhs = nullptr;
        Real* d_z = nullptr;              // Measurement vector
        // Real* d_hx = nullptr;           // Removed: Managed by MeasurementFunctions/CudaDataManager
        Real* d_deltaX = nullptr;          // State correction
        Real* d_x_old = nullptr;           // Previous state
        Real* d_x_new = nullptr;           // Updated state
        Real* d_partial = nullptr;         // Partial reduction buffer (for norms, weighted sums)
        Real* d_WH_values = nullptr;        // Pooled: W*H values for gain matrix computation
        Real* d_pMW = nullptr;             // Pooled: MW values (computed on GPU) - may point to unified pool
        Real* d_qMVAR = nullptr;           // Pooled: MVAR values (computed on GPU) - may point to unified pool
        Real* d_iPU = nullptr;            // Pooled: Current in p.u. (computed on GPU) - may point to unified pool
        Real* d_iAmps = nullptr;          // Pooled: Current in Amperes (computed on GPU) - may point to unified pool
        size_t residualSize = 0;
        size_t weightsSize = 0;
        size_t weightedResidualSize = 0;
        size_t rhsSize = 0;
        size_t zSize = 0;
        // size_t hxSize = 0;              // Removed
        size_t deltaXSize = 0;
        size_t stateSize = 0;
        size_t partialSize = 0;            // Max grid size for partial reductions - may point to unified pool
        size_t WH_valuesSize = 0;          // Size of WH_values buffer
        size_t pMWSize = 0;                // Size tracking for unified pool
        size_t qMVARSize = 0;
        size_t iPUSize = 0;
        size_t iAmpsSize = 0;
        size_t derivedQuantitiesSize = 0;  // Size of pooled derived quantity buffers
        
        ~CudaMemoryPool() {
            if (d_residual) cudaFree(d_residual);
            if (d_weights) cudaFree(d_weights);
            if (d_weightedResidual) cudaFree(d_weightedResidual);
            if (d_rhs) cudaFree(d_rhs);
            if (d_z) cudaFree(d_z);
            // if (d_hx) cudaFree(d_hx);   // Removed
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
            // Ensure partial reduction buffer (for max grid size)
            // This buffer is reused across iterations for norm computations
            constexpr size_t blockSize = 256;
            size_t maxGridSize = KernelConfig<blockSize>::gridSize(std::max(nMeas, nStates));
            cuda::ensureCapacity(d_partial, partialSize, maxGridSize);
            
            // Ensure measurement buffers (critical - return on failure)
            if (!cuda::ensureCapacity(d_residual, residualSize, nMeas)) {
                return;
            }
            if (!cuda::ensureCapacity(d_weights, weightsSize, nMeas)) {
                return;
            }
            if (!cuda::ensureCapacity(d_weightedResidual, weightedResidualSize, nMeas)) {
                return;
            }
            
            // Ensure state buffers
            if (!cuda::ensureCapacity(d_rhs, rhsSize, nStates)) {
                return;
            }
            cuda::ensureCapacity(d_z, zSize, nMeas);
            
            cuda::ensureCapacity(d_deltaX, deltaXSize, nStates);
            
            // Ensure state vectors (allocate both together)
            if (stateSize < nStates) {
                cuda::freeBuffer(d_x_old, stateSize);
                cuda::freeBuffer(d_x_new, stateSize);
                if (cuda::allocateBuffer(d_x_old, stateSize, nStates) &&
                    cuda::allocateBuffer(d_x_new, stateSize, nStates)) {
                    stateSize = nStates;
                } else {
                    cuda::freeBuffer(d_x_old, stateSize);
                    cuda::freeBuffer(d_x_new, stateSize);
                    stateSize = 0;
                }
            }
            
            // Ensure WH_values buffer for gain matrix computation (pooled to avoid per-iteration allocation)
            if (nMeas > 0) {
                size_t maxH_nnz = nMeas * nStates;  // Conservative upper bound
                cuda::ensureCapacity(d_WH_values, WH_valuesSize, maxH_nnz);
            }
            
            // Ensure derived quantity buffers (for power flow MW/MVAR/I_PU/I_Amps)
            if (nBranches > 0 && derivedQuantitiesSize < nBranches) {
                // Free all if any allocation fails
                cuda::freeBuffer(d_pMW, derivedQuantitiesSize);
                cuda::freeBuffer(d_qMVAR, derivedQuantitiesSize);
                cuda::freeBuffer(d_iPU, derivedQuantitiesSize);
                cuda::freeBuffer(d_iAmps, derivedQuantitiesSize);
                
                if (cuda::allocateBuffer(d_pMW, derivedQuantitiesSize, nBranches) &&
                    cuda::allocateBuffer(d_qMVAR, derivedQuantitiesSize, nBranches) &&
                    cuda::allocateBuffer(d_iPU, derivedQuantitiesSize, nBranches) &&
                    cuda::allocateBuffer(d_iAmps, derivedQuantitiesSize, nBranches)) {
                    derivedQuantitiesSize = nBranches;
                } else {
                    // Cleanup on failure
                    cuda::freeBuffer(d_pMW, derivedQuantitiesSize);
                    cuda::freeBuffer(d_qMVAR, derivedQuantitiesSize);
                    cuda::freeBuffer(d_iPU, derivedQuantitiesSize);
                    cuda::freeBuffer(d_iAmps, derivedQuantitiesSize);
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
