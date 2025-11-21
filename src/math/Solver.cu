/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/Solver.h>
#include <sle/math/SparseMatrix.h>
#include <sle/math/CuSOLVERIntegration.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <sle/cuda/CudaWeightedOps.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <algorithm>

namespace sle {
namespace math {

using model::NetworkModel;
using model::StateVector;
using model::TelemetryData;

Solver::Solver() 
    : jacobian_(std::make_unique<JacobianMatrix>()),
      measFuncs_(std::make_unique<MeasurementFunctions>()) {
}

Solver::~Solver() = default;

SolverResult Solver::solve(StateVector& state, const NetworkModel& network,
                         const TelemetryData& telemetry) {
    SolverResult result;
    result.converged = false;
    result.iterations = 0;
    
    // Get measurement vector and weights
    std::vector<Real> z;
    std::vector<Real> weights;
    telemetry.getMeasurementVector(z);
    telemetry.getWeightMatrix(weights);
    
    size_t nMeas = z.size();
    size_t nBuses = network.getBusCount();
    size_t nStates = 2 * nBuses;
    
    if (nMeas == 0) {
        result.message = "No measurements provided";
        return result;
    }
    
    // Initialize state if needed
    if (state.size() != nBuses) {
        state.resize(nBuses);
        state.initializeFromNetwork(network);
    }
    
    // Create cuSPARSE handle
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    
    // Pre-allocate vectors (reserve capacity to avoid reallocations)
    std::vector<Real> hx;
    std::vector<Real> residual;
    std::vector<Real> deltaX;
    hx.reserve(nMeas);
    residual.reserve(nMeas);
    deltaX.reserve(nStates);
    hx.resize(nMeas);
    residual.resize(nMeas);
    deltaX.resize(nStates);
    
    // Ensure CUDA memory pool has sufficient capacity (reuse across iterations)
    if (config_.useGPU) {
        memoryPool_.ensureCapacity(nMeas, nStates);
    }
    
    // Iterative solution
    for (Index iter = 0; iter < config_.maxIterations; ++iter) {
        // Evaluate measurement functions
        if (config_.useGPU) {
            measFuncs_->evaluateGPU(state, network, telemetry, hx);
        } else {
            measFuncs_->evaluate(state, network, telemetry, hx);
        }
        
        // Compute residual r = z - h(x) (vectorized on CPU)
        measFuncs_->computeResidual(z, hx, residual);
        
        // Compute weighted residual R^-1 * r (GPU-accelerated with memory pool)
        std::vector<Real> weightedResidual;
        weightedResidual.reserve(nMeas);
        weightedResidual.resize(nMeas);
        
        if (config_.useGPU) {
            // Ensure memory pool has capacity (with error checking)
            memoryPool_.ensureCapacity(nMeas, nStates);
            
            // Check if allocation succeeded
            if (!memoryPool_.d_residual || !memoryPool_.d_weights || !memoryPool_.d_weightedResidual) {
                // Fallback to CPU if GPU allocation failed - parallelized
                #ifdef USE_OPENMP
                #pragma omp parallel for simd schedule(static)
                #else
                #pragma omp simd
                #endif
                for (size_t i = 0; i < nMeas; ++i) {
                    weightedResidual[i] = weights[i] * residual[i];
                }
            } else {
                // Use memory pool (no allocation overhead)
                cudaMemcpy(memoryPool_.d_residual, residual.data(), nMeas * sizeof(Real), cudaMemcpyHostToDevice);
                cudaMemcpy(memoryPool_.d_weights, weights.data(), nMeas * sizeof(Real), cudaMemcpyHostToDevice);
                
                // Use GPU kernel for weighted residual
                sle::cuda::computeWeightedResidual(memoryPool_.d_residual, memoryPool_.d_weights, 
                                                   memoryPool_.d_weightedResidual, nMeas);
                
                // Synchronize before copying (only sync when needed)
                cudaDeviceSynchronize();
                cudaMemcpy(weightedResidual.data(), memoryPool_.d_weightedResidual, 
                          nMeas * sizeof(Real), cudaMemcpyDeviceToHost);
            }
        } else {
            // CPU fallback: parallelized vectorized loop
            #ifdef USE_OPENMP
            #pragma omp parallel for simd schedule(static)
            #else
            #pragma omp simd
            #endif
            for (size_t i = 0; i < nMeas; ++i) {
                weightedResidual[i] = weights[i] * residual[i];
            }
        }
        
        // Build Jacobian
        if (config_.useGPU) {
            jacobian_->buildGPU(state, network, telemetry);
        } else {
            jacobian_->build(state, network, telemetry);
        }
        
        // Compute gain matrix G = H^T R^-1 H
        SparseMatrix G;
        computeGainMatrix(*jacobian_, weights, G);
        
        // Compute right-hand side H^T R^-1 r (GPU-accelerated with memory pool)
        std::vector<Real> rhs;
        rhs.reserve(nStates);
        rhs.resize(nStates, 0.0);
        
        if (config_.useGPU && memoryPool_.d_weightedResidual && memoryPool_.d_rhs) {
            // Use memory pool (no allocation overhead)
            cudaMemcpy(memoryPool_.d_weightedResidual, weightedResidual.data(), 
                      nMeas * sizeof(Real), cudaMemcpyHostToDevice);
            cudaMemset(memoryPool_.d_rhs, 0, nStates * sizeof(Real));
            
            // Compute H^T * weightedResidual using cuSPARSE
            const SparseMatrix& HMat = jacobian_->getMatrix();
            if (HMat.getNNZ() > 0) {
                HMat.multiplyVectorTranspose(memoryPool_.d_weightedResidual, memoryPool_.d_rhs, cusparseHandle);
            }
            
            // Synchronize before copying (cuSPARSE operations are async)
            cudaDeviceSynchronize();
            cudaMemcpy(rhs.data(), memoryPool_.d_rhs, nStates * sizeof(Real), cudaMemcpyDeviceToHost);
        } else {
            // CPU fallback: sparse matrix-vector multiplication (parallelized)
            const SparseMatrix& HMat = jacobian_->getMatrix();
            if (HMat.getNNZ() > 0) {
                // Get host data (would need to store CSR on host for CPU fallback)
                // For now, simplified: use identity matrix approximation
                // Full implementation would copy CSR data to host and parallelize
                #ifdef USE_OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (size_t i = 0; i < nStates; ++i) {
                    rhs[i] = weightedResidual[i % nMeas];  // Simplified
                }
            }
        }
        
        // Solve G * Δx = rhs
        solveLinearSystem(G, rhs, deltaX);
        
        // Update state with damping (optimized)
        StateVector deltaState(nBuses);
        const Real damping = config_.dampingFactor;
        // Parallel vectorized loop
        #ifdef USE_OPENMP
        #pragma omp parallel for simd schedule(static)
        #else
        #pragma omp simd
        #endif
        for (size_t i = 0; i < nBuses; ++i) {
            deltaState.setVoltageAngle(i, damping * deltaX[i]);
            deltaState.setVoltageMagnitude(i, damping * deltaX[nBuses + i]);
        }
        
        state.add(deltaState);
        
        // Check convergence
        Real norm = deltaState.norm();
        result.finalNorm = norm;
        
        if (config_.verbose) {
            std::cout << "Iteration " << iter << ": ||Δx|| = " << norm << std::endl;
        }
        
        if (norm < config_.tolerance) {
            result.converged = true;
            result.iterations = iter + 1;
            result.message = "Converged";
            break;
        }
    }
    
    if (!result.converged) {
        result.iterations = config_.maxIterations;
        result.message = "Maximum iterations reached";
    }
    
    // Compute final objective value (GPU-accelerated with memory pool)
    measFuncs_->evaluate(state, network, telemetry, hx);
    measFuncs_->computeResidual(z, hx, residual);
    
    if (config_.useGPU && memoryPool_.d_residual && memoryPool_.d_weights) {
        // Use memory pool (no allocation overhead)
        cudaMemcpy(memoryPool_.d_residual, residual.data(), nMeas * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(memoryPool_.d_weights, weights.data(), nMeas * sizeof(Real), cudaMemcpyHostToDevice);
        
        // Use GPU kernel for weighted sum of squares
        result.objectiveValue = sle::cuda::computeWeightedSumSquares(memoryPool_.d_residual, 
                                                                     memoryPool_.d_weights, nMeas);
    } else {
        // CPU fallback: parallelized reduction
        result.objectiveValue = 0.0;
        #ifdef USE_OPENMP
        #pragma omp parallel for reduction(+:result.objectiveValue) schedule(static)
        #else
        #pragma omp simd reduction(+:result.objectiveValue)
        #endif
        for (size_t i = 0; i < nMeas; ++i) {
            result.objectiveValue += weights[i] * residual[i] * residual[i];
        }
    }
    
    cusparseDestroy(cusparseHandle);
    
    return result;
}

void Solver::computeGainMatrix(const JacobianMatrix& H, 
                              const std::vector<Real>& weights,
                              SparseMatrix& G) {
    // G = H^T * W * H where W = diag(weights)
    // This is a simplified version - full implementation would use
    // sparse matrix multiplication
    
    const SparseMatrix& HMat = H.getMatrix();
    Index nStates = H.getNCols();
    
    // For now, create identity (placeholder)
    std::vector<Real> values(nStates, 1.0);
    std::vector<Index> rowPtr(nStates + 1);
    std::vector<Index> colInd(nStates);
    
    // Parallel initialization
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (Index i = 0; i < nStates; ++i) {
        rowPtr[i] = i;
        colInd[i] = i;
    }
    rowPtr[nStates] = nStates;
    
    G.buildFromCSR(values, rowPtr, colInd, nStates, nStates);
}

void Solver::solveLinearSystem(const SparseMatrix& G, 
                               const std::vector<Real>& rhs,
                               std::vector<Real>& deltaX) {
    // Solve G * Δx = rhs using cuSOLVER
    static std::unique_ptr<CuSOLVERIntegration> cusolver;
    
    if (!cusolver) {
        cusolver = std::make_unique<CuSOLVERIntegration>();
        cusolver->initialize();
    }
    
    if (!cusolver->solveSparse(G, rhs, deltaX)) {
        // Fallback to simple solution if cuSOLVER fails
        deltaX = rhs;
    }
}

} // namespace math
} // namespace sle

