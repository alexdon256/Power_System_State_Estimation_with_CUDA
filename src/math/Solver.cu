/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/Solver.h>
#include <sle/math/SparseMatrix.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <sle/cuda/CudaSolverOps.h>
#include <sle/cuda/CudaSparseOps.h>
#include <sle/cuda/CudaDataManager.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <sle/math/JacobianMatrix.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace sle {
namespace math {

using model::NetworkModel;
using model::StateVector;
using model::TelemetryData;
using model::Bus;
using model::Branch;
using cuda::DeviceBus;
using cuda::DeviceBranch;

Solver::Solver() 
    : jacobian_(std::make_unique<JacobianMatrix>()),
      measFuncs_(std::make_unique<MeasurementFunctions>()),
      sharedDataManager_(std::make_unique<cuda::CudaDataManager>()),
      cusolverHandle_(nullptr), cusparseDescr_(nullptr) {
    // Share CudaDataManager with MeasurementFunctions
    measFuncs_->setDataManager(sharedDataManager_.get());
    
    // Create CUDA streams for overlapping operations (required)
    CUDA_CHECK_THROW(cudaStreamCreate(&computeStream_));
    CUDA_CHECK_THROW(cudaStreamCreate(&transferStream_));
    
    // Initialize persistent handles (merged from CuSOLVERIntegration)
    cusparseCreate(&cusparseHandle_);
    cusparseSetStream(cusparseHandle_, computeStream_);
    
    cusolverSpCreate(&cusolverHandle_);
    cusolverSpSetStream(cusolverHandle_, computeStream_);
    
    cusparseCreateMatDescr(&cusparseDescr_);
    cusparseSetMatType(cusparseDescr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(cusparseDescr_, CUSPARSE_INDEX_BASE_ZERO);
}

Solver::~Solver() {
    // Free pinned memory buffers
    if (h_pinned_z_) cudaFreeHost(h_pinned_z_);
    if (h_pinned_weights_) cudaFreeHost(h_pinned_weights_);
    if (h_pinned_state_) cudaFreeHost(h_pinned_state_);
    
    // Destroy handles (merged from CuSOLVERIntegration)
    if (cusparseHandle_) cusparseDestroy(cusparseHandle_);
    if (cusolverHandle_) cusolverSpDestroy(cusolverHandle_);
    if (cusparseDescr_) cusparseDestroyMatDescr(cusparseDescr_);
    
    // Destroy CUDA streams
        if (computeStream_) cudaStreamDestroy(computeStream_);
        if (transferStream_) cudaStreamDestroy(transferStream_);
    }

SolverResult Solver::solve(StateVector& state, const NetworkModel& network,
                         const TelemetryData& telemetry, bool reuseStructure) {
    return solve(state, network, telemetry, {}, reuseStructure);
}

SolverResult Solver::solve(StateVector& state, const NetworkModel& network,
                         const TelemetryData& telemetry, const std::vector<Real>& weightsOverride,
                         bool reuseStructure) {
    SolverResult result;
    result.converged = false;
    result.iterations = 0;
    
    // Get measurement vector and weights
    std::vector<Real> z;
    std::vector<Real> weights;
    telemetry.getMeasurementVector(z);
    
    if (!weightsOverride.empty()) {
        if (weightsOverride.size() != z.size()) {
            result.message = "Weights override size mismatch";
            return result;
        }
        weights = weightsOverride;
    } else {
    telemetry.getWeightMatrix(weights);
    }
    
    size_t nMeas = z.size();
    size_t nBuses = network.getBusCount();
    size_t nStates = 2 * nBuses;
    
    lastNMeas_ = static_cast<Index>(nMeas);
    
    if (nMeas == 0) {
        result.message = "No measurements provided";
        return result;
    }
    
    // Initialize state if needed
    if (state.size() != nBuses) {
        state.resize(nBuses);
        state.initializeFromNetwork(network);
    }
    
    // CUDA-EXCLUSIVE: Ensure GPU memory pool has sufficient capacity
    size_t nBranches = network.getBranchCount();
    memoryPool_.ensureCapacity(nMeas, nStates, nBranches);
    
    // Verify all allocations succeeded (CUDA-exclusive, no fallback)
    if (!memoryPool_.d_residual || !memoryPool_.d_weights || 
        !memoryPool_.d_weightedResidual || !memoryPool_.d_rhs ||
        !memoryPool_.d_z || !memoryPool_.d_deltaX ||
        !memoryPool_.d_x_old || !memoryPool_.d_x_new || !memoryPool_.d_partial) {
        result.message = "CUDA memory allocation failed";
        return result;
    }
    // Note: Derived quantity buffers (d_pMW, etc.) are optional and allocated on-demand
    
    // OPTIMIZATION: Use pinned memory and async transfers with dedicated streams
    // Allocate/ensure pinned memory buffers for frequently transferred data
    cuda::allocatePinnedBuffer(h_pinned_z_, pinned_z_size_, nMeas);
    cuda::allocatePinnedBuffer(h_pinned_weights_, pinned_weights_size_, nMeas);
    cuda::allocatePinnedBuffer(h_pinned_state_, pinned_state_size_, nStates);
    
    // Copy to pinned memory (fast CPU-side copy)
    Real* z_src = h_pinned_z_ ? h_pinned_z_ : z.data();
    Real* weights_src = h_pinned_weights_ ? h_pinned_weights_ : weights.data();
    if (h_pinned_z_) std::copy(z.begin(), z.end(), h_pinned_z_);
    if (h_pinned_weights_) std::copy(weights.begin(), weights.end(), h_pinned_weights_);
    
    // Use dedicated transfer stream for memory transfers (can overlap with compute)
    cudaStream_t transferStream = transferStream_;
    
    // Launch async transfers on transfer stream (can overlap with compute operations)
    CUDA_CHECK_THROW(cudaMemcpyAsync(memoryPool_.d_z, z_src, nMeas * sizeof(Real), cudaMemcpyHostToDevice, transferStream_));
    CUDA_CHECK_THROW(cudaMemcpyAsync(memoryPool_.d_weights, weights_src, nMeas * sizeof(Real), cudaMemcpyHostToDevice, transferStream_));
    
    auto& stateVec = state.getStateVector();
    Real* state_src = h_pinned_state_ ? h_pinned_state_ : stateVec.data();
    if (h_pinned_state_) std::copy(stateVec.begin(), stateVec.end(), h_pinned_state_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(memoryPool_.d_x_old, state_src, nStates * sizeof(Real), cudaMemcpyHostToDevice, transferStream_));
    
    // Synchronize transfer stream before using data in compute stream
    cudaStreamSynchronize(transferStream_);
    
    // OPTIMIZATION: Build Jacobian structure only if needed
    if (reuseStructure) {
        // Safety check: Verify dimensions match even if reuse requested
        if (jacobian_->getNCols() != 2 * network.getBusCount() ||
            jacobian_->getNRows() != z.size()) {
            reuseStructure = false; // Force rebuild if dimensions mismatch
        }
    }
    
    if (!reuseStructure) {
    jacobian_->buildStructure(network, telemetry);
    }
    
    Real norm = 0.0;
    Real* d_hx_final = nullptr;  // Store d_hx from last iteration to reuse for objective
    
    // OPTIMIZATION: Get compute stream once for reuse
    cudaStream_t stream = computeStream_;
    
    for (Index iter = 0; iter < config_.maxIterations; ++iter) {
        // OPTIMIZATION: Pass stream to evaluateGPU for asynchronous execution
        // Pass reuseStructure so MeasurementFunctions knows whether to re-upload static data
        // Note: For iterations > 0, we always reuse topology as it doesn't change within solve loop
        bool currentReuse = reuseStructure || (iter > 0);
        Real* d_hx = measFuncs_->evaluateGPU(state, network, telemetry, currentReuse, stream);
        d_hx_final = d_hx;  // Keep reference to last computed hx
        
        // Fused: Compute residual and weighted residual in one kernel
        // OPTIMIZATION: Use compute stream for overlapping operations
        sle::cuda::computeResidualAndWeighted(memoryPool_.d_z, d_hx, memoryPool_.d_weights,
                                              memoryPool_.d_residual, memoryPool_.d_weightedResidual,
                                              static_cast<Index>(nMeas), stream);
        
        // Build Jacobian using shared CudaDataManager (reuses GPU data from MeasurementFunctions)
        // Structure already built, only values are recomputed
        // OPTIMIZATION: Pass stream for asynchronous execution
        jacobian_->buildGPU(state, network, telemetry, sharedDataManager_.get(), stream);
        
        // Compute gain matrix on GPU using cuSPARSE
        // Use persistent gainMatrix_ to avoid re-allocation
        computeGainMatrixGPU(*jacobian_, memoryPool_.d_weights, gainMatrix_, cusparseHandle_);
        
        // OPTIMIZATION: Use async memset with compute stream for better overlap
        cudaMemsetAsync(memoryPool_.d_rhs, 0, nStates * sizeof(Real), stream);
        
        const SparseMatrix& HMat = jacobian_->getMatrix();
        if (HMat.getNNZ() > 0) {
            // OPTIMIZATION: Use pooled SpMV buffer to avoid per-call allocation
            size_t spmvBufferSize = memoryPool_.spmvBufferSize;
            HMat.multiplyVectorTranspose(memoryPool_.d_weightedResidual, memoryPool_.d_rhs, cusparseHandle_,
                                        memoryPool_.d_spmvBuffer, &spmvBufferSize);
            // Update pool size if buffer was resized
            if (spmvBufferSize > memoryPool_.spmvBufferSize) {
                if (memoryPool_.d_spmvBuffer) {
                    cudaFree(memoryPool_.d_spmvBuffer);
        }
                cudaError_t err = cudaMalloc(&memoryPool_.d_spmvBuffer, spmvBufferSize);
                if (err == cudaSuccess) {
                    memoryPool_.spmvBufferSize = spmvBufferSize;
                } else {
                    memoryPool_.d_spmvBuffer = nullptr;
                    memoryPool_.spmvBufferSize = 0;
                }
            }
        }
        
        solveLinearSystemGPU(gainMatrix_, memoryPool_.d_rhs, memoryPool_.d_deltaX, nStates);
        
        // Fused: Update state and compute norm in one kernel (using pooled buffer)
        const Real damping = config_.dampingFactor;
        // OPTIMIZATION: Use compute stream for overlapping operations
        Real normSq = sle::cuda::updateStateAndComputeNorm(memoryPool_.d_x_old, memoryPool_.d_deltaX,
                                                           memoryPool_.d_x_new, damping,
                                                           static_cast<Index>(nStates),
                                                           memoryPool_.d_partial, memoryPool_.partialSize,
                                                           stream);
        norm = std::sqrt(normSq);
        
        std::swap(memoryPool_.d_x_old, memoryPool_.d_x_new);
        
        result.finalNorm = norm;
        
        if (config_.verbose) {
            std::cout << "Iteration " << iter << ": ||Δx|| = " << norm << std::endl;
        }
        
        if (norm < config_.tolerance) {
            result.converged = true;
            result.iterations = iter + 1;
            result.message = "Converged";
            // OPTIMIZATION: Compute objective value during last iteration (reuse d_hx)
            result.objectiveValue = sle::cuda::computeResidualAndObjective(memoryPool_.d_z, d_hx_final,
                                                                           memoryPool_.d_weights,
                                                                           memoryPool_.d_residual,
                                                                           static_cast<Index>(nMeas),
                                                                           memoryPool_.d_partial, memoryPool_.partialSize,
                                                                           computeStream_);
            break;
        }
        
        // Copy state to host only if needed for next iteration
        // evaluateGPU can work with GPU state, but updateFromStateVector needs host data
        // Only copy if not converged and not last iteration
        if (iter < config_.maxIterations - 1 && norm >= config_.tolerance) {
            // OPTIMIZATION: Use pinned memory and transfer stream for async copy
            Real* state_dst = h_pinned_state_ ? h_pinned_state_ : stateVec.data();
    cudaStream_t transferStream = transferStream_;
            cuda::asyncCopyD2HAndSync(memoryPool_.d_x_old, state_dst, nStates, transferStream_);
            cudaStreamSynchronize(transferStream);
            // Copy from pinned to stateVec if using pinned memory
            if (h_pinned_state_) {
                std::copy(h_pinned_state_, h_pinned_state_ + nStates, stateVec.begin());
            }
            state.updateFromStateVector();
        }
    }
    
    if (!result.converged) {
        result.iterations = config_.maxIterations;
        result.message = "Maximum iterations reached";
        // OPTIMIZATION: Compute objective value using last iteration's d_hx (reuse, don't recompute)
        if (d_hx_final) {
            // OPTIMIZATION: Use compute stream for overlapping operations
            result.objectiveValue = sle::cuda::computeResidualAndObjective(memoryPool_.d_z, d_hx_final,
                                                                           memoryPool_.d_weights,
                                                                           memoryPool_.d_residual,
                                                                           static_cast<Index>(nMeas),
                                                                           memoryPool_.d_partial, memoryPool_.partialSize,
                                                                           computeStream_);
        }
    }
    
    // OPTIMIZATION: Only copy state to host if needed (for storeComputedValues or user access)
    // No need to copy for final evaluation since we already computed objective from GPU data
    // Use pinned memory and transfer stream for async copy
    Real* state_dst = h_pinned_state_ ? h_pinned_state_ : stateVec.data();
    cudaStream_t transferStream = transferStream_;
    cuda::asyncCopyD2HAndSync(memoryPool_.d_x_old, state_dst, nStates, transferStream_);
    cudaStreamSynchronize(transferStream);
    // Copy from pinned to stateVec if using pinned memory
    if (h_pinned_state_) {
        std::copy(h_pinned_state_, h_pinned_state_ + nStates, stateVec.begin());
    }
    state.updateFromStateVector();
    
    return result;
}

void Solver::getLastResiduals(std::vector<Real>& residuals) const {
    if (!memoryPool_.d_residual || lastNMeas_ == 0) {
        residuals.clear();
        return;
    }
    
    residuals.resize(lastNMeas_);
    // Use transfer stream for consistency, though this is likely a blocking call
    cudaMemcpy(residuals.data(), memoryPool_.d_residual, lastNMeas_ * sizeof(Real), cudaMemcpyDeviceToHost);
}

void Solver::storeComputedValues(model::StateVector& state, model::NetworkModel& network) {
    // OPTIMIZED: Reuse power values already computed on GPU during solver iterations
    // instead of recomputing them, eliminating redundant GPU computations
    
    size_t nBuses = network.getBusCount();
    size_t nBranches = network.getBranchCount();
    
    if (nBuses == 0) return;
    
    // 1. Store voltage estimates (extract from state vector - already on host)
    const Real PI = 3.14159265359;
    const Real RAD_TO_DEG = 180.0 / PI;
    auto buses = network.getBuses();
    for (size_t i = 0; i < nBuses && i < buses.size(); ++i) {
        Bus* bus = buses[i];
        Real vPU = state.getVoltageMagnitude(static_cast<Index>(i));
        Real thetaRad = state.getVoltageAngle(static_cast<Index>(i));
        Real vKV = vPU * bus->getBaseKV();
        Real thetaDeg = thetaRad * RAD_TO_DEG;
        bus->setVoltEstimates(vPU, vKV, thetaRad, thetaDeg);
    }
    
    // 2. Copy power injections from GPU (already computed by evaluateGPU)
    if (!sharedDataManager_ || !sharedDataManager_->isInitialized()) {
        return;  // No GPU data available
    }
    
    Real* d_pInjection = sharedDataManager_->getPInjection();
    Real* d_qInjection = sharedDataManager_->getQInjection();
    
    if (d_pInjection && d_qInjection) {
        // OPTIMIZATION: Batch power injection transfers using async operations
        std::vector<Real> pInjection(nBuses), qInjection(nBuses);
    cudaStream_t stream = computeStream_;
        cuda::asyncCopyD2H(d_pInjection, pInjection.data(), nBuses, computeStream_);
        cuda::asyncCopyD2H(d_qInjection, qInjection.data(), nBuses, computeStream_);
        cudaStreamSynchronize(stream);
        
        Real baseMVA = network.getBaseMVA();
        // OPTIMIZATION: OpenMP parallelization for independent loop
#ifdef USE_OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < nBuses && i < buses.size(); ++i) {
            Bus* bus = buses[i];
            Real pMW = pInjection[i] * baseMVA;
            Real qMVAR = qInjection[i] * baseMVA;
            bus->setPowerInjections(pInjection[i], qInjection[i], pMW, qMVAR);
        }
    }
    
    // 3. For power flows: compute MW/MVAR/I_PU/I_Amps from existing GPU P/Q flows
    // (P/Q flows already computed, no need to recompute them)
    if (nBranches == 0) return;
    
    Real* d_pFlow = sharedDataManager_->getPFlow();
    Real* d_qFlow = sharedDataManager_->getQFlow();
    
    if (d_pFlow && d_qFlow) {
        // Get device pointers for branches and buses
        Real* d_v = sharedDataManager_->getStateV();
        DeviceBus* d_buses = sharedDataManager_->getBuses();
        DeviceBranch* d_branches = sharedDataManager_->getBranches();
        
        if (d_v && d_buses && d_branches) {
            // Use pooled GPU buffers for MW/MVAR/I_PU/I_Amps (no per-call allocation)
            // Ensure buffers are allocated
            if (memoryPool_.derivedQuantitiesSize < nBranches) {
                memoryPool_.ensureCapacity(0, 0, nBranches);
            }
            
            Real* d_pMW = memoryPool_.d_pMW;
            Real* d_qMVAR = memoryPool_.d_qMVAR;
            Real* d_iPU = memoryPool_.d_iPU;
            Real* d_iAmps = memoryPool_.d_iAmps;
            
            if (!d_pMW || !d_qMVAR || !d_iPU || !d_iAmps) {
                // Pool allocation failed, skip derived quantities
                // Still copy P/Q flows (use async with transfer stream for consistency)
                std::vector<Real> pFlow(nBranches), qFlow(nBranches);
    cudaStream_t transferStream = transferStream_;
                cuda::asyncCopyD2H(d_pFlow, pFlow.data(), nBranches, transferStream_);
                cuda::asyncCopyD2H(d_qFlow, qFlow.data(), nBranches, transferStream_);
                cudaStreamSynchronize(transferStream);
                
                auto branches = network.getBranches();
                // OPTIMIZATION: OpenMP parallelization for independent loop
#ifdef USE_OPENMP
                #pragma omp parallel for
#endif
                for (size_t i = 0; i < nBranches && i < branches.size(); ++i) {
                    branches[i]->setPowerFlow(pFlow[i], qFlow[i], 0.0, 0.0, 0.0, 0.0);
                }
                return;
            }
            
            Real baseMVA = network.getBaseMVA();
            
            // Compute MW/MVAR/I_PU/I_Amps from existing P/Q flows (no recomputation of P/Q)
            sle::cuda::computePowerFlowDerivedGPU(
                d_v, d_pFlow, d_qFlow, d_branches, d_buses, baseMVA,
                d_pMW, d_qMVAR, d_iPU, d_iAmps,
                static_cast<Index>(nBranches), static_cast<Index>(nBuses));
            
            // OPTIMIZATION: Batch all memory transfers using async operations with stream
            // This allows overlapping transfers and reduces kernel launch overhead
            std::vector<Real> pFlow(nBranches), qFlow(nBranches);
            std::vector<Real> pMW(nBranches), qMVAR(nBranches), iPU(nBranches), iAmps(nBranches);
    cudaStream_t stream = computeStream_;
            
            // Launch all async transfers (can overlap with other operations)
            cuda::asyncCopyD2H(d_pFlow, pFlow.data(), nBranches, computeStream_);
            cuda::asyncCopyD2H(d_qFlow, qFlow.data(), nBranches, computeStream_);
            cuda::asyncCopyD2H(d_pMW, pMW.data(), nBranches, computeStream_);
            cuda::asyncCopyD2H(d_qMVAR, qMVAR.data(), nBranches, computeStream_);
            cuda::asyncCopyD2H(d_iPU, iPU.data(), nBranches, computeStream_);
            cuda::asyncCopyD2H(d_iAmps, iAmps.data(), nBranches, computeStream_);
            
            // Synchronize once after all transfers are launched
            cudaStreamSynchronize(stream);
            
            // Store in Branch objects
            // OPTIMIZATION: OpenMP parallelization for independent loop
            auto branches = network.getBranches();
#ifdef USE_OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < nBranches && i < branches.size(); ++i) {
                branches[i]->setPowerFlow(pFlow[i], qFlow[i], pMW[i], qMVAR[i], iAmps[i], iPU[i]);
            }
            // Note: No need to free - buffers are pooled and reused
        }
    }
}

void Solver::computeGainMatrixGPU(const JacobianMatrix& H,
                                 const Real* d_weights,
                                 SparseMatrix& G, cusparseHandle_t cusparseHandle) {
    // G = H^T * W * H where W = diag(weights)
    // Use GPU device pointers directly
    
    Index nStates = H.getNCols();
    Index nMeas = H.getNRows();
    
    if (nStates == 0 || nMeas == 0) {
        std::vector<Real> values;
        std::vector<Index> rowPtr(nStates + 1, 0);
        std::vector<Index> colInd;
        G.buildFromCSR(values, rowPtr, colInd, nStates, nStates);
        return;
    }
    
    const SparseMatrix& HMat = H.getMatrix();
    if (HMat.getNNZ() == 0) {
        std::vector<Real> values;
        std::vector<Index> rowPtr(nStates + 1, 0);
        std::vector<Index> colInd;
        G.buildFromCSR(values, rowPtr, colInd, nStates, nStates);
        return;
    }
    
    // Get device pointers from H
    const Real* H_values = HMat.getValues();
    const Index* H_rowPtr = HMat.getRowPtr();
    const Index* H_colInd = HMat.getColInd();
    
    // Compute G = H^T * W * H entirely on GPU
    // Try direct formula first (G_ij = sum_k(H_ki * w_k * H_kj)) - faster for sparse matrices
    // If that fails, try SpGEMM (CUDA-exclusive, no CPU fallback)
    // Note: G_rowPtr, G_values, G_colInd are allocated here and given to SparseMatrix (which takes ownership)
    // We can't pool them because SparseMatrix owns them
    Real* G_values = nullptr;
    Index* G_rowPtr = nullptr;
    Index* G_colInd = nullptr;
    Index G_nnz = 0;
    
    bool reuse = (G.getNNZ() > 0 && G.getValues() && G.getRowPtr() && G.getColInd());
    
    if (reuse) {
        G_values = G.getValues();
        G_rowPtr = G.getRowPtr();
        G_colInd = G.getColInd();
        G_nnz = G.getNNZ();
    } else {
    // Allocate row pointer array (SparseMatrix will take ownership)
    cudaError_t err = cudaMalloc(&G_rowPtr, (nStates + 1) * sizeof(Index));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for gain matrix row pointers");
        }
    }
    
    // Try direct GPU computation first (computes G_ij = sum_k(H_ki * w_k * H_kj) directly)
    bool success = sle::cuda::computeGainMatrixDirect(cusparseHandle,
                                                      H_values, H_rowPtr, H_colInd,
                                                      d_weights,
                                                      G_values, G_rowPtr, G_colInd,
                                                      nMeas, nStates, G_nnz);
    
    // If direct method fails, try SpGEMM
    if (!success || G_nnz == 0) {
        if (!reuse) {
        if (G_rowPtr) cudaFree(G_rowPtr);
        if (G_values) cudaFree(G_values);
        if (G_colInd) cudaFree(G_colInd);
        
        // Reallocate row pointer for SpGEMM
            cudaError_t err = cudaMalloc(&G_rowPtr, (nStates + 1) * sizeof(Index));
            if (err != cudaSuccess) {
                 throw std::runtime_error("Failed to allocate GPU memory for gain matrix row pointers");
            }
        }
        
        // OPTIMIZATION: Use pooled WH_values buffer and pooled SpGEMM workspace to avoid per-iteration allocation
            success = sle::cuda::computeGainMatrixGPU(cusparseHandle,
                                                      H_values, H_rowPtr, H_colInd,
                                                      d_weights,
                                                      G_values, G_rowPtr, G_colInd,
                                                      nMeas, nStates, G_nnz,
                                                  memoryPool_.d_WH_values, memoryPool_.WH_valuesSize,
                                                  memoryPool_.d_spgemmBuffer1, memoryPool_.spgemmBuffer1Size,
                                                  memoryPool_.d_spgemmBuffer2, memoryPool_.spgemmBuffer2Size);
    }
    
    if (!success || G_nnz == 0) {
        if (!reuse) {
        if (G_rowPtr) cudaFree(G_rowPtr);
        if (G_values) cudaFree(G_values);
        if (G_colInd) cudaFree(G_colInd);
        }
        throw std::runtime_error("GPU gain matrix computation failed - CUDA-exclusive, no CPU fallback");
    }
    
    // Only buildFromDevicePointers if NOT reusing (to avoid double ownership/clear issues)
    if (!reuse) {
    // Build SparseMatrix directly from device pointers (avoids host-device copy)
    // SparseMatrix takes ownership of the device memory
    G.buildFromDevicePointers(G_values, G_rowPtr, G_colInd, nStates, nStates, G_nnz);
    }
}


void Solver::solveLinearSystemGPU(const SparseMatrix& G, Real* d_rhs, Real* d_deltaX, Index nStates) {
    // CUDA-EXCLUSIVE: Solve G * Δx = rhs entirely on GPU using cuSOLVER
    // Merged from CuSOLVERIntegration::solveSparseGPU
    
    if (G.getNNZ() == 0 || nStates == 0) {
        cudaMemsetAsync(d_deltaX, 0, nStates * sizeof(Real), computeStream_);
        return;
    }
    
    // Solve using cuSOLVER QR factorization (all data on GPU)
    int singularity = 0;
    double tol = 1e-6;
    int reorder = 0;
    
    cusolverStatus_t status = cusolverSpDcsrlsvqr(
        cusolverHandle_, nStates, G.getNNZ(), cusparseDescr_,
        G.getValues(), G.getRowPtr(), G.getColInd(),
        d_rhs, tol, reorder, d_deltaX, &singularity);
    
    if (status != CUSOLVER_STATUS_SUCCESS || singularity != -1) {
        // If solve fails, zero out deltaX (should not happen in normal operation)
        cudaMemsetAsync(d_deltaX, 0, nStates * sizeof(Real), computeStream_);
    }
}

} // namespace math
} // namespace sle

