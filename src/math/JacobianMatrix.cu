/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/JacobianMatrix.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <sle/cuda/CudaNetworkUtils.h>
#include <sle/cuda/CudaDataManager.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace sle {
namespace math {

using model::NetworkModel;
using model::StateVector;
using model::TelemetryData;

JacobianMatrix::JacobianMatrix() : nRows_(0), nCols_(0), structureBuilt_(false) {
}

JacobianMatrix::~JacobianMatrix() = default;

void JacobianMatrix::buildStructure(const NetworkModel& network,
                                    const TelemetryData& telemetry) {
    auto measurements = telemetry.getMeasurements();
    nRows_ = measurements.size();
    nCols_ = 2 * network.getBusCount();  // angles + magnitudes
    
    std::vector<Index> rowPtr(nRows_ + 1, 0);
    std::vector<Index> colInd;
    
    // Build CSR structure
    for (size_t i = 0; i < measurements.size(); ++i) {
        const auto* m = measurements[i];
        if (!m) continue;
        
        switch (m->getType()) {
            case MeasurementType::P_FLOW:
            case MeasurementType::Q_FLOW: {
                // Depends on from and to bus
                Index fromBus = network.getBusIndex(m->getFromBus());
                Index toBus = network.getBusIndex(m->getToBus());
                if (fromBus >= 0) colInd.push_back(fromBus);
                if (toBus >= 0) colInd.push_back(toBus);
                if (fromBus >= 0) colInd.push_back(nCols_ / 2 + fromBus);
                if (toBus >= 0) colInd.push_back(nCols_ / 2 + toBus);
                break;
            }
            case MeasurementType::P_INJECTION:
            case MeasurementType::Q_INJECTION: {
                // Depends on connected buses only (sparse structure)
                // Power injection at a bus depends on:
                // 1. The bus itself (angle and magnitude)
                // 2. All buses directly connected via branches
                Index busIdx = network.getBusIndex(m->getLocation());
                if (busIdx >= 0) {
                    // Add the bus itself
                    colInd.push_back(busIdx);  // dP/dθ or dQ/dθ
                    colInd.push_back(nCols_ / 2 + busIdx);  // dP/dV or dQ/dV
                    
                    // Add all connected buses (via branches)
                    auto branchesFrom = network.getBranchesFromBus(m->getLocation());
                    auto branchesTo = network.getBranchesToBus(m->getLocation());
                    
                    // Add buses connected via outgoing branches
                    for (const auto* branch : branchesFrom) {
                        Index toBus = network.getBusIndex(branch->getToBus());
                        if (toBus >= 0 && toBus != busIdx) {
                            colInd.push_back(toBus);  // dP/dθ or dQ/dθ
                            colInd.push_back(nCols_ / 2 + toBus);  // dP/dV or dQ/dV
                        }
                    }
                    
                    // Add buses connected via incoming branches
                    for (const auto* branch : branchesTo) {
                        Index fromBus = network.getBusIndex(branch->getFromBus());
                        if (fromBus >= 0 && fromBus != busIdx) {
                            colInd.push_back(fromBus);  // dP/dθ or dQ/dθ
                            colInd.push_back(nCols_ / 2 + fromBus);  // dP/dV or dQ/dV
                        }
                    }
                }
                break;
            }
            case MeasurementType::V_MAGNITUDE: {
                Index busIdx = network.getBusIndex(m->getLocation());
                if (busIdx >= 0) {
                    colInd.push_back(nCols_ / 2 + busIdx);
                }
                break;
            }
            case MeasurementType::I_MAGNITUDE: {
                // Similar to flow
                Index fromBus = network.getBusIndex(m->getFromBus());
                Index toBus = network.getBusIndex(m->getToBus());
                if (fromBus >= 0) colInd.push_back(fromBus);
                if (toBus >= 0) colInd.push_back(toBus);
                if (fromBus >= 0) colInd.push_back(nCols_ / 2 + fromBus);
                if (toBus >= 0) colInd.push_back(nCols_ / 2 + toBus);
                break;
            }
            default:
                break;
        }
        
        rowPtr[i + 1] = colInd.size();
    }
    
    // Build matrix structure (values will be filled by buildGPU)
    std::vector<Real> values(colInd.size(), 0.0);
    matrix_.buildFromCSR(values, rowPtr, colInd, nRows_, nCols_);
    
    // Store host CSR structure (only structure needed, values stay on GPU)
    hostRowPtr_ = rowPtr;
    hostColInd_ = colInd;
    structureBuilt_ = true;
}

void JacobianMatrix::buildGPU(const StateVector& state,
                             const NetworkModel& network,
                             const TelemetryData& telemetry) {
    // Call overloaded version without shared data manager (backward compatibility)
    buildGPU(state, network, telemetry, nullptr);
}

void JacobianMatrix::buildGPU(const StateVector& state,
                             const NetworkModel& network,
                             const TelemetryData& telemetry,
                             cuda::CudaDataManager* dataManager) {
    // OPTIMIZATION: Build structure only once (structure doesn't change between iterations)
    if (!structureBuilt_) {
        buildStructure(network, telemetry);
    }
    
    // Get state data
    const auto& v = state.getMagnitudes();
    const auto& theta = state.getAngles();
    size_t nBuses = network.getBusCount();
    size_t nBranches = network.getBranchCount();
    size_t nMeasurements = telemetry.getMeasurements().size();
    
    if (nMeasurements == 0 || nBuses == 0 || matrix_.getNNZ() == 0) {
        return;  // No measurements or empty matrix
    }
    
    // Use structure already built by buildStructure (called at start of buildGPU)
    // Reuse stored hostRowPtr_ and hostColInd_ to avoid duplicate structure building
    if (hostRowPtr_.empty() || hostColInd_.empty()) {
        throw std::runtime_error("Jacobian structure not built - buildStructure must be called first");
    }
    
    std::vector<Index> jacobianRowPtr = hostRowPtr_;
    std::vector<Index> jacobianColInd = hostColInd_;
    
    // If shared CudaDataManager is provided, reuse its GPU data (avoids duplicate allocations)
    if (dataManager) {
        // Verify data manager is initialized with correct sizes
        // Note: MeasurementFunctions should have initialized it already
        const auto& v = state.getMagnitudes();
        const auto& theta = state.getAngles();
        dataManager->updateState(v.data(), theta.data(), static_cast<Index>(nBuses));
        
        // Use buildGPUFromPointers with shared data
        buildGPUFromPointers(
            dataManager->getStateV(),
            dataManager->getStateTheta(),
            dataManager->getBuses(),
            dataManager->getBranches(),
            dataManager->getBranchFromBus(),
            dataManager->getBranchFromBusRowPtr(),
            dataManager->getBranchToBus(),
            dataManager->getBranchToBusRowPtr(),
            dataManager->getMeasurementTypes(),
            dataManager->getMeasurementLocations(),
            dataManager->getMeasurementBranches(),
            static_cast<Index>(nBuses),
            static_cast<Index>(nBranches),
            static_cast<Index>(nMeasurements));
        return;
    }
    
    // Fallback: Build device structures if no shared data manager
    // Prepare device data structures using utility functions
    std::vector<sle::cuda::DeviceBus> deviceBuses;
    sle::cuda::buildDeviceBuses(network, deviceBuses);
    
    std::vector<sle::cuda::DeviceBranch> deviceBranches;
    sle::cuda::buildDeviceBranches(network, deviceBranches);
    
    // Build CSR adjacency lists
    std::vector<Index> branchFromBus, branchToBus;
    std::vector<Index> branchFromBusRowPtr(nBuses + 1, 0);
    std::vector<Index> branchToBusRowPtr(nBuses + 1, 0);
    sle::cuda::buildCSRAdjacencyLists(network, branchFromBus, branchFromBusRowPtr,
                                      branchToBus, branchToBusRowPtr);
    
    // Prepare measurement data (reuse utility functions)
    std::vector<Index> measurementTypes;
    std::vector<Index> measurementLocations;
    std::vector<Index> measurementBranches;
    measurementTypes.reserve(nMeasurements);
    measurementLocations.reserve(nMeasurements);
    measurementBranches.reserve(nMeasurements);
    
    auto measurements = telemetry.getMeasurements();
    for (const auto* meas : measurements) {
        if (!meas) continue;
        measurementTypes.push_back(sle::cuda::mapMeasurementTypeToIndex(meas->getType()));
        measurementLocations.push_back(network.getBusIndex(meas->getLocation()));
        measurementBranches.push_back(sle::cuda::findBranchIndex(network, 
                                                                  meas->getFromBus(), 
                                                                  meas->getToBus()));
    }
    
    // Allocate GPU memory (fallback path - not using shared data manager)
    Real* d_v = nullptr;
    Real* d_theta = nullptr;
    sle::cuda::DeviceBus* d_buses = nullptr;
    sle::cuda::DeviceBranch* d_branches = nullptr;
    Index* d_branchFromBus = nullptr;
    Index* d_branchToBus = nullptr;
    Index* d_branchFromBusRowPtr = nullptr;
    Index* d_branchToBusRowPtr = nullptr;
    Index* d_measurementTypes = nullptr;
    Index* d_measurementLocations = nullptr;
    Index* d_measurementBranches = nullptr;
    Index* d_jacobianRowPtr = nullptr;
    Index* d_jacobianColInd = nullptr;
    Real* d_jacobianValues = nullptr;
    
    cudaError_t err;
    err = cudaMalloc(&d_v, nBuses * sizeof(Real));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_theta, nBuses * sizeof(Real));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_buses, nBuses * sizeof(sle::cuda::DeviceBus));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_branches, nBranches * sizeof(sle::cuda::DeviceBranch));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_branchFromBus, branchFromBus.size() * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_branchToBus, branchToBus.size() * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_branchFromBusRowPtr, (nBuses + 1) * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_branchToBusRowPtr, (nBuses + 1) * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_measurementTypes, nMeasurements * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_measurementLocations, nMeasurements * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_measurementBranches, nMeasurements * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_jacobianRowPtr, (nRows_ + 1) * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_jacobianColInd, jacobianColInd.size() * sizeof(Index));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_jacobianValues, jacobianColInd.size() * sizeof(Real));
    if (err != cudaSuccess) goto cleanup;
    
    // Copy data to GPU
    cudaMemcpy(d_v, v.data(), nBuses * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta.data(), nBuses * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buses, deviceBuses.data(), nBuses * sizeof(sle::cuda::DeviceBus), cudaMemcpyHostToDevice);
    cudaMemcpy(d_branches, deviceBranches.data(), nBranches * sizeof(sle::cuda::DeviceBranch), cudaMemcpyHostToDevice);
    cudaMemcpy(d_branchFromBus, branchFromBus.data(), branchFromBus.size() * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_branchToBus, branchToBus.data(), branchToBus.size() * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_branchFromBusRowPtr, branchFromBusRowPtr.data(), (nBuses + 1) * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_branchToBusRowPtr, branchToBusRowPtr.data(), (nBuses + 1) * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_measurementTypes, measurementTypes.data(), nMeasurements * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_measurementLocations, measurementLocations.data(), nMeasurements * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_measurementBranches, measurementBranches.data(), nMeasurements * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobianRowPtr, jacobianRowPtr.data(), (nRows_ + 1) * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobianColInd, jacobianColInd.data(), jacobianColInd.size() * sizeof(Index), cudaMemcpyHostToDevice);
    
    // Call CUDA kernel to compute Jacobian values
    // OPTIMIZATION: Use stream for asynchronous execution
    sle::cuda::computeJacobian(
        d_v, d_theta, d_buses, d_branches,
        d_branchFromBus, d_branchFromBusRowPtr,
        d_branchToBus, d_branchToBusRowPtr,
        d_measurementTypes, d_measurementLocations, d_measurementBranches,
        d_jacobianRowPtr, d_jacobianColInd, d_jacobianValues,
        static_cast<Index>(nMeasurements), static_cast<Index>(nBuses), static_cast<Index>(nBranches),
        stream);
    
    // OPTIMIZATION: Use stream synchronization instead of device sync
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    // Build sparse matrix directly from device pointers (avoids host-device copy)
    // SparseMatrix takes ownership of the device memory
    matrix_.buildFromDevicePointers(d_jacobianValues, d_jacobianRowPtr, d_jacobianColInd,
                                    nRows_, nCols_, static_cast<Index>(jacobianColInd.size()));
    
    // Store structure (values stay on GPU)
    hostRowPtr_ = jacobianRowPtr;
    hostColInd_ = jacobianColInd;
    
    // Mark that we don't need to free these pointers (SparseMatrix owns them now)
    d_jacobianValues = nullptr;
    d_jacobianRowPtr = nullptr;
    d_jacobianColInd = nullptr;
    
cleanup:
    // Free GPU memory (except Jacobian pointers if transferred to SparseMatrix)
    if (d_v) cudaFree(d_v);
    if (d_theta) cudaFree(d_theta);
    if (d_buses) cudaFree(d_buses);
    if (d_branches) cudaFree(d_branches);
    if (d_branchFromBus) cudaFree(d_branchFromBus);
    if (d_branchToBus) cudaFree(d_branchToBus);
    if (d_branchFromBusRowPtr) cudaFree(d_branchFromBusRowPtr);
    if (d_branchToBusRowPtr) cudaFree(d_branchToBusRowPtr);
    if (d_measurementTypes) cudaFree(d_measurementTypes);
    if (d_measurementLocations) cudaFree(d_measurementLocations);
    if (d_measurementBranches) cudaFree(d_measurementBranches);
    // Note: d_jacobianRowPtr, d_jacobianColInd, d_jacobianValues are owned by SparseMatrix if buildFromDevicePointers succeeded
    if (d_jacobianRowPtr) cudaFree(d_jacobianRowPtr);
    if (d_jacobianColInd) cudaFree(d_jacobianColInd);
    if (d_jacobianValues) cudaFree(d_jacobianValues);
}

void JacobianMatrix::buildGPUFromPointers(const Real* d_v, const Real* d_theta,
                                          const sle::cuda::DeviceBus* d_buses, 
                                          const sle::cuda::DeviceBranch* d_branches,
                                          const Index* d_branchFromBus, 
                                          const Index* d_branchFromBusRowPtr,
                                          const Index* d_branchToBus, 
                                          const Index* d_branchToBusRowPtr,
                                          const Index* d_measurementTypes, 
                                          const Index* d_measurementLocations,
                                          const Index* d_measurementBranches,
                                          Index nBuses, Index nBranches, Index nMeasurements) {
    // CUDA-EXCLUSIVE: Build Jacobian using GPU pointers (reuses GPU data)
    // This eliminates redundant host-device transfers
    
    if (nMeasurements == 0 || nBuses == 0 || matrix_.getNNZ() == 0) {
        return;
    }
    
    // Get Jacobian CSR structure (must be built first via buildStructure)
    std::vector<Index> jacobianRowPtr(nRows_ + 1, 0);
    std::vector<Index> jacobianColInd;
    jacobianColInd.reserve(matrix_.getNNZ());
    
    // Use existing hostRowPtr_ and hostColInd_ (must be built via buildStructure first)
    if (hostRowPtr_.empty() || hostColInd_.empty()) {
        throw std::runtime_error("Jacobian structure not built - call buildStructure first");
    }
    
    jacobianRowPtr = hostRowPtr_;
    jacobianColInd = hostColInd_;
    
    // OPTIMIZATION: Check if we can reuse existing matrix buffers (Zero-Allocation Update)
    // If matrix dimensions and NNZ match, we assume structure is identical (since we only support static topology)
    bool canReuse = (matrix_.getNNZ() == static_cast<Index>(jacobianColInd.size()) &&
                     matrix_.getNRows() == nRows_ &&
                     matrix_.getNCols() == nCols_ &&
                     matrix_.getValues() != nullptr &&
                     matrix_.getRowPtr() != nullptr &&
                     matrix_.getColInd() != nullptr);
    
    // Allocate GPU memory only if not reusing
    Index* d_jacobianRowPtr = nullptr;
    Index* d_jacobianColInd = nullptr;
    Real* d_jacobianValues = nullptr;
    
    if (canReuse) {
        // Reuse existing GPU buffers
        d_jacobianRowPtr = matrix_.getRowPtr();
        d_jacobianColInd = matrix_.getColInd();
        d_jacobianValues = matrix_.getValues();
    } else {
        cudaError_t err;
        err = cudaMalloc(&d_jacobianRowPtr, (nRows_ + 1) * sizeof(Index));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_jacobianRowPtr");
        
        err = cudaMalloc(&d_jacobianColInd, jacobianColInd.size() * sizeof(Index));
        if (err != cudaSuccess) {
            cudaFree(d_jacobianRowPtr);
            throw std::runtime_error("Failed to allocate d_jacobianColInd");
        }
        
        err = cudaMalloc(&d_jacobianValues, jacobianColInd.size() * sizeof(Real));
        if (err != cudaSuccess) {
            cudaFree(d_jacobianRowPtr);
            cudaFree(d_jacobianColInd);
            throw std::runtime_error("Failed to allocate d_jacobianValues");
        }
        
        // Copy Jacobian structure to GPU (only needed on initialization)
        cudaMemcpy(d_jacobianRowPtr, jacobianRowPtr.data(), (nRows_ + 1) * sizeof(Index), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jacobianColInd, jacobianColInd.data(), jacobianColInd.size() * sizeof(Index), cudaMemcpyHostToDevice);
    }
    
    // Call CUDA kernel using provided GPU pointers (reuses GPU data)
    // OPTIMIZATION: Use stream for asynchronous execution
    sle::cuda::computeJacobian(
        d_v, d_theta, d_buses, d_branches,
        d_branchFromBus, d_branchFromBusRowPtr,
        d_branchToBus, d_branchToBusRowPtr,
        d_measurementTypes, d_measurementLocations, d_measurementBranches,
        d_jacobianRowPtr, d_jacobianColInd, d_jacobianValues,
        nMeasurements, nBuses, nBranches,
        stream);
    
    // OPTIMIZATION: Use stream synchronization instead of device sync
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    if (!canReuse) {
        // Build sparse matrix directly from device pointers (avoids host-device copy)
        // SparseMatrix takes ownership of the device memory
        matrix_.buildFromDevicePointers(d_jacobianValues, d_jacobianRowPtr, d_jacobianColInd,
                                        nRows_, nCols_, static_cast<Index>(jacobianColInd.size()));
        
        // Mark that we don't need to free these pointers (SparseMatrix owns them now)
        d_jacobianValues = nullptr;
        d_jacobianRowPtr = nullptr;
        d_jacobianColInd = nullptr;
    }
    
    // Store structure (values stay on GPU)
    hostRowPtr_ = jacobianRowPtr;
    hostColInd_ = jacobianColInd;
}

void JacobianMatrix::getHostCSR(std::vector<Real>& values, std::vector<Index>& rowPtr, std::vector<Index>& colInd) const {
    // CUDA-EXCLUSIVE: Only provide structure, values stay on GPU
    // This method is kept for compatibility but should not be used for CPU computations
    rowPtr = hostRowPtr_;
    colInd = hostColInd_;
    
    // Copy values from GPU if needed (for debugging/compatibility only)
    if (matrix_.getNNZ() > 0) {
        values.resize(matrix_.getNNZ());
        cudaMemcpy(values.data(), matrix_.getValues(), matrix_.getNNZ() * sizeof(Real), cudaMemcpyDeviceToHost);
    } else {
        values.clear();
    }
}

} // namespace math
} // namespace sle

