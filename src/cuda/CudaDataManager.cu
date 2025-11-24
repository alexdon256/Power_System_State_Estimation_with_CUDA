/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/cuda/CudaDataManager.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <sle/cuda/CudaUtils.h>
#include <sle/cuda/UnifiedCudaMemoryPool.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace sle {
namespace cuda {

CudaDataManager::CudaDataManager()
    : initialized_(false), nBuses_(0), nBranches_(0), nMeasurements_(0),
      d_v_(nullptr), d_theta_(nullptr),
      d_buses_(nullptr), d_branches_(nullptr),
      d_measurementTypes_(nullptr), d_measurementLocations_(nullptr),
      d_measurementBranches_(nullptr),
      d_branchFromBus_(nullptr), d_branchFromBusRowPtr_(nullptr),
      d_branchToBus_(nullptr), d_branchToBusRowPtr_(nullptr),
      branchFromBusSize_(0), branchToBusSize_(0),
      d_pInjection_(nullptr), d_qInjection_(nullptr),
      d_pFlow_(nullptr), d_qFlow_(nullptr), d_hx_(nullptr),
      pInjectionSize_(0), qInjectionSize_(0), pFlowSize_(0), qFlowSize_(0), hxSize_(0),
      useUnifiedPool_(true) {
}

CudaDataManager::~CudaDataManager() {
    freeMemory();
}

void CudaDataManager::initialize(Index nBuses, Index nBranches, Index nMeasurements, bool useUnifiedPool) {
    if (initialized_ && (nBuses_ != nBuses || nBranches_ != nBranches || nMeasurements_ != nMeasurements)) {
        freeMemory();
    }
    
    nBuses_ = nBuses;
    nBranches_ = nBranches;
    nMeasurements_ = nMeasurements;
    useUnifiedPool_ = useUnifiedPool;
    
    if (!initialized_) {
        if (useUnifiedPool_) {
            allocateWithUnifiedPool();
        } else {
            allocateMemory();
        }
        initialized_ = true;
    }
}

void CudaDataManager::allocateMemory() {
    // State data
    try {
        CUDA_CHECK_THROW(cudaMalloc(&d_v_, nBuses_ * sizeof(Real)));
        CUDA_CHECK_THROW(cudaMalloc(&d_theta_, nBuses_ * sizeof(Real)));
        
        // Network data
        CUDA_CHECK_THROW(cudaMalloc(&d_buses_, nBuses_ * sizeof(DeviceBus)));
        CUDA_CHECK_THROW(cudaMalloc(&d_branches_, nBranches_ * sizeof(DeviceBranch)));
        
        // Measurement data
        CUDA_CHECK_THROW(cudaMalloc(&d_measurementTypes_, nMeasurements_ * sizeof(Index)));
        CUDA_CHECK_THROW(cudaMalloc(&d_measurementLocations_, nMeasurements_ * sizeof(Index)));
        CUDA_CHECK_THROW(cudaMalloc(&d_measurementBranches_, nMeasurements_ * sizeof(Index)));
        
        // Output buffers
        CUDA_CHECK_THROW(cudaMalloc(&d_pInjection_, nBuses_ * sizeof(Real)));
        CUDA_CHECK_THROW(cudaMalloc(&d_qInjection_, nBuses_ * sizeof(Real)));
        CUDA_CHECK_THROW(cudaMalloc(&d_pFlow_, nBranches_ * sizeof(Real)));
        CUDA_CHECK_THROW(cudaMalloc(&d_qFlow_, nBranches_ * sizeof(Real)));
        CUDA_CHECK_THROW(cudaMalloc(&d_hx_, nMeasurements_ * sizeof(Real)));
    } catch (...) {
        freeMemory();
        throw;
    }
    
    // Adjacency lists allocated on update (size varies)
    d_branchFromBus_ = nullptr;
    d_branchFromBusRowPtr_ = nullptr;
    d_branchToBus_ = nullptr;
    d_branchToBusRowPtr_ = nullptr;
    
    // Initialize size tracking
    vSize_ = nBuses_;
    thetaSize_ = nBuses_;
    busesSize_ = nBuses_;
    branchesSize_ = nBranches_;
    pInjectionSize_ = nBuses_;
    qInjectionSize_ = nBuses_;
    pFlowSize_ = nBranches_;
    qFlowSize_ = nBranches_;
    hxSize_ = nMeasurements_;
}

void CudaDataManager::allocateWithUnifiedPool() {
    auto& pool = UnifiedCudaMemoryPool::getInstance();
    
    // Allocate measurement-specific buffers (not shared)
    try {
        CUDA_CHECK_THROW(cudaMalloc(&d_measurementTypes_, nMeasurements_ * sizeof(Index)));
        CUDA_CHECK_THROW(cudaMalloc(&d_measurementLocations_, nMeasurements_ * sizeof(Index)));
        CUDA_CHECK_THROW(cudaMalloc(&d_measurementBranches_, nMeasurements_ * sizeof(Index)));
    } catch (...) {
        if (d_measurementTypes_) cudaFree(d_measurementTypes_);
        if (d_measurementLocations_) cudaFree(d_measurementLocations_);
        if (d_measurementBranches_) cudaFree(d_measurementBranches_);
        throw;
    }
    
    // Get shared buffers from unified pool
    pool.ensureStateBuffers(nBuses_, d_v_, vSize_, d_theta_, thetaSize_);
    pool.ensureNetworkBuffers(nBuses_, nBranches_, d_buses_, busesSize_, d_branches_, branchesSize_);
    pool.ensurePowerInjectionBuffers(nBuses_, d_pInjection_, pInjectionSize_, d_qInjection_, qInjectionSize_);
    pool.ensurePowerFlowBuffers(nBranches_, d_pFlow_, pFlowSize_, d_qFlow_, qFlowSize_);
    pool.ensureMeasurementBuffer(nMeasurements_, d_hx_, hxSize_);
    
    // Adjacency lists allocated on update (size varies)
    d_branchFromBus_ = nullptr;
    d_branchFromBusRowPtr_ = nullptr;
    d_branchToBus_ = nullptr;
    d_branchToBusRowPtr_ = nullptr;
}

void CudaDataManager::freeMemory() {
    // Only free buffers we own (not from unified pool)
    if (!useUnifiedPool_) {
        if (d_v_) cudaFree(d_v_);
        if (d_theta_) cudaFree(d_theta_);
        if (d_buses_) cudaFree(d_buses_);
        if (d_branches_) cudaFree(d_branches_);
        if (d_pInjection_) cudaFree(d_pInjection_);
        if (d_qInjection_) cudaFree(d_qInjection_);
        if (d_pFlow_) cudaFree(d_pFlow_);
        if (d_qFlow_) cudaFree(d_qFlow_);
        if (d_hx_) cudaFree(d_hx_);
    }
    
    // Always free measurement-specific buffers (we own these)
    if (d_measurementTypes_) cudaFree(d_measurementTypes_);
    if (d_measurementLocations_) cudaFree(d_measurementLocations_);
    if (d_measurementBranches_) cudaFree(d_measurementBranches_);
    
    // Always free adjacency buffers (we own these)
    if (d_branchFromBus_) cudaFree(d_branchFromBus_);
    if (d_branchFromBusRowPtr_) cudaFree(d_branchFromBusRowPtr_);
    if (d_branchToBus_) cudaFree(d_branchToBus_);
    if (d_branchToBusRowPtr_) cudaFree(d_branchToBusRowPtr_);
    
    d_v_ = nullptr;
    d_theta_ = nullptr;
    d_buses_ = nullptr;
    d_branches_ = nullptr;
    d_measurementTypes_ = nullptr;
    d_measurementLocations_ = nullptr;
    d_measurementBranches_ = nullptr;
    d_branchFromBus_ = nullptr;
    d_branchFromBusRowPtr_ = nullptr;
    d_branchToBus_ = nullptr;
    d_branchToBusRowPtr_ = nullptr;
    d_pInjection_ = nullptr;
    d_qInjection_ = nullptr;
    d_pFlow_ = nullptr;
    d_qFlow_ = nullptr;
    d_hx_ = nullptr;
    
    vSize_ = 0;
    thetaSize_ = 0;
    busesSize_ = 0;
    branchesSize_ = 0;
    pInjectionSize_ = 0;
    qInjectionSize_ = 0;
    pFlowSize_ = 0;
    qFlowSize_ = 0;
    hxSize_ = 0;
    
    initialized_ = false;
}

void CudaDataManager::updateState(const Real* v, const Real* theta, Index nBuses) {
    if (!initialized_ || nBuses != nBuses_) {
        throw std::runtime_error("CudaDataManager not initialized or size mismatch");
    }
    
    cudaMemcpy(d_v_, v, nBuses * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta_, theta, nBuses * sizeof(Real), cudaMemcpyHostToDevice);
}

void CudaDataManager::updateNetwork(const DeviceBus* buses, const DeviceBranch* branches,
                                    Index nBuses, Index nBranches) {
    if (!initialized_ || nBuses != nBuses_ || nBranches != nBranches_) {
        throw std::runtime_error("CudaDataManager not initialized or size mismatch");
    }
    
    cudaMemcpy(d_buses_, buses, nBuses * sizeof(DeviceBus), cudaMemcpyHostToDevice);
    cudaMemcpy(d_branches_, branches, nBranches * sizeof(DeviceBranch), cudaMemcpyHostToDevice);
}

void CudaDataManager::updateMeasurements(const Index* types, const Index* locations,
                                        const Index* branches, Index nMeasurements) {
    if (!initialized_ || nMeasurements != nMeasurements_) {
        throw std::runtime_error("CudaDataManager not initialized or size mismatch");
    }
    
    cudaMemcpy(d_measurementTypes_, types, nMeasurements * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_measurementLocations_, locations, nMeasurements * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_measurementBranches_, branches, nMeasurements * sizeof(Index), cudaMemcpyHostToDevice);
}

void CudaDataManager::updateAdjacency(const Index* branchFromBus, const Index* branchFromBusRowPtr,
                                     const Index* branchToBus, const Index* branchToBusRowPtr,
                                     Index nBuses, Index fromSize, Index toSize) {
    if (!initialized_ || nBuses != nBuses_) {
        throw std::runtime_error("CudaDataManager not initialized or size mismatch");
    }
    
    // Free old allocations if sizes changed
    if (fromSize != branchFromBusSize_) {
        if (d_branchFromBus_) cudaFree(d_branchFromBus_);
        branchFromBusSize_ = fromSize;
        if (fromSize > 0) {
            cudaError_t err = cudaMalloc(&d_branchFromBus_, fromSize * sizeof(Index));
            if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_branchFromBus");
        }
    }
    
    if (toSize != branchToBusSize_) {
        if (d_branchToBus_) cudaFree(d_branchToBus_);
        branchToBusSize_ = toSize;
        if (toSize > 0) {
            cudaError_t err = cudaMalloc(&d_branchToBus_, toSize * sizeof(Index));
            if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_branchToBus");
        }
    }
    
    // Allocate row pointers if needed
    if (!d_branchFromBusRowPtr_) {
        cudaError_t err = cudaMalloc(&d_branchFromBusRowPtr_, (nBuses + 1) * sizeof(Index));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_branchFromBusRowPtr");
    }
    
    if (!d_branchToBusRowPtr_) {
        cudaError_t err = cudaMalloc(&d_branchToBusRowPtr_, (nBuses + 1) * sizeof(Index));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_branchToBusRowPtr");
    }
    
    // Copy data
    if (fromSize > 0) {
        cudaMemcpy(d_branchFromBus_, branchFromBus, fromSize * sizeof(Index), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_branchFromBusRowPtr_, branchFromBusRowPtr, (nBuses + 1) * sizeof(Index), cudaMemcpyHostToDevice);
    
    if (toSize > 0) {
        cudaMemcpy(d_branchToBus_, branchToBus, toSize * sizeof(Index), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_branchToBusRowPtr_, branchToBusRowPtr, (nBuses + 1) * sizeof(Index), cudaMemcpyHostToDevice);
}

} // namespace cuda
} // namespace sle

