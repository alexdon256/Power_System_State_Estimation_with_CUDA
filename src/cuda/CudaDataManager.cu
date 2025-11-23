/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/cuda/CudaDataManager.h>
#include <sle/cuda/CudaPowerFlow.h>
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
      d_pFlow_(nullptr), d_qFlow_(nullptr), d_hx_(nullptr) {
}

CudaDataManager::~CudaDataManager() {
    freeMemory();
}

void CudaDataManager::initialize(Index nBuses, Index nBranches, Index nMeasurements) {
    if (initialized_ && (nBuses_ != nBuses || nBranches_ != nBranches || nMeasurements_ != nMeasurements)) {
        freeMemory();
    }
    
    nBuses_ = nBuses;
    nBranches_ = nBranches;
    nMeasurements_ = nMeasurements;
    
    if (!initialized_) {
        allocateMemory();
        initialized_ = true;
    }
}

void CudaDataManager::allocateMemory() {
    cudaError_t err;
    
    // State data
    err = cudaMalloc(&d_v_, nBuses_ * sizeof(Real));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_v");
    
    err = cudaMalloc(&d_theta_, nBuses_ * sizeof(Real));
    if (err != cudaSuccess) {
        cudaFree(d_v_);
        throw std::runtime_error("Failed to allocate d_theta");
    }
    
    // Network data
    err = cudaMalloc(&d_buses_, nBuses_ * sizeof(DeviceBus));
    if (err != cudaSuccess) {
        cudaFree(d_v_);
        cudaFree(d_theta_);
        throw std::runtime_error("Failed to allocate d_buses");
    }
    
    err = cudaMalloc(&d_branches_, nBranches_ * sizeof(DeviceBranch));
    if (err != cudaSuccess) {
        cudaFree(d_v_);
        cudaFree(d_theta_);
        cudaFree(d_buses_);
        throw std::runtime_error("Failed to allocate d_branches");
    }
    
    // Measurement data
    err = cudaMalloc(&d_measurementTypes_, nMeasurements_ * sizeof(Index));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_measurementTypes");
    }
    
    err = cudaMalloc(&d_measurementLocations_, nMeasurements_ * sizeof(Index));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_measurementLocations");
    }
    
    err = cudaMalloc(&d_measurementBranches_, nMeasurements_ * sizeof(Index));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_measurementBranches");
    }
    
    // Output buffers
    err = cudaMalloc(&d_pInjection_, nBuses_ * sizeof(Real));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_pInjection");
    }
    
    err = cudaMalloc(&d_qInjection_, nBuses_ * sizeof(Real));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_qInjection");
    }
    
    err = cudaMalloc(&d_pFlow_, nBranches_ * sizeof(Real));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_pFlow");
    }
    
    err = cudaMalloc(&d_qFlow_, nBranches_ * sizeof(Real));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_qFlow");
    }
    
    err = cudaMalloc(&d_hx_, nMeasurements_ * sizeof(Real));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_hx");
    }
    
    // Adjacency lists allocated on update (size varies)
    d_branchFromBus_ = nullptr;
    d_branchFromBusRowPtr_ = nullptr;
    d_branchToBus_ = nullptr;
    d_branchToBusRowPtr_ = nullptr;
}

void CudaDataManager::freeMemory() {
    if (d_v_) cudaFree(d_v_);
    if (d_theta_) cudaFree(d_theta_);
    if (d_buses_) cudaFree(d_buses_);
    if (d_branches_) cudaFree(d_branches_);
    if (d_measurementTypes_) cudaFree(d_measurementTypes_);
    if (d_measurementLocations_) cudaFree(d_measurementLocations_);
    if (d_measurementBranches_) cudaFree(d_measurementBranches_);
    if (d_branchFromBus_) cudaFree(d_branchFromBus_);
    if (d_branchFromBusRowPtr_) cudaFree(d_branchFromBusRowPtr_);
    if (d_branchToBus_) cudaFree(d_branchToBus_);
    if (d_branchToBusRowPtr_) cudaFree(d_branchToBusRowPtr_);
    if (d_pInjection_) cudaFree(d_pInjection_);
    if (d_qInjection_) cudaFree(d_qInjection_);
    if (d_pFlow_) cudaFree(d_pFlow_);
    if (d_qFlow_) cudaFree(d_qFlow_);
    if (d_hx_) cudaFree(d_hx_);
    
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

