/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Unified GPU data manager - keeps all data on GPU and reuses across operations
 */

#ifndef SLE_CUDA_CUDADATAMANAGER_H
#define SLE_CUDA_CUDADATAMANAGER_H

#include <sle/Types.h>
#include <memory>

namespace sle {
namespace cuda {

// Forward declarations
struct DeviceBus;
struct DeviceBranch;

// Unified GPU data manager - keeps all data on GPU
class CudaDataManager {
public:
    CudaDataManager();
    ~CudaDataManager();
    
    // Initialize with network size (allocates GPU memory)
    void initialize(Index nBuses, Index nBranches, Index nMeasurements);
    
    // Update state on GPU (copies from host once)
    void updateState(const Real* v, const Real* theta, Index nBuses);
    
    // Update network data on GPU (copies from host once, reused)
    void updateNetwork(const DeviceBus* buses, const DeviceBranch* branches,
                      Index nBuses, Index nBranches);
    
    // Update measurement data on GPU
    void updateMeasurements(const Index* types, const Index* locations,
                           const Index* branches, Index nMeasurements);
    
    // Update CSR adjacency lists on GPU
    void updateAdjacency(const Index* branchFromBus, const Index* branchFromBusRowPtr,
                        const Index* branchToBus, const Index* branchToBusRowPtr,
                        Index nBuses, Index fromSize, Index toSize);
    
    // Get device pointers (data stays on GPU)
    Real* getStateV() const { return d_v_; }
    Real* getStateTheta() const { return d_theta_; }
    DeviceBus* getBuses() const { return d_buses_; }
    DeviceBranch* getBranches() const { return d_branches_; }
    Index* getMeasurementTypes() const { return d_measurementTypes_; }
    Index* getMeasurementLocations() const { return d_measurementLocations_; }
    Index* getMeasurementBranches() const { return d_measurementBranches_; }
    Index* getBranchFromBus() const { return d_branchFromBus_; }
    Index* getBranchFromBusRowPtr() const { return d_branchFromBusRowPtr_; }
    Index* getBranchToBus() const { return d_branchToBus_; }
    Index* getBranchToBusRowPtr() const { return d_branchToBusRowPtr_; }
    
    // Output buffers (computed on GPU, can be read when needed)
    Real* getPInjection() const { return d_pInjection_; }
    Real* getQInjection() const { return d_qInjection_; }
    Real* getPFlow() const { return d_pFlow_; }
    Real* getQFlow() const { return d_qFlow_; }
    Real* getHx() const { return d_hx_; }
    
    // Check if initialized
    bool isInitialized() const { return initialized_; }
    
private:
    bool initialized_;
    Index nBuses_;
    Index nBranches_;
    Index nMeasurements_;
    
    // State data (updated each iteration)
    // May point to unified pool buffers if useUnifiedPool_ is true
    Real* d_v_;
    Real* d_theta_;
    size_t vSize_;
    size_t thetaSize_;
    
    // Network data (updated when network changes)
    // May point to unified pool buffers if useUnifiedPool_ is true
    DeviceBus* d_buses_;
    DeviceBranch* d_branches_;
    size_t busesSize_;
    size_t branchesSize_;
    
    // Measurement data (updated when measurements change)
    Index* d_measurementTypes_;
    Index* d_measurementLocations_;
    Index* d_measurementBranches_;
    
    // Adjacency lists (updated when network changes)
    Index* d_branchFromBus_;
    Index* d_branchFromBusRowPtr_;
    Index* d_branchToBus_;
    Index* d_branchToBusRowPtr_;
    Index branchFromBusSize_;
    Index branchToBusSize_;
    
    // Output buffers (computed on GPU)
    // These may point to unified pool buffers if useUnifiedPool_ is true
    Real* d_pInjection_;
    Real* d_qInjection_;
    Real* d_pFlow_;
    Real* d_qFlow_;
    Real* d_hx_;
    
    // Track sizes for unified pool buffers
    size_t pInjectionSize_;
    size_t qInjectionSize_;
    size_t pFlowSize_;
    size_t qFlowSize_;
    size_t hxSize_;
    
    void allocateMemory();
    void freeMemory();
};

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDADATAMANAGER_H

