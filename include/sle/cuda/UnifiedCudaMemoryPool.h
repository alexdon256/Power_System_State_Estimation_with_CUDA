/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Unified GPU memory pool singleton for sharing buffers across components
 * Eliminates duplicate allocations and improves memory utilization
 */

#ifndef SLE_CUDA_UNIFIEDCUDAMEMORYPOOL_H
#define SLE_CUDA_UNIFIEDCUDAMEMORYPOOL_H

#include <sle/Types.h>
#include <sle/cuda/CudaUtils.h>
#include <memory>
#include <mutex>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

namespace sle {
namespace cuda {

// Forward declarations
struct DeviceBus;
struct DeviceBranch;

// Unified GPU memory pool singleton
// Pools commonly used buffers to eliminate duplicate allocations
class UnifiedCudaMemoryPool {
public:
    // Get singleton instance
    static UnifiedCudaMemoryPool& getInstance();
    
    // Disable copy and assignment
    UnifiedCudaMemoryPool(const UnifiedCudaMemoryPool&) = delete;
    UnifiedCudaMemoryPool& operator=(const UnifiedCudaMemoryPool&) = delete;
    
    // Ensure state buffers (d_v, d_theta) - shared by Solver, NetworkModel, JacobianMatrix
    void ensureStateBuffers(size_t nBuses, Real*& d_v, size_t& vSize, Real*& d_theta, size_t& thetaSize);
    
    // Ensure network buffers (d_buses, d_branches) - shared by multiple components
    void ensureNetworkBuffers(size_t nBuses, size_t nBranches,
                              DeviceBus*& d_buses, size_t& busesSize,
                              DeviceBranch*& d_branches, size_t& branchesSize);
    
    // Ensure power injection buffers - shared by NetworkModel and Solver
    void ensurePowerInjectionBuffers(size_t nBuses,
                                     Real*& d_pInjection, size_t& pInjectionSize,
                                     Real*& d_qInjection, size_t& qInjectionSize);
    
    // Ensure power flow buffers - shared by NetworkModel and Solver
    void ensurePowerFlowBuffers(size_t nBranches,
                                Real*& d_pFlow, size_t& pFlowSize,
                                Real*& d_qFlow, size_t& qFlowSize);
    
    // Ensure measurement function buffer (d_hx) - shared by MeasurementFunctions and Solver
    void ensureMeasurementBuffer(size_t nMeasurements, Real*& d_hx, size_t& hxSize);
    
    // Ensure partial reduction buffer - shared across all reduction operations
    void ensurePartialBuffer(size_t requiredSize, Real*& d_partial, size_t& partialSize);
    
    // Ensure derived quantity buffers (MW/MVAR/I) - shared by NetworkModel and Solver
    void ensureDerivedQuantityBuffers(size_t nBranches,
                                      Real*& d_pMW, size_t& pMWSize,
                                      Real*& d_qMVAR, size_t& qMVARSize,
                                      Real*& d_iPU, size_t& iPUSize,
                                      Real*& d_iAmps, size_t& iAmpsSize);
    
    // Ensure CSR adjacency buffers - shared by NetworkModel and JacobianMatrix
    void ensureAdjacencyBuffers(size_t nBuses, size_t maxCSRSize,
                                Index*& d_branchFromBus, size_t& branchFromBusSize,
                                Index*& d_branchToBus, size_t& branchToBusSize,
                                Index*& d_branchFromBusRowPtr, size_t& branchFromBusRowPtrSize,
                                Index*& d_branchToBusRowPtr, size_t& branchToBusRowPtrSize);
    
    // Clear all buffers (for testing or memory cleanup)
    void clear();
    
    // Get memory usage statistics
    struct MemoryStats {
        size_t totalBuffers;
        size_t totalBytes;
        size_t stateBuffersBytes;
        size_t networkBuffersBytes;
        size_t powerBuffersBytes;
        size_t measurementBuffersBytes;
        size_t reductionBuffersBytes;
    };
    MemoryStats getStats() const;
    
private:
    UnifiedCudaMemoryPool() = default;
    ~UnifiedCudaMemoryPool();
    
    // Thread-safe mutex for concurrent access
    mutable std::mutex mutex_;
    
    // State buffers
    Real* d_v_ = nullptr;
    Real* d_theta_ = nullptr;
    size_t vSize_ = 0;
    size_t thetaSize_ = 0;
    
    // Network buffers
    DeviceBus* d_buses_ = nullptr;
    DeviceBranch* d_branches_ = nullptr;
    size_t busesSize_ = 0;
    size_t branchesSize_ = 0;
    
    // Power injection buffers
    Real* d_pInjection_ = nullptr;
    Real* d_qInjection_ = nullptr;
    size_t pInjectionSize_ = 0;
    size_t qInjectionSize_ = 0;
    
    // Power flow buffers
    Real* d_pFlow_ = nullptr;
    Real* d_qFlow_ = nullptr;
    size_t pFlowSize_ = 0;
    size_t qFlowSize_ = 0;
    
    // Measurement buffer
    Real* d_hx_ = nullptr;
    size_t hxSize_ = 0;
    
    // Partial reduction buffer
    Real* d_partial_ = nullptr;
    size_t partialSize_ = 0;
    
    // Derived quantity buffers
    Real* d_pMW_ = nullptr;
    Real* d_qMVAR_ = nullptr;
    Real* d_iPU_ = nullptr;
    Real* d_iAmps_ = nullptr;
    size_t pMWSize_ = 0;
    size_t qMVARSize_ = 0;
    size_t iPUSize_ = 0;
    size_t iAmpsSize_ = 0;
    
    // CSR adjacency buffers
    Index* d_branchFromBus_ = nullptr;
    Index* d_branchToBus_ = nullptr;
    Index* d_branchFromBusRowPtr_ = nullptr;
    Index* d_branchToBusRowPtr_ = nullptr;
    size_t branchFromBusSize_ = 0;
    size_t branchToBusSize_ = 0;
    size_t branchFromBusRowPtrSize_ = 0;
    size_t branchToBusRowPtrSize_ = 0;
    
    void freeAllBuffers();
};

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_UNIFIEDCUDAMEMORYPOOL_H

