/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Unified GPU memory pool singleton implementation
 */

#include <sle/cuda/UnifiedCudaMemoryPool.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <cuda_runtime.h>

namespace sle {
namespace cuda {

UnifiedCudaMemoryPool& UnifiedCudaMemoryPool::getInstance() {
    static UnifiedCudaMemoryPool instance;
    return instance;
}

UnifiedCudaMemoryPool::~UnifiedCudaMemoryPool() {
    freeAllBuffers();
}

void UnifiedCudaMemoryPool::ensureStateBuffers(size_t nBuses, Real*& d_v, size_t& vSize, 
                                                Real*& d_theta, size_t& thetaSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    cuda::ensureCapacity(d_v_, vSize_, nBuses);
    cuda::ensureCapacity(d_theta_, thetaSize_, nBuses);
    
    d_v = d_v_;
    vSize = vSize_;
    d_theta = d_theta_;
    thetaSize = thetaSize_;
}

void UnifiedCudaMemoryPool::ensureNetworkBuffers(size_t nBuses, size_t nBranches,
                                                  DeviceBus*& d_buses, size_t& busesSize,
                                                  DeviceBranch*& d_branches, size_t& branchesSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    cuda::ensureCapacity(d_buses_, busesSize_, nBuses);
    cuda::ensureCapacity(d_branches_, branchesSize_, nBranches);
    
    d_buses = d_buses_;
    busesSize = busesSize_;
    d_branches = d_branches_;
    branchesSize = branchesSize_;
}

void UnifiedCudaMemoryPool::ensurePowerInjectionBuffers(size_t nBuses,
                                                         Real*& d_pInjection, size_t& pInjectionSize,
                                                         Real*& d_qInjection, size_t& qInjectionSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    cuda::ensureCapacity(d_pInjection_, pInjectionSize_, nBuses);
    cuda::ensureCapacity(d_qInjection_, qInjectionSize_, nBuses);
    
    d_pInjection = d_pInjection_;
    pInjectionSize = pInjectionSize_;
    d_qInjection = d_qInjection_;
    qInjectionSize = qInjectionSize_;
}

void UnifiedCudaMemoryPool::ensurePowerFlowBuffers(size_t nBranches,
                                                    Real*& d_pFlow, size_t& pFlowSize,
                                                    Real*& d_qFlow, size_t& qFlowSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    cuda::ensureCapacity(d_pFlow_, pFlowSize_, nBranches);
    cuda::ensureCapacity(d_qFlow_, qFlowSize_, nBranches);
    
    d_pFlow = d_pFlow_;
    pFlowSize = pFlowSize_;
    d_qFlow = d_qFlow_;
    qFlowSize = qFlowSize_;
}

void UnifiedCudaMemoryPool::ensureMeasurementBuffer(size_t nMeasurements, Real*& d_hx, size_t& hxSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    cuda::ensureCapacity(d_hx_, hxSize_, nMeasurements);
    
    d_hx = d_hx_;
    hxSize = hxSize_;
}

void UnifiedCudaMemoryPool::ensurePartialBuffer(size_t requiredSize, Real*& d_partial, size_t& partialSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Partial buffer grows to maximum required size across all operations
    if (partialSize_ < requiredSize) {
        cuda::ensureCapacity(d_partial_, partialSize_, requiredSize);
    }
    
    d_partial = d_partial_;
    partialSize = partialSize_;
}

void UnifiedCudaMemoryPool::ensureDerivedQuantityBuffers(size_t nBranches,
                                                          Real*& d_pMW, size_t& pMWSize,
                                                          Real*& d_qMVAR, size_t& qMVARSize,
                                                          Real*& d_iPU, size_t& iPUSize,
                                                          Real*& d_iAmps, size_t& iAmpsSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    cuda::ensureCapacity(d_pMW_, pMWSize_, nBranches);
    cuda::ensureCapacity(d_qMVAR_, qMVARSize_, nBranches);
    cuda::ensureCapacity(d_iPU_, iPUSize_, nBranches);
    cuda::ensureCapacity(d_iAmps_, iAmpsSize_, nBranches);
    
    d_pMW = d_pMW_;
    pMWSize = pMWSize_;
    d_qMVAR = d_qMVAR_;
    qMVARSize = qMVARSize_;
    d_iPU = d_iPU_;
    iPUSize = iPUSize_;
    d_iAmps = d_iAmps_;
    iAmpsSize = iAmpsSize_;
}

void UnifiedCudaMemoryPool::ensureAdjacencyBuffers(size_t nBuses, size_t maxCSRSize,
                                                    Index*& d_branchFromBus, size_t& branchFromBusSize,
                                                    Index*& d_branchToBus, size_t& branchToBusSize,
                                                    Index*& d_branchFromBusRowPtr, size_t& branchFromBusRowPtrSize,
                                                    Index*& d_branchToBusRowPtr, size_t& branchToBusRowPtrSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    cuda::ensureCapacity(d_branchFromBus_, branchFromBusSize_, maxCSRSize);
    cuda::ensureCapacity(d_branchToBus_, branchToBusSize_, maxCSRSize);
    cuda::ensureCapacity(d_branchFromBusRowPtr_, branchFromBusRowPtrSize_, nBuses + 1);
    cuda::ensureCapacity(d_branchToBusRowPtr_, branchToBusRowPtrSize_, nBuses + 1);
    
    d_branchFromBus = d_branchFromBus_;
    branchFromBusSize = branchFromBusSize_;
    d_branchToBus = d_branchToBus_;
    branchToBusSize = branchToBusSize_;
    d_branchFromBusRowPtr = d_branchFromBusRowPtr_;
    branchFromBusRowPtrSize = branchFromBusRowPtrSize_;
    d_branchToBusRowPtr = d_branchToBusRowPtr_;
    branchToBusRowPtrSize = branchToBusRowPtrSize_;
}

void UnifiedCudaMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    freeAllBuffers();
}

UnifiedCudaMemoryPool::MemoryStats UnifiedCudaMemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    MemoryStats stats = {};
    size_t totalBytes = 0;
    
    // State buffers
    if (d_v_) totalBytes += vSize_ * sizeof(Real);
    if (d_theta_) totalBytes += thetaSize_ * sizeof(Real);
    stats.stateBuffersBytes = (vSize_ + thetaSize_) * sizeof(Real);
    
    // Network buffers
    if (d_buses_) totalBytes += busesSize_ * sizeof(DeviceBus);
    if (d_branches_) totalBytes += branchesSize_ * sizeof(DeviceBranch);
    stats.networkBuffersBytes = busesSize_ * sizeof(DeviceBus) + branchesSize_ * sizeof(DeviceBranch);
    
    // Power buffers
    if (d_pInjection_) totalBytes += pInjectionSize_ * sizeof(Real);
    if (d_qInjection_) totalBytes += qInjectionSize_ * sizeof(Real);
    if (d_pFlow_) totalBytes += pFlowSize_ * sizeof(Real);
    if (d_qFlow_) totalBytes += qFlowSize_ * sizeof(Real);
    if (d_pMW_) totalBytes += pMWSize_ * sizeof(Real);
    if (d_qMVAR_) totalBytes += qMVARSize_ * sizeof(Real);
    if (d_iPU_) totalBytes += iPUSize_ * sizeof(Real);
    if (d_iAmps_) totalBytes += iAmpsSize_ * sizeof(Real);
    stats.powerBuffersBytes = (pInjectionSize_ + qInjectionSize_ + pFlowSize_ + qFlowSize_ +
                               pMWSize_ + qMVARSize_ + iPUSize_ + iAmpsSize_) * sizeof(Real);
    
    // Measurement buffer
    if (d_hx_) totalBytes += hxSize_ * sizeof(Real);
    stats.measurementBuffersBytes = hxSize_ * sizeof(Real);
    
    // Reduction buffer
    if (d_partial_) totalBytes += partialSize_ * sizeof(Real);
    stats.reductionBuffersBytes = partialSize_ * sizeof(Real);
    
    // Adjacency buffers
    if (d_branchFromBus_) totalBytes += branchFromBusSize_ * sizeof(Index);
    if (d_branchToBus_) totalBytes += branchToBusSize_ * sizeof(Index);
    if (d_branchFromBusRowPtr_) totalBytes += branchFromBusRowPtrSize_ * sizeof(Index);
    if (d_branchToBusRowPtr_) totalBytes += branchToBusRowPtrSize_ * sizeof(Index);
    
    stats.totalBytes = totalBytes;
    stats.totalBuffers = (d_v_ ? 1 : 0) + (d_theta_ ? 1 : 0) + (d_buses_ ? 1 : 0) + 
                         (d_branches_ ? 1 : 0) + (d_pInjection_ ? 1 : 0) + (d_qInjection_ ? 1 : 0) +
                         (d_pFlow_ ? 1 : 0) + (d_qFlow_ ? 1 : 0) + (d_hx_ ? 1 : 0) + 
                         (d_partial_ ? 1 : 0) + (d_pMW_ ? 1 : 0) + (d_qMVAR_ ? 1 : 0) +
                         (d_iPU_ ? 1 : 0) + (d_iAmps_ ? 1 : 0) + (d_branchFromBus_ ? 1 : 0) +
                         (d_branchToBus_ ? 1 : 0) + (d_branchFromBusRowPtr_ ? 1 : 0) +
                         (d_branchToBusRowPtr_ ? 1 : 0);
    
    return stats;
}

void UnifiedCudaMemoryPool::freeAllBuffers() {
    cuda::freeBuffer(d_v_, vSize_);
    cuda::freeBuffer(d_theta_, thetaSize_);
    cuda::freeBuffer(d_buses_, busesSize_);
    cuda::freeBuffer(d_branches_, branchesSize_);
    cuda::freeBuffer(d_pInjection_, pInjectionSize_);
    cuda::freeBuffer(d_qInjection_, qInjectionSize_);
    cuda::freeBuffer(d_pFlow_, pFlowSize_);
    cuda::freeBuffer(d_qFlow_, qFlowSize_);
    cuda::freeBuffer(d_hx_, hxSize_);
    cuda::freeBuffer(d_partial_, partialSize_);
    cuda::freeBuffer(d_pMW_, pMWSize_);
    cuda::freeBuffer(d_qMVAR_, qMVARSize_);
    cuda::freeBuffer(d_iPU_, iPUSize_);
    cuda::freeBuffer(d_iAmps_, iAmpsSize_);
    cuda::freeBuffer(d_branchFromBus_, branchFromBusSize_);
    cuda::freeBuffer(d_branchToBus_, branchToBusSize_);
    cuda::freeBuffer(d_branchFromBusRowPtr_, branchFromBusRowPtrSize_);
    cuda::freeBuffer(d_branchToBusRowPtr_, branchToBusRowPtrSize_);
}

} // namespace cuda
} // namespace sle

