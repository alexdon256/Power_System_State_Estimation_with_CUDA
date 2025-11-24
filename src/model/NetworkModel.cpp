/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <algorithm>
#include <cmath>

#ifdef USE_CUDA
#include <sle/cuda/CudaPowerFlow.h>
#include <sle/cuda/CudaDataManager.h>
#include <sle/cuda/CudaNetworkUtils.h>
#include <sle/cuda/CudaUtils.h>
#include <sle/cuda/UnifiedCudaMemoryPool.h>
#include <cuda_runtime.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <cassert>

namespace sle {
namespace model {

#ifdef USE_CUDA
// NetworkModel now uses UnifiedCudaMemoryPool instead of its own pool
// This eliminates duplicate allocations and improves memory utilization
#endif

NetworkModel::NetworkModel()
#ifdef USE_CUDA
    : useUnifiedPool_(true)
    , deviceDataDirty_(true)
    , adjacencyDirty_(true)
#else
    : adjacencyDirty_(true)
#endif
    , baseMVA_(100.0)
    , referenceBus_(-1) {
    // Initialization order: CUDA members (if enabled) -> adjacencyDirty_ -> baseMVA_ -> referenceBus_
    // NetworkModel now uses UnifiedCudaMemoryPool by default (eliminates duplicate allocations)
}

NetworkModel::~NetworkModel() = default;

Bus* NetworkModel::addBus(BusId id, const std::string& name) {
    auto it = busIndexMap_.find(id);
    if (it != busIndexMap_.end()) {
        return buses_[it->second].get();
    }
    
    auto bus = std::make_unique<Bus>(id, name);
    Bus* busPtr = bus.get();
    Index idx = static_cast<Index>(buses_.size());
    busIndexMap_[id] = idx;
    if (!name.empty()) {
        busNameMap_[name] = idx;  // Add to name index for O(1) lookup
    }
    buses_.push_back(std::move(bus));
    invalidateCaches();
    return busPtr;
}

Bus* NetworkModel::getBus(BusId id) {
    auto it = busIndexMap_.find(id);
    if (it != busIndexMap_.end()) {
        return buses_[it->second].get();
    }
    return nullptr;
}

const Bus* NetworkModel::getBus(BusId id) const {
    auto it = busIndexMap_.find(id);
    if (it != busIndexMap_.end()) {
        return buses_[it->second].get();
    }
    return nullptr;
}

Bus* NetworkModel::getBusByName(const std::string& name) {
    auto it = busNameMap_.find(name);
    if (it != busNameMap_.end() && static_cast<size_t>(it->second) < buses_.size()) {
        return buses_[it->second].get();
    }
    return nullptr;
}

const Bus* NetworkModel::getBusByName(const std::string& name) const {
    auto it = busNameMap_.find(name);
    if (it != busNameMap_.end() && static_cast<size_t>(it->second) < buses_.size()) {
        return buses_[it->second].get();
    }
    return nullptr;
}

std::vector<Bus*> NetworkModel::getBuses() {
    std::vector<Bus*> result;
    result.reserve(buses_.size());
    for (auto& bus : buses_) {
        result.push_back(bus.get());
    }
    return result;
}

std::vector<const Bus*> NetworkModel::getBuses() const {
    std::vector<const Bus*> result;
    result.reserve(buses_.size());
    for (const auto& bus : buses_) {
        result.push_back(bus.get());
    }
    return result;
}

Branch* NetworkModel::addBranch(BranchId id, BusId fromBus, BusId toBus) {
    // Validate bus indices exist in network
    Index fromBusIdx = getBusIndex(fromBus);
    Index toBusIdx = getBusIndex(toBus);
    if (fromBusIdx < 0 || toBusIdx < 0) {
        #ifndef NDEBUG
        assert(fromBusIdx >= 0 && toBusIdx >= 0 && "Branch buses must exist in network");
        #endif
        // In release builds, return nullptr to indicate failure
        return nullptr;
    }
    
    // Prevent self-loops (branch from bus to itself)
    if (fromBus == toBus) {
        #ifndef NDEBUG
        assert(fromBus != toBus && "Branch cannot connect bus to itself");
        #endif
        return nullptr;
    }
    
    auto it = branchIndexMap_.find(id);
    if (it != branchIndexMap_.end()) {
        return branches_[it->second].get();
    }
    
    auto branch = std::make_unique<Branch>(id, fromBus, toBus);
    Branch* branchPtr = branch.get();
    branchIndexMap_[id] = branches_.size();
    branches_.push_back(std::move(branch));
    invalidateCaches();
    return branchPtr;
}

Branch* NetworkModel::getBranch(BranchId id) {
    auto it = branchIndexMap_.find(id);
    if (it != branchIndexMap_.end()) {
        return branches_[it->second].get();
    }
    return nullptr;
}

const Branch* NetworkModel::getBranch(BranchId id) const {
    auto it = branchIndexMap_.find(id);
    if (it != branchIndexMap_.end()) {
        return branches_[it->second].get();
    }
    return nullptr;
}

std::vector<Branch*> NetworkModel::getBranches() {
    std::vector<Branch*> result;
    result.reserve(branches_.size());
    for (auto& branch : branches_) {
        result.push_back(branch.get());
    }
    return result;
}

std::vector<const Branch*> NetworkModel::getBranches() const {
    std::vector<const Branch*> result;
    result.reserve(branches_.size());
    for (const auto& branch : branches_) {
        result.push_back(branch.get());
    }
    return result;
}

std::vector<Branch*> NetworkModel::getBranchesFromBus(BusId busId) {
    updateAdjacencyLists();
    Index busIdx = getBusIndex(busId);
    if (busIdx < 0 || static_cast<size_t>(busIdx) >= branchesFromBus_.size()) {
        return {};
    }
    
    std::vector<Branch*> result;
    result.reserve(branchesFromBus_[busIdx].size());
    for (Index brIdx : branchesFromBus_[busIdx]) {
        if (static_cast<size_t>(brIdx) < branches_.size()) {
            result.push_back(branches_[brIdx].get());
        }
    }
    return result;
}

std::vector<const Branch*> NetworkModel::getBranchesFromBus(BusId busId) const {
    const_cast<NetworkModel*>(this)->updateAdjacencyLists();
    Index busIdx = getBusIndex(busId);
    if (busIdx < 0 || static_cast<size_t>(busIdx) >= branchesFromBus_.size()) {
        return {};
    }
    
    std::vector<const Branch*> result;
    result.reserve(branchesFromBus_[busIdx].size());
    for (Index brIdx : branchesFromBus_[busIdx]) {
        if (static_cast<size_t>(brIdx) < branches_.size()) {
            result.push_back(branches_[brIdx].get());
        }
    }
    return result;
}

std::vector<Branch*> NetworkModel::getBranchesToBus(BusId busId) {
    updateAdjacencyLists();
    Index busIdx = getBusIndex(busId);
    if (busIdx < 0 || static_cast<size_t>(busIdx) >= branchesToBus_.size()) {
        return {};
    }
    
    std::vector<Branch*> result;
    result.reserve(branchesToBus_[busIdx].size());
    for (Index brIdx : branchesToBus_[busIdx]) {
        if (static_cast<size_t>(brIdx) < branches_.size()) {
            result.push_back(branches_[brIdx].get());
        }
    }
    return result;
}

std::vector<const Branch*> NetworkModel::getBranchesToBus(BusId busId) const {
    const_cast<NetworkModel*>(this)->updateAdjacencyLists();
    Index busIdx = getBusIndex(busId);
    if (busIdx < 0 || static_cast<size_t>(busIdx) >= branchesToBus_.size()) {
        return {};
    }
    
    std::vector<const Branch*> result;
    result.reserve(branchesToBus_[busIdx].size());
    for (Index brIdx : branchesToBus_[busIdx]) {
        if (static_cast<size_t>(brIdx) < branches_.size()) {
            result.push_back(branches_[brIdx].get());
        }
    }
    return result;
}

void NetworkModel::setReferenceBus(BusId busId) {
    referenceBus_ = busId;
    Bus* bus = getBus(busId);
    if (bus) {
        bus->setType(BusType::Slack);
    }
}

Index NetworkModel::getBusIndex(BusId id) const {
    auto it = busIndexMap_.find(id);
    return (it != busIndexMap_.end()) ? it->second : -1;
}

Index NetworkModel::getBranchIndex(BranchId id) const {
    auto it = branchIndexMap_.find(id);
    return (it != branchIndexMap_.end()) ? it->second : -1;
}

void NetworkModel::removeBus(BusId id) {
    auto it = busIndexMap_.find(id);
    if (it != busIndexMap_.end()) {
        Index idx = it->second;
        buses_.erase(buses_.begin() + idx);
        busIndexMap_.erase(it);
        
        // Update indices for all buses after the removed one (indices shifted down by 1)
        for (size_t i = idx; i < buses_.size(); ++i) {
            BusId busId = buses_[i]->getId();
            busIndexMap_[busId] = static_cast<Index>(i);
        }
        
        // Update name index if bus had a name
        for (auto itName = busNameMap_.begin(); itName != busNameMap_.end();) {
            if (itName->second == idx) {
                itName = busNameMap_.erase(itName);
            } else if (itName->second > idx) {
                --itName->second;  // Adjust index for buses after removed one
                ++itName;
            } else {
                ++itName;
            }
        }
        
        invalidateCaches();
    }
}

void NetworkModel::updateBus(BusId id, const Bus& busData) {
    // Direct lookup
    auto it = busIndexMap_.find(id);
    if (it != busIndexMap_.end() && static_cast<size_t>(it->second) < buses_.size()) {
        *buses_[it->second] = busData;  // Uses copy assignment operator
    }
}

void NetworkModel::updateBranch(BranchId id, const Branch& branchData) {
    // Direct lookup
    auto it = branchIndexMap_.find(id);
    if (it != branchIndexMap_.end() && static_cast<size_t>(it->second) < branches_.size()) {
        *branches_[it->second] = branchData;  // Uses copy assignment operator
    }
}

void NetworkModel::removeBus(BusId id) {
    auto it = busIndexMap_.find(id);
    if (it != busIndexMap_.end()) {
        Index idx = it->second;
        
        // Remove from name map if bus has a name
        if (idx >= 0 && static_cast<size_t>(idx) < buses_.size()) {
            const std::string& name = buses_[idx]->getName();
            if (!name.empty()) {
                busNameMap_.erase(name);
            }
        }
        
        buses_.erase(buses_.begin() + idx);
        busIndexMap_.erase(it);
        
        // Update indices for all buses after the removed one (indices shifted down by 1)
        for (size_t i = idx; i < buses_.size(); ++i) {
            BusId busId = buses_[i]->getId();
            busIndexMap_[busId] = static_cast<Index>(i);
            
            // Update name map if bus has a name
            const std::string& name = buses_[i]->getName();
            if (!name.empty()) {
                busNameMap_[name] = static_cast<Index>(i);
            }
        }
        
        invalidateCaches();
    }
}

void NetworkModel::removeBranch(BranchId id) {
    auto it = branchIndexMap_.find(id);
    if (it != branchIndexMap_.end()) {
        Index idx = it->second;
        branches_.erase(branches_.begin() + idx);
        branchIndexMap_.erase(it);
        
        // Update indices for all branches after the removed one (indices shifted down by 1)
        for (size_t i = idx; i < branches_.size(); ++i) {
            BranchId branchId = branches_[i]->getId();
            branchIndexMap_[branchId] = static_cast<Index>(i);
        }
        
        invalidateCaches();
    }
}

void NetworkModel::clear() {
    buses_.clear();
    branches_.clear();
    busIndexMap_.clear();
    branchIndexMap_.clear();
    busNameMap_.clear();  // Clear name index
    referenceBus_ = -1;
    // Clear cached vectors
    cachedPInjection_.clear();
    cachedQInjection_.clear();
    invalidateCaches();
}

void NetworkModel::invalidateCaches() {
    // Note: We don't clear cached vectors here - they're resized on-demand
    // This allows reuse across multiple computations without reallocation
#ifdef USE_CUDA
    deviceDataDirty_ = true;
#endif
    adjacencyDirty_ = true;
}

void NetworkModel::updateAdjacencyLists() const {
    if (!adjacencyDirty_) return;
    
    size_t nBuses = buses_.size();
    
    // Only resize if size changed (avoid unnecessary reallocation)
    if (branchesFromBus_.size() != nBuses) {
        branchesFromBus_.clear();
        branchesFromBus_.resize(nBuses);
    } else {
        // Clear existing entries but keep capacity
        for (auto& vec : branchesFromBus_) {
            vec.clear();
        }
    }
    
    if (branchesToBus_.size() != nBuses) {
        branchesToBus_.clear();
        branchesToBus_.resize(nBuses);
    } else {
        // Clear existing entries but keep capacity
        for (auto& vec : branchesToBus_) {
            vec.clear();
        }
    }
    
    for (size_t i = 0; i < branches_.size(); ++i) {
        const Branch* branch = branches_[i].get();
        Index fromIdx = getBusIndex(branch->getFromBus());
        Index toIdx = getBusIndex(branch->getToBus());
        
        if (fromIdx >= 0 && static_cast<size_t>(fromIdx) < nBuses) {
            branchesFromBus_[fromIdx].push_back(static_cast<Index>(i));
        }
        if (toIdx >= 0 && static_cast<size_t>(toIdx) < nBuses) {
            branchesToBus_[toIdx].push_back(static_cast<Index>(i));
        }
    }
    
    adjacencyDirty_ = false;
}

#ifdef USE_CUDA
void NetworkModel::ensureGPUCapacity(size_t nBuses, size_t nBranches) const {
    if (useUnifiedPool_) {
        // Use unified memory pool (eliminates duplicate allocations)
        auto& unifiedPool = cuda::UnifiedCudaMemoryPool::getInstance();
        
        // Get shared buffers from unified pool
        Real* d_v = nullptr;
        Real* d_theta = nullptr;
        size_t vSize = 0, thetaSize = 0;
        unifiedPool.ensureStateBuffers(nBuses, d_v, vSize, d_theta, thetaSize);
        
        sle::cuda::DeviceBus* d_buses = nullptr;
        sle::cuda::DeviceBranch* d_branches = nullptr;
        size_t busesSize = 0, branchesSize = 0;
        unifiedPool.ensureNetworkBuffers(nBuses, nBranches, d_buses, busesSize, d_branches, branchesSize);
        
        Real* d_pInjection = nullptr;
        Real* d_qInjection = nullptr;
        size_t pInjectionSize = 0, qInjectionSize = 0;
        unifiedPool.ensurePowerInjectionBuffers(nBuses, d_pInjection, pInjectionSize, d_qInjection, qInjectionSize);
        
        Real* d_pFlow = nullptr;
        Real* d_qFlow = nullptr;
        size_t pFlowSize = 0, qFlowSize = 0;
        unifiedPool.ensurePowerFlowBuffers(nBranches, d_pFlow, pFlowSize, d_qFlow, qFlowSize);
        
        Real* d_pMW = nullptr;
        Real* d_qMVAR = nullptr;
        Real* d_iPU = nullptr;
        Real* d_iAmps = nullptr;
        size_t pMWSize = 0, qMVARSize = 0, iPUSize = 0, iAmpsSize = 0;
        unifiedPool.ensureDerivedQuantityBuffers(nBranches, d_pMW, pMWSize, d_qMVAR, qMVARSize, d_iPU, iPUSize, d_iAmps, iAmpsSize);
        
        // CSR adjacency buffers (size varies, managed separately)
        size_t maxCSRSize = nBranches;
        Index* d_branchFromBus = nullptr;
        Index* d_branchToBus = nullptr;
        Index* d_branchFromBusRowPtr = nullptr;
        Index* d_branchToBusRowPtr = nullptr;
        size_t branchFromBusSize = 0, branchToBusSize = 0;
        size_t branchFromBusRowPtrSize = 0, branchToBusRowPtrSize = 0;
        unifiedPool.ensureAdjacencyBuffers(nBuses, maxCSRSize,
                                           d_branchFromBus, branchFromBusSize,
                                           d_branchToBus, branchToBusSize,
                                           d_branchFromBusRowPtr, branchFromBusRowPtrSize,
                                           d_branchToBusRowPtr, branchToBusRowPtrSize);
    } else {
        // Legacy mode: allocate local buffers (for backward compatibility)
        // This path is deprecated and should not be used in new code
        // Kept for backward compatibility only
    }
}

void NetworkModel::updateDeviceData() const {
    if (!deviceDataDirty_) return;
    
    // Ensure adjacency lists are up to date (needed for CSR format)
    updateAdjacencyLists();
    
    size_t nBuses = buses_.size();
    size_t nBranches = branches_.size();
    
    // Handle empty network case
    if (nBuses == 0) {
        cachedDeviceBuses_.clear();
        cachedDeviceBranches_.clear();
        cachedBranchFromBus_.clear();
        cachedBranchToBus_.clear();
        cachedBranchFromBusRowPtr_.clear();
        cachedBranchToBusRowPtr_.clear();
        deviceDataDirty_ = false;
        return;
    }
    
    // OPTIMIZATION: Use CudaNetworkUtils to build device data (eliminates duplicate code)
    // This matches the approach used in MeasurementFunctions and JacobianMatrix
    sle::cuda::buildDeviceBuses(*this, cachedDeviceBuses_);
    sle::cuda::buildDeviceBranches(*this, cachedDeviceBranches_);
    
    // OPTIMIZATION: Try GPU-accelerated CSR building first (eliminates CPU-GPU transfer)
    if (useUnifiedPool_) {
        auto& unifiedPool = cuda::UnifiedCudaMemoryPool::getInstance();
        
        // Get DeviceBranch buffer from unified pool
        sle::cuda::DeviceBranch* d_branches = nullptr;
        size_t branchesSize = 0;
        sle::cuda::DeviceBus* d_buses_temp = nullptr;
        size_t busesSize = 0;
        unifiedPool.ensureNetworkBuffers(nBuses, nBranches, d_buses_temp, busesSize, d_branches, branchesSize);
        
        if (d_branches) {
            // Copy DeviceBranch data to GPU (one-time transfer)
            cudaMemcpy(d_branches, cachedDeviceBranches_.data(), 
                      nBranches * sizeof(sle::cuda::DeviceBranch), 
                      cudaMemcpyHostToDevice);
            
            // Get CSR buffers from unified pool
            size_t maxCSRSize = nBranches;
            Index* d_branchFromBus = nullptr;
            Index* d_branchToBus = nullptr;
            Index* d_branchFromBusRowPtr = nullptr;
            Index* d_branchToBusRowPtr = nullptr;
            size_t branchFromBusSize = 0, branchToBusSize = 0;
            size_t branchFromBusRowPtrSize = 0, branchToBusRowPtrSize = 0;
            unifiedPool.ensureAdjacencyBuffers(nBuses, maxCSRSize,
                                               d_branchFromBus, branchFromBusSize,
                                               d_branchToBus, branchToBusSize,
                                               d_branchFromBusRowPtr, branchFromBusRowPtrSize,
                                               d_branchToBusRowPtr, branchToBusRowPtrSize);
            
            // Build CSR on GPU
            if (sle::cuda::buildCSRAdjacencyListsGPU(
                    d_branches, d_branchFromBus, d_branchToBus,
                    d_branchFromBusRowPtr, d_branchToBusRowPtr,
                    static_cast<Index>(nBranches), static_cast<Index>(nBuses), nullptr)) {
                
                // Copy results back to host (for compatibility with existing code)
                // Resize host vectors to match GPU results
                Index nnz_from = 0, nnz_to = 0;
                if (nBuses > 0) {
                    Index host_rowPtr[nBuses + 1];
                    cudaMemcpy(host_rowPtr, d_branchFromBusRowPtr, (nBuses + 1) * sizeof(Index), cudaMemcpyDeviceToHost);
                    nnz_from = host_rowPtr[nBuses];
                    
                    cudaMemcpy(host_rowPtr, d_branchToBusRowPtr, (nBuses + 1) * sizeof(Index), cudaMemcpyDeviceToHost);
                    nnz_to = host_rowPtr[nBuses];
                }
                
                cachedBranchFromBus_.resize(nnz_from);
                cachedBranchToBus_.resize(nnz_to);
                cachedBranchFromBusRowPtr_.resize(nBuses + 1);
                cachedBranchToBusRowPtr_.resize(nBuses + 1);
                
                if (nnz_from > 0) {
                    cudaMemcpy(cachedBranchFromBus_.data(), d_branchFromBus, 
                              nnz_from * sizeof(Index), cudaMemcpyDeviceToHost);
                }
                if (nnz_to > 0) {
                    cudaMemcpy(cachedBranchToBus_.data(), d_branchToBus, 
                              nnz_to * sizeof(Index), cudaMemcpyDeviceToHost);
                }
                cudaMemcpy(cachedBranchFromBusRowPtr_.data(), d_branchFromBusRowPtr, 
                          (nBuses + 1) * sizeof(Index), cudaMemcpyDeviceToHost);
                cudaMemcpy(cachedBranchToBusRowPtr_.data(), d_branchToBusRowPtr, 
                          (nBuses + 1) * sizeof(Index), cudaMemcpyDeviceToHost);
                
                deviceDataDirty_ = false;
                return;  // Success using GPU
            }
            // If GPU build failed, fall through to CPU version
        }
    }
    
    // Fallback: Use CPU version (for backward compatibility or when GPU unavailable)
    sle::cuda::buildCSRAdjacencyLists(*this,
                                      cachedBranchFromBus_,
                                      cachedBranchFromBusRowPtr_,
                                      cachedBranchToBus_,
                                      cachedBranchToBusRowPtr_);
    
    deviceDataDirty_ = false;
}
#endif

void NetworkModel::computePowerInjections(const StateVector& state) {
    size_t nBuses = buses_.size();
    
    // Validate state vector size matches network size
    if (state.size() != nBuses) {
        #ifndef NDEBUG
        assert(state.size() == nBuses && "StateVector size must match network bus count");
        #endif
        return;
    }
    
    // Resize cached vectors only if network size changed
    if (cachedPInjection_.size() != nBuses) {
        cachedPInjection_.resize(nBuses);
        cachedQInjection_.resize(nBuses);
    }
    
    // Compute power injections (reuse cached vectors)
    computePowerInjections(state, cachedPInjection_, cachedQInjection_);
    
    // Store in Bus objects
    // OPTIMIZATION: OpenMP parallelization for independent loop
    Real baseMVA = getBaseMVA();
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < nBuses && i < cachedPInjection_.size(); ++i) {
        Bus* bus = buses_[i].get();
        Real pMW = cachedPInjection_[i] * baseMVA;
        Real qMVAR = cachedQInjection_[i] * baseMVA;
        bus->setPowerInjections(cachedPInjection_[i], cachedQInjection_[i], pMW, qMVAR);
    }
}

void NetworkModel::computePowerInjections(const StateVector& state,
                                          std::vector<Real>& pInjection, 
                                          std::vector<Real>& qInjection,
                                          sle::cuda::CudaDataManager* dataManager) const {
    // CUDA-EXCLUSIVE: All computations on GPU
    // OPTIMIZED: Use shared CudaDataManager if provided (reuses GPU data), otherwise use local pool
    size_t nBuses = buses_.size();
    size_t nBranches = branches_.size();
    
    // Handle empty network
    if (nBuses == 0) {
        pInjection.clear();
        qInjection.clear();
        return;
    }
    
    // Resize output vectors if needed (reuse capacity if already allocated)
    if (pInjection.size() != nBuses) {
        pInjection.resize(nBuses);
    }
    if (qInjection.size() != nBuses) {
        qInjection.resize(nBuses);
    }
    
    // Zero-initialize (faster than assign for already-sized vectors)
    std::fill(pInjection.begin(), pInjection.end(), 0.0);
    std::fill(qInjection.begin(), qInjection.end(), 0.0);
    
#ifdef USE_CUDA
    // OPTIMIZATION: Use shared CudaDataManager if provided (eliminates duplicate allocations)
    if (dataManager && dataManager->isInitialized()) {
        // Use shared data manager - reuses GPU data already allocated
        try {
            // Update state in shared data manager
            const auto& v = state.getMagnitudes();
            const auto& theta = state.getAngles();
            dataManager->updateState(v.data(), theta.data(), static_cast<Index>(nBuses));
            
            // Update network data if needed (builds CSR format)
            updateDeviceData();
            
            // Update network data in shared data manager
            dataManager->updateNetwork(
                cachedDeviceBuses_.data(),
                cachedDeviceBranches_.data(),
                static_cast<Index>(nBuses),
                static_cast<Index>(nBranches));
            
            // Update adjacency lists in shared data manager
            dataManager->updateAdjacency(
                cachedBranchFromBus_.data(),
                cachedBranchFromBusRowPtr_.data(),
                cachedBranchToBus_.data(),
                cachedBranchToBusRowPtr_.data(),
                static_cast<Index>(nBuses),
                static_cast<Index>(cachedBranchFromBus_.size()),
                static_cast<Index>(cachedBranchToBus_.size()));
            
            // Launch GPU kernel using shared data manager pointers
            sle::cuda::computeAllPowerInjectionsGPU(
                dataManager->getStateV(),
                dataManager->getStateTheta(),
                dataManager->getBuses(),
                dataManager->getBranches(),
                dataManager->getBranchFromBus(),
                dataManager->getBranchFromBusRowPtr(),
                dataManager->getBranchToBus(),
                dataManager->getBranchToBusRowPtr(),
                dataManager->getPInjection(),
                dataManager->getQInjection(),
                static_cast<Index>(nBuses),
                static_cast<Index>(nBranches));
            
            // Copy back results
            cudaMemcpy(pInjection.data(), dataManager->getPInjection(), 
                      nBuses * sizeof(Real), cudaMemcpyDeviceToHost);
            cudaMemcpy(qInjection.data(), dataManager->getQInjection(), 
                      nBuses * sizeof(Real), cudaMemcpyDeviceToHost);
            
            return;  // Success using shared data manager
        } catch (const std::exception& e) {
            // Fall through to local pool if shared manager fails
            // (for backward compatibility)
        }
    }
    
    // Fallback: Use unified memory pool (when dataManager not provided)
    // CUDA-EXCLUSIVE: GPU required
    try {
        // Ensure unified memory pool has capacity
        ensureGPUCapacity(nBuses, nBranches);
        
        // Get buffers from unified pool
        auto& unifiedPool = cuda::UnifiedCudaMemoryPool::getInstance();
        
        Real* d_v = nullptr;
        Real* d_theta = nullptr;
        size_t vSize = 0, thetaSize = 0;
        unifiedPool.ensureStateBuffers(nBuses, d_v, vSize, d_theta, thetaSize);
        
        sle::cuda::DeviceBus* d_buses = nullptr;
        sle::cuda::DeviceBranch* d_branches = nullptr;
        size_t busesSize = 0, branchesSize = 0;
        unifiedPool.ensureNetworkBuffers(nBuses, nBranches, d_buses, busesSize, d_branches, branchesSize);
        
        Real* d_pInjection = nullptr;
        Real* d_qInjection = nullptr;
        size_t pInjectionSize = 0, qInjectionSize = 0;
        unifiedPool.ensurePowerInjectionBuffers(nBuses, d_pInjection, pInjectionSize, d_qInjection, qInjectionSize);
        
        size_t maxCSRSize = nBranches;
        Index* d_branchFromBus = nullptr;
        Index* d_branchToBus = nullptr;
        Index* d_branchFromBusRowPtr = nullptr;
        Index* d_branchToBusRowPtr = nullptr;
        size_t branchFromBusSize = 0, branchToBusSize = 0;
        size_t branchFromBusRowPtrSize = 0, branchToBusRowPtrSize = 0;
        unifiedPool.ensureAdjacencyBuffers(nBuses, maxCSRSize,
                                           d_branchFromBus, branchFromBusSize,
                                           d_branchToBus, branchToBusSize,
                                           d_branchFromBusRowPtr, branchFromBusRowPtrSize,
                                           d_branchToBusRowPtr, branchToBusRowPtrSize);
        
        // Check if allocations succeeded
        if (!d_v || !d_theta || !d_buses || !d_branches ||
            !d_branchFromBus || !d_branchToBus ||
            !d_branchFromBusRowPtr || !d_branchToBusRowPtr ||
            !d_pInjection || !d_qInjection) {
            throw std::runtime_error("CUDA memory allocation failed for power injections");
        }
        
        // Update cached device data if needed (builds CSR format)
        updateDeviceData();
        
        // Get state vectors
        std::vector<Real> v = state.getMagnitudes();
        std::vector<Real> theta = state.getAngles();
        
        // Copy to device (reuse existing allocations from unified pool)
        cudaMemcpy(d_v, v.data(), nBuses * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_theta, theta.data(), nBuses * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_buses, cachedDeviceBuses_.data(), 
                  nBuses * sizeof(sle::cuda::DeviceBus), cudaMemcpyHostToDevice);
        cudaMemcpy(d_branches, cachedDeviceBranches_.data(), 
                  nBranches * sizeof(sle::cuda::DeviceBranch), cudaMemcpyHostToDevice);
        
        // Copy CSR format: column indices and row pointers
        size_t fromCSRSize = cachedBranchFromBus_.size();
        size_t toCSRSize = cachedBranchToBus_.size();
        
        // Validate CSR sizes match allocated memory
        if (fromCSRSize > branchFromBusSize || toCSRSize > branchToBusSize) {
            throw std::runtime_error("CUDA CSR buffer size mismatch for power injections");
        }
        
        // Safe to copy - sizes are valid
        if (fromCSRSize > 0) {
            cudaMemcpy(d_branchFromBus, cachedBranchFromBus_.data(), 
                      fromCSRSize * sizeof(Index), cudaMemcpyHostToDevice);
        }
        if (toCSRSize > 0) {
            cudaMemcpy(d_branchToBus, cachedBranchToBus_.data(), 
                      toCSRSize * sizeof(Index), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_branchFromBusRowPtr, cachedBranchFromBusRowPtr_.data(), 
                  (nBuses + 1) * sizeof(Index), cudaMemcpyHostToDevice);
        cudaMemcpy(d_branchToBusRowPtr, cachedBranchToBusRowPtr_.data(), 
                  (nBuses + 1) * sizeof(Index), cudaMemcpyHostToDevice);
        
        // Launch GPU kernel with CSR format (O(avg_degree) complexity)
        sle::cuda::computeAllPowerInjectionsGPU(
            d_v, d_theta, d_buses, d_branches,
            d_branchFromBus, d_branchFromBusRowPtr,
            d_branchToBus, d_branchToBusRowPtr,
            d_pInjection, d_qInjection,
            static_cast<Index>(nBuses), static_cast<Index>(nBranches));
        
        // Copy back (async could be used here for better performance)
        cudaMemcpy(pInjection.data(), d_pInjection, 
                  nBuses * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(qInjection.data(), d_qInjection, 
                  nBuses * sizeof(Real), cudaMemcpyDeviceToHost);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("CUDA power injection computation failed: ") + e.what());
    }
#else
    throw std::runtime_error("CUDA is required for NetworkModel::computePowerInjections()");
#endif
}


} // namespace model
} // namespace sle

