/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <sle/cuda/CudaDataManager.h>
#include <sle/cuda/CudaNetworkUtils.h>
#include <sle/cuda/CudaUtils.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <cassert>

namespace sle {
namespace model {

NetworkModel::NetworkModel()
    : deviceDataDirty_(true)
    , internalDataManager_(nullptr)
    , adjacencyDirty_(true)
    , baseMVA_(100.0)
    , referenceBus_(-1) {
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
    Index branchIdx = static_cast<Index>(branches_.size());
    branchIndexMap_[id] = branchIdx;
    branches_.push_back(std::move(branch));
    
    // Update fast lookup map for (fromBus, toBus) pairs
    branchBusPairMap_[std::make_pair(fromBus, toBus)] = branchIdx;
    
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

Branch* NetworkModel::getBranchByBuses(BusId fromBus, BusId toBus) {
    auto it = branchBusPairMap_.find(std::make_pair(fromBus, toBus));
    if (it != branchBusPairMap_.end()) {
        Index idx = it->second;
        if (static_cast<size_t>(idx) < branches_.size()) {
            return branches_[idx].get();
        }
    }
    return nullptr;
}

const Branch* NetworkModel::getBranchByBuses(BusId fromBus, BusId toBus) const {
    auto it = branchBusPairMap_.find(std::make_pair(fromBus, toBus));
    if (it != branchBusPairMap_.end()) {
        Index idx = it->second;
        if (static_cast<size_t>(idx) < branches_.size()) {
            return branches_[idx].get();
        }
    }
    return nullptr;
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
        Index idx = it->second;
        Branch* oldBranch = branches_[idx].get();
        
        // Update fast lookup map if bus IDs changed
        if (oldBranch) {
            auto oldPair = std::make_pair(oldBranch->getFromBus(), oldBranch->getToBus());
            branchBusPairMap_.erase(oldPair);
        }
        
        *branches_[idx] = branchData;  // Uses copy assignment operator
        
        // Update fast lookup map with new bus IDs
        Branch* newBranch = branches_[idx].get();
        if (newBranch) {
            auto newPair = std::make_pair(newBranch->getFromBus(), newBranch->getToBus());
            branchBusPairMap_[newPair] = idx;
        }
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
        
        // Remove from fast lookup map
        if (static_cast<size_t>(idx) < branches_.size()) {
            const Branch* branch = branches_[idx].get();
            if (branch) {
                branchBusPairMap_.erase(std::make_pair(branch->getFromBus(), branch->getToBus()));
            }
        }
        
        branches_.erase(branches_.begin() + idx);
        branchIndexMap_.erase(it);
        
        // Update indices for all branches after the removed one (indices shifted down by 1)
        // Also update fast lookup map for affected branches
        for (size_t i = idx; i < branches_.size(); ++i) {
            BranchId branchId = branches_[i]->getId();
            branchIndexMap_[branchId] = static_cast<Index>(i);
            
            // Update fast lookup map
            const Branch* branch = branches_[i].get();
            if (branch) {
                auto pairKey = std::make_pair(branch->getFromBus(), branch->getToBus());
                branchBusPairMap_[pairKey] = static_cast<Index>(i);
            }
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
    branchBusPairMap_.clear();  // Clear fast lookup map
    referenceBus_ = -1;
    // Clear cached vectors
    cachedPInjection_.clear();
    cachedQInjection_.clear();
    invalidateCaches();
}

void NetworkModel::invalidateCaches() {
    // Note: We don't clear cached vectors here - they're resized on-demand
    // This allows reuse across multiple computations without reallocation
    deviceDataDirty_ = true;
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
            if (branch->isOn()) { // Only include active branches
            branchesFromBus_[fromIdx].push_back(static_cast<Index>(i));
            }
        }
        if (toIdx >= 0 && static_cast<size_t>(toIdx) < nBuses) {
            if (branch->isOn()) { // Only include active branches
            branchesToBus_[toIdx].push_back(static_cast<Index>(i));
            }
        }
    }
    
    adjacencyDirty_ = false;
}

sle::cuda::CudaDataManager* NetworkModel::getInternalDataManager() const {
    if (!internalDataManager_) {
        internalDataManager_ = std::make_shared<sle::cuda::CudaDataManager>();
    }
    return internalDataManager_.get();
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
    sle::cuda::buildDeviceBuses(*this, cachedDeviceBuses_);
    sle::cuda::buildDeviceBranches(*this, cachedDeviceBranches_);
    
    // Fallback: Use CPU version (for backward compatibility or when GPU unavailable)
    sle::cuda::buildCSRAdjacencyLists(*this,
                                      cachedBranchFromBus_,
                                      cachedBranchFromBusRowPtr_,
                                      cachedBranchToBus_,
                                      cachedBranchToBusRowPtr_);
    
    deviceDataDirty_ = false;
}

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
    // Only grow vectors, never shrink, to avoid reallocations
    if (cachedPInjection_.size() < nBuses) {
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
    
    // Determine which data manager to use
    sle::cuda::CudaDataManager* activeManager = dataManager;
    if (!activeManager) {
        activeManager = getInternalDataManager();
    }
    
    try {
        // Initialize if needed (allocates GPU memory)
        if (!activeManager->isInitialized()) {
            activeManager->initialize(static_cast<Index>(nBuses), 
                                     static_cast<Index>(nBranches), 
                                     0); // No measurements needed for power injections
        }
        
            // Update state in shared data manager
            const auto& v = state.getMagnitudes();
            const auto& theta = state.getAngles();
        activeManager->updateState(v.data(), theta.data(), static_cast<Index>(nBuses));
            
            // Update network data if needed (builds CSR format)
            updateDeviceData();
            
            // Update network data in shared data manager
        activeManager->updateNetwork(
                cachedDeviceBuses_.data(),
                cachedDeviceBranches_.data(),
                static_cast<Index>(nBuses),
                static_cast<Index>(nBranches));
            
            // Update adjacency lists in shared data manager
        activeManager->updateAdjacency(
                cachedBranchFromBus_.data(),
                cachedBranchFromBusRowPtr_.data(),
                cachedBranchToBus_.data(),
                cachedBranchToBusRowPtr_.data(),
                static_cast<Index>(nBuses),
                static_cast<Index>(cachedBranchFromBus_.size()),
                static_cast<Index>(cachedBranchToBus_.size()));
            
            // Launch GPU kernel using shared data manager pointers
            sle::cuda::computeAllPowerInjectionsGPU(
            activeManager->getStateV(),
            activeManager->getStateTheta(),
            activeManager->getBuses(),
            activeManager->getBranches(),
            activeManager->getBranchFromBus(),
            activeManager->getBranchFromBusRowPtr(),
            activeManager->getBranchToBus(),
            activeManager->getBranchToBusRowPtr(),
            activeManager->getPInjection(),
            activeManager->getQInjection(),
                static_cast<Index>(nBuses),
                static_cast<Index>(nBranches));
            
            // Copy back results
        cudaMemcpy(pInjection.data(), activeManager->getPInjection(), 
                  nBuses * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(qInjection.data(), activeManager->getQInjection(), 
                  nBuses * sizeof(Real), cudaMemcpyDeviceToHost);
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("CUDA power injection computation failed: ") + e.what());
    }
}


} // namespace model
} // namespace sle
