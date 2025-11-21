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
#include <sle/cuda/CudaMemoryManager.h>
#include <cuda_runtime.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace sle {
namespace model {

#ifdef USE_CUDA
// GPU memory pool structure
struct NetworkModel::CudaMemoryPool {
    Real* d_v = nullptr;
    Real* d_theta = nullptr;
    sle::cuda::DeviceBus* d_buses = nullptr;
    sle::cuda::DeviceBranch* d_branches = nullptr;
    Index* d_branchFromBus = nullptr;
    Index* d_branchToBus = nullptr;
    Real* d_pInjection = nullptr;
    Real* d_qInjection = nullptr;
    Real* d_pFlow = nullptr;
    Real* d_qFlow = nullptr;
    
    size_t vSize = 0;
    size_t thetaSize = 0;
    size_t busesSize = 0;
    size_t branchesSize = 0;
    size_t branchFromBusSize = 0;
    size_t branchToBusSize = 0;
    size_t pInjectionSize = 0;
    size_t qInjectionSize = 0;
    size_t pFlowSize = 0;
    size_t qFlowSize = 0;
    
    ~CudaMemoryPool() {
        if (d_v) cudaFree(d_v);
        if (d_theta) cudaFree(d_theta);
        if (d_buses) cudaFree(d_buses);
        if (d_branches) cudaFree(d_branches);
        if (d_branchFromBus) cudaFree(d_branchFromBus);
        if (d_branchToBus) cudaFree(d_branchToBus);
        if (d_pInjection) cudaFree(d_pInjection);
        if (d_qInjection) cudaFree(d_qInjection);
        if (d_pFlow) cudaFree(d_pFlow);
        if (d_qFlow) cudaFree(d_qFlow);
    }
};
#endif

NetworkModel::NetworkModel() 
    : baseMVA_(100.0)
#ifdef USE_CUDA
    , gpuMemoryPool_(std::make_unique<CudaMemoryPool>())
    , deviceDataDirty_(true)
#endif
    , adjacencyDirty_(true)
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
    busIndexMap_[id] = buses_.size();
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

void NetworkModel::buildAdmittanceMatrix(std::vector<Complex>& Y, 
                                         std::vector<Index>& rowPtr,
                                         std::vector<Index>& colInd) const {
    const size_t n = buses_.size();
    Y.clear();
    rowPtr.clear();
    colInd.clear();
    
    Y.reserve(n * 4);  // Estimate: average 4 connections per bus
    colInd.reserve(n * 4);
    rowPtr.resize(n + 1, 0);
    
    // Build structure
    colInd.reserve(n * 4);  // Estimate: average 4 connections
    
    for (size_t i = 0; i < n; ++i) {
        BusId busId = buses_[i]->getId();
        std::vector<Index> neighbors;
        neighbors.reserve(8);
        
        // Self admittance (shunt)
        neighbors.push_back(i);
        
        // Branches
        for (const auto& branch : branches_) {
            if (branch->getFromBus() == busId) {
                Index j = getBusIndex(branch->getToBus());
                if (j >= 0) neighbors.push_back(j);
            } else if (branch->getToBus() == busId) {
                Index j = getBusIndex(branch->getFromBus());
                if (j >= 0) neighbors.push_back(j);
            }
        }
        
        // Sort and deduplicate
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        
        rowPtr[i + 1] = rowPtr[i] + neighbors.size();
        for (Index j : neighbors) {
            colInd.push_back(j);
        }
    }
    
    // Fill values
    Y.resize(colInd.size(), Complex(0.0, 0.0));
    
    for (size_t i = 0; i < n; ++i) {
        BusId busId = buses_[i]->getId();
        Index start = rowPtr[i];
        Index end = rowPtr[i + 1];
        
        for (Index idx = start; idx < end; ++idx) {
            Index j = colInd[idx];
            
            if (static_cast<Index>(i) == j) {
                // Diagonal: sum of shunt admittances
                Complex yShunt(buses_[i]->getGShunt(), buses_[i]->getBShunt());
                
                // Add branch charging
                for (const auto& branch : branches_) {
                    if (branch->getFromBus() == busId || branch->getToBus() == busId) {
                        yShunt += Complex(0.0, branch->getB() / 2.0);
                    }
                }
                
                Y[idx] = yShunt;
            } else {
                // Off-diagonal: branch admittance
                BusId jBusId = buses_[j]->getId();
                for (const auto& branch : branches_) {
                    Complex yBranch = branch->getAdmittance();
                    Real tap = branch->getTapRatio();
                    
                    if (branch->getFromBus() == busId && branch->getToBus() == jBusId) {
                        Y[idx] = -yBranch / (tap * tap);
                        break;
                    } else if (branch->getFromBus() == jBusId && branch->getToBus() == busId) {
                        Y[idx] = -yBranch / (tap * tap);
                        break;
                    }
                }
            }
        }
    }
}

void NetworkModel::updateBus(BusId id, const Bus& busData) {
    Bus* bus = getBus(id);
    if (bus) {
        *bus = busData;  // Would need copy assignment operator
    }
}

void NetworkModel::updateBranch(BranchId id, const Branch& branchData) {
    Branch* branch = getBranch(id);
    if (branch) {
        *branch = branchData;  // Would need copy assignment operator
    }
}

void NetworkModel::removeBus(BusId id) {
    auto it = busIndexMap_.find(id);
    if (it != busIndexMap_.end()) {
        Index idx = it->second;
        buses_.erase(buses_.begin() + idx);
        busIndexMap_.erase(it);
        
        // Rebuild index map
        busIndexMap_.clear();
        for (size_t i = 0; i < buses_.size(); ++i) {
            busIndexMap_[buses_[i]->getId()] = i;
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
        
        // Rebuild index map
        branchIndexMap_.clear();
        for (size_t i = 0; i < branches_.size(); ++i) {
            branchIndexMap_[branches_[i]->getId()] = i;
        }
        invalidateCaches();
    }
}

void NetworkModel::invalidateAdmittanceMatrix() {
    // Mark that admittance matrix needs to be rebuilt
    // This would be used by solvers to know when to recompute
    invalidateCaches();
}

void NetworkModel::clear() {
    buses_.clear();
    branches_.clear();
    busIndexMap_.clear();
    branchIndexMap_.clear();
    referenceBus_ = -1;
    invalidateCaches();
}

void NetworkModel::invalidateCaches() {
#ifdef USE_CUDA
    deviceDataDirty_ = true;
#endif
    adjacencyDirty_ = true;
}

void NetworkModel::updateAdjacencyLists() const {
    if (!adjacencyDirty_) return;
    
    size_t nBuses = buses_.size();
    branchesFromBus_.clear();
    branchesToBus_.clear();
    branchesFromBus_.resize(nBuses);
    branchesToBus_.resize(nBuses);
    
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
    if (!gpuMemoryPool_) return;
    
    auto& pool = *gpuMemoryPool_;
    cudaError_t err;
    
    if (pool.vSize < nBuses) {
        if (pool.d_v) cudaFree(pool.d_v);
        err = cudaMalloc(&pool.d_v, nBuses * sizeof(Real));
        pool.vSize = (err == cudaSuccess) ? nBuses : 0;
    }
    if (pool.thetaSize < nBuses) {
        if (pool.d_theta) cudaFree(pool.d_theta);
        err = cudaMalloc(&pool.d_theta, nBuses * sizeof(Real));
        pool.thetaSize = (err == cudaSuccess) ? nBuses : 0;
    }
    if (pool.busesSize < nBuses) {
        if (pool.d_buses) cudaFree(pool.d_buses);
        err = cudaMalloc(&pool.d_buses, nBuses * sizeof(sle::cuda::DeviceBus));
        pool.busesSize = (err == cudaSuccess) ? nBuses : 0;
    }
    if (pool.branchesSize < nBranches) {
        if (pool.d_branches) cudaFree(pool.d_branches);
        err = cudaMalloc(&pool.d_branches, nBranches * sizeof(sle::cuda::DeviceBranch));
        pool.branchesSize = (err == cudaSuccess) ? nBranches : 0;
    }
    if (pool.branchFromBusSize < nBranches) {
        if (pool.d_branchFromBus) cudaFree(pool.d_branchFromBus);
        err = cudaMalloc(&pool.d_branchFromBus, nBranches * sizeof(Index));
        pool.branchFromBusSize = (err == cudaSuccess) ? nBranches : 0;
    }
    if (pool.branchToBusSize < nBranches) {
        if (pool.d_branchToBus) cudaFree(pool.d_branchToBus);
        err = cudaMalloc(&pool.d_branchToBus, nBranches * sizeof(Index));
        pool.branchToBusSize = (err == cudaSuccess) ? nBranches : 0;
    }
    if (pool.pInjectionSize < nBuses) {
        if (pool.d_pInjection) cudaFree(pool.d_pInjection);
        err = cudaMalloc(&pool.d_pInjection, nBuses * sizeof(Real));
        pool.pInjectionSize = (err == cudaSuccess) ? nBuses : 0;
    }
    if (pool.qInjectionSize < nBuses) {
        if (pool.d_qInjection) cudaFree(pool.d_qInjection);
        err = cudaMalloc(&pool.d_qInjection, nBuses * sizeof(Real));
        pool.qInjectionSize = (err == cudaSuccess) ? nBuses : 0;
    }
    if (pool.pFlowSize < nBranches) {
        if (pool.d_pFlow) cudaFree(pool.d_pFlow);
        err = cudaMalloc(&pool.d_pFlow, nBranches * sizeof(Real));
        pool.pFlowSize = (err == cudaSuccess) ? nBranches : 0;
    }
    if (pool.qFlowSize < nBranches) {
        if (pool.d_qFlow) cudaFree(pool.d_qFlow);
        err = cudaMalloc(&pool.d_qFlow, nBranches * sizeof(Real));
        pool.qFlowSize = (err == cudaSuccess) ? nBranches : 0;
    }
}

void NetworkModel::updateDeviceData() const {
    if (!deviceDataDirty_) return;
    
    size_t nBuses = buses_.size();
    size_t nBranches = branches_.size();
    
    cachedDeviceBuses_.clear();
    cachedDeviceBuses_.reserve(nBuses);
    for (const auto& bus : buses_) {
        sle::cuda::DeviceBus db;
        db.baseKV = bus->getBaseKV();
        db.gShunt = bus->getGShunt();
        db.bShunt = bus->getBShunt();
        cachedDeviceBuses_.push_back(db);
    }
    
    cachedDeviceBranches_.clear();
    cachedBranchFromBus_.clear();
    cachedBranchToBus_.clear();
    cachedDeviceBranches_.reserve(nBranches);
    cachedBranchFromBus_.reserve(nBranches);
    cachedBranchToBus_.reserve(nBranches);
    
    for (const auto& branch : branches_) {
        sle::cuda::DeviceBranch db;
        db.fromBus = getBusIndex(branch->getFromBus());
        db.toBus = getBusIndex(branch->getToBus());
        db.r = branch->getR();
        db.x = branch->getX();
        db.b = branch->getB();
        db.tapRatio = branch->getTapRatio();
        db.phaseShift = branch->getPhaseShift();
        cachedDeviceBranches_.push_back(db);
        cachedBranchFromBus_.push_back(db.fromBus);
        cachedBranchToBus_.push_back(db.toBus);
    }
    
    deviceDataDirty_ = false;
}
#endif

void NetworkModel::computeVoltEstimates(const StateVector& state, bool /* useGPU */) {
    size_t nBuses = buses_.size();
    const Real PI = 3.14159265359;
    const Real RAD_TO_DEG = 180.0 / PI;
    
    // Optimized CPU path with vectorization hints
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < nBuses; ++i) {
        Bus* bus = buses_[i].get();
        Real vPU = state.getVoltageMagnitude(static_cast<Index>(i));
        Real thetaRad = state.getVoltageAngle(static_cast<Index>(i));
        Real vKV = vPU * bus->getBaseKV();
        Real thetaDeg = thetaRad * RAD_TO_DEG;
        
        bus->setVoltEstimates(vPU, vKV, thetaRad, thetaDeg);
    }
}

void NetworkModel::computePowerInjections(const StateVector& state, bool useGPU) {
    size_t nBuses = buses_.size();
    std::vector<Real> pInjection, qInjection;
    
    // Compute power injections (using existing method)
    computePowerInjections(state, pInjection, qInjection, useGPU);
    
    // Store in Bus objects
    Real baseMVA = getBaseMVA();
    for (size_t i = 0; i < nBuses && i < pInjection.size(); ++i) {
        Bus* bus = buses_[i].get();
        Real pMW = pInjection[i] * baseMVA;
        Real qMVAR = qInjection[i] * baseMVA;
        bus->setPowerInjections(pInjection[i], qInjection[i], pMW, qMVAR);
    }
}

void NetworkModel::computePowerFlows(const StateVector& state, bool useGPU) {
    size_t nBranches = branches_.size();
    std::vector<Real> pFlow, qFlow;
    
    // Compute power flows (using existing method)
    computePowerFlows(state, pFlow, qFlow, useGPU);
    
    // Store in Branch objects
    Real baseMVA = getBaseMVA();
    auto voltages = state.getMagnitudes();
    
    for (size_t i = 0; i < nBranches && i < pFlow.size(); ++i) {
        Branch* branch = branches_[i].get();
        Index fromIdx = getBusIndex(branch->getFromBus());
        
        if (fromIdx >= 0 && static_cast<size_t>(fromIdx) < voltages.size()) {
            Real vFrom = voltages[fromIdx];
            Real p = pFlow[i];
            Real q = qFlow[i];
            Real pMW = p * baseMVA;
            Real qMVAR = q * baseMVA;
            
            // Compute current
            Real iPU = branch->computeCurrentMagnitude(p, q, vFrom);
            
            // Convert to Amperes
            Bus* fromBus = getBus(branch->getFromBus());
            Real baseKV = fromBus ? fromBus->getBaseKV() : 100.0;
            Real baseCurrent = (baseMVA * 1000.0) / (std::sqrt(3.0) * baseKV);
            Real iAmps = iPU * baseCurrent;
            
            branch->setPowerFlow(p, q, pMW, qMVAR, iAmps, iPU);
        }
    }
}

void NetworkModel::computePowerInjections(const StateVector& state,
                                          std::vector<Real>& pInjection, 
                                          std::vector<Real>& qInjection,
                                          bool useGPU) const {
    size_t nBuses = buses_.size();
    (void)useGPU;  // May be used in GPU path
    pInjection.assign(nBuses, 0.0);
    qInjection.assign(nBuses, 0.0);
    
#ifdef USE_CUDA
    if (useGPU && gpuMemoryPool_) {
        try {
            // Ensure GPU memory pool has capacity
            ensureGPUCapacity(nBuses, nBranches);
            auto& pool = *gpuMemoryPool_;
            
            // Check if allocations succeeded
            if (!pool.d_v || !pool.d_theta || !pool.d_buses || !pool.d_branches ||
                !pool.d_branchFromBus || !pool.d_branchToBus ||
                !pool.d_pInjection || !pool.d_qInjection) {
                useGPU = false;  // Fall back to CPU
            } else {
                // Update cached device data if needed
                updateDeviceData();
                
                // Get state vectors
                std::vector<Real> v = state.getMagnitudes();
                std::vector<Real> theta = state.getAngles();
                
                // Copy to device (reuse existing allocations)
                cudaMemcpy(pool.d_v, v.data(), nBuses * sizeof(Real), cudaMemcpyHostToDevice);
                cudaMemcpy(pool.d_theta, theta.data(), nBuses * sizeof(Real), cudaMemcpyHostToDevice);
                cudaMemcpy(pool.d_buses, cachedDeviceBuses_.data(), 
                          nBuses * sizeof(sle::cuda::DeviceBus), cudaMemcpyHostToDevice);
                cudaMemcpy(pool.d_branches, cachedDeviceBranches_.data(), 
                          nBranches * sizeof(sle::cuda::DeviceBranch), cudaMemcpyHostToDevice);
                cudaMemcpy(pool.d_branchFromBus, cachedBranchFromBus_.data(), 
                          nBranches * sizeof(Index), cudaMemcpyHostToDevice);
                cudaMemcpy(pool.d_branchToBus, cachedBranchToBus_.data(), 
                          nBranches * sizeof(Index), cudaMemcpyHostToDevice);
                
                // Launch GPU kernel
                sle::cuda::computeAllPowerInjectionsGPU(
                    pool.d_v, pool.d_theta, pool.d_buses, pool.d_branches,
                    pool.d_branchFromBus, pool.d_branchToBus,
                    pool.d_pInjection, pool.d_qInjection,
                    static_cast<Index>(nBuses), static_cast<Index>(nBranches));
                
                // Copy back (async could be used here for better performance)
                cudaMemcpy(pInjection.data(), pool.d_pInjection, 
                          nBuses * sizeof(Real), cudaMemcpyDeviceToHost);
                cudaMemcpy(qInjection.data(), pool.d_qInjection, 
                          nBuses * sizeof(Real), cudaMemcpyDeviceToHost);
                
                return;  // Success, return early
            }
        } catch (...) {
            // Fall back to CPU if GPU fails
            useGPU = false;
        }
    }
#endif
    
    // CPU fallback - optimized with adjacency lists and vectorization
    updateAdjacencyLists();
    
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < nBuses; ++i) {
        const Bus* bus = buses_[i].get();
        Real v = state.getVoltageMagnitude(static_cast<Index>(i));
        Real v2 = v * v;  // Reuse v^2 computation
        // Note: theta not needed for shunt-only contribution calculation
        
        // Shunt contribution
        pInjection[i] = v2 * bus->getGShunt();
        qInjection[i] = -v2 * bus->getBShunt();
        
        // Branch contributions (outgoing) - use cached adjacency list
        for (Index brIdx : branchesFromBus_[i]) {
            const Branch* branch = branches_[brIdx].get();
            Index toIdx = getBusIndex(branch->getToBus());
            if (toIdx >= 0) {
                Real pFlow, qFlow;
                branch->computePowerFlow(state, static_cast<Index>(i), toIdx, pFlow, qFlow);
                pInjection[i] += pFlow;
                qInjection[i] += qFlow;
            }
        }
        
        // Branch contributions (incoming - reverse direction)
        for (Index brIdx : branchesToBus_[i]) {
            const Branch* branch = branches_[brIdx].get();
            Index fromIdx = getBusIndex(branch->getFromBus());
            if (fromIdx >= 0) {
                Real pFlow, qFlow;
                branch->computePowerFlow(state, fromIdx, static_cast<Index>(i), pFlow, qFlow);
                pInjection[i] -= pFlow;  // Reverse direction
                qInjection[i] -= qFlow;
            }
        }
    }
}

void NetworkModel::computePowerFlows(const StateVector& state,
                                     std::vector<Real>& pFlow, 
                                     std::vector<Real>& qFlow,
                                     bool useGPU) const {
    (void)useGPU;  // May be used in GPU path
    size_t nBranches = branches_.size();
    pFlow.assign(nBranches, 0.0);
    qFlow.assign(nBranches, 0.0);
    
#ifdef USE_CUDA
    if (useGPU && gpuMemoryPool_) {
        try {
            size_t nBuses = buses_.size();
            ensureGPUCapacity(nBuses, nBranches);
            auto& pool = *gpuMemoryPool_;
            
            if (!pool.d_v || !pool.d_theta || !pool.d_branches || 
                !pool.d_pFlow || !pool.d_qFlow) {
                useGPU = false;
            } else {
                // Update cached device data if needed
                updateDeviceData();
                
                // Get state vectors
                std::vector<Real> v = state.getMagnitudes();
                std::vector<Real> theta = state.getAngles();
                
                // Copy to device (reuse existing allocations)
                cudaMemcpy(pool.d_v, v.data(), nBuses * sizeof(Real), cudaMemcpyHostToDevice);
                cudaMemcpy(pool.d_theta, theta.data(), nBuses * sizeof(Real), cudaMemcpyHostToDevice);
                cudaMemcpy(pool.d_branches, cachedDeviceBranches_.data(), 
                          nBranches * sizeof(sle::cuda::DeviceBranch), cudaMemcpyHostToDevice);
                
                // Launch GPU kernel
                sle::cuda::computeAllPowerFlowsGPU(
                    pool.d_v, pool.d_theta, pool.d_branches,
                    pool.d_pFlow, pool.d_qFlow,
                    static_cast<Index>(nBranches), static_cast<Index>(nBuses));
                
                // Copy back
                cudaMemcpy(pFlow.data(), pool.d_pFlow, 
                          nBranches * sizeof(Real), cudaMemcpyDeviceToHost);
                cudaMemcpy(qFlow.data(), pool.d_qFlow, 
                          nBranches * sizeof(Real), cudaMemcpyDeviceToHost);
                
                return;  // Success, return early
            }
        } catch (...) {
            useGPU = false;
        }
    }
#endif
    
    // CPU fallback - optimized with vectorization
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < nBranches; ++i) {
        const Branch* branch = branches_[i].get();
        Index fromIdx = getBusIndex(branch->getFromBus());
        Index toIdx = getBusIndex(branch->getToBus());
        
        if (fromIdx >= 0 && toIdx >= 0) {
            branch->computePowerFlow(state, fromIdx, toIdx, pFlow[i], qFlow[i]);
        }
    }
}

} // namespace model
} // namespace sle

