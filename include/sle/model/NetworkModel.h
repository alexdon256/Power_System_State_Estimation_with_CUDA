/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_NETWORKMODEL_H
#define SLE_MODEL_NETWORKMODEL_H

#include <sle/model/Bus.h>
#include <sle/model/Branch.h>
#include <sle/Types.h>
#include <vector>
#include <unordered_map>
#include <memory>

#ifdef USE_CUDA
#include <sle/cuda/CudaPowerFlow.h>
#endif

namespace sle {
namespace model {

class NetworkModel {
public:
    NetworkModel();
    ~NetworkModel();
    
    // Delete copy constructor and assignment (contains unique_ptr vectors)
    NetworkModel(const NetworkModel&) = delete;
    NetworkModel& operator=(const NetworkModel&) = delete;
    
    // Allow move constructor and assignment
    NetworkModel(NetworkModel&&) = default;
    NetworkModel& operator=(NetworkModel&&) = default;
    
    // Bus management
    Bus* addBus(BusId id, const std::string& name = "");
    Bus* getBus(BusId id);
    const Bus* getBus(BusId id) const;
    Bus* getBusByName(const std::string& name);  // Search by name (case-sensitive)
    const Bus* getBusByName(const std::string& name) const;  // Search by name (case-sensitive)
    std::vector<Bus*> getBuses();
    std::vector<const Bus*> getBuses() const;
    size_t getBusCount() const { return buses_.size(); }
    
    // Branch management
    Branch* addBranch(BranchId id, BusId fromBus, BusId toBus);
    Branch* getBranch(BranchId id);
    const Branch* getBranch(BranchId id) const;
    std::vector<Branch*> getBranches();
    std::vector<const Branch*> getBranches() const;
    size_t getBranchCount() const { return branches_.size(); }
    
    // Get branches connected to a bus
    std::vector<Branch*> getBranchesFromBus(BusId busId);
    std::vector<Branch*> getBranchesToBus(BusId busId);
    
    // Base values
    void setBaseMVA(Real baseMVA) { baseMVA_ = baseMVA; }
    Real getBaseMVA() const { return baseMVA_; }
    
    // Reference bus
    void setReferenceBus(BusId busId);
    BusId getReferenceBus() const { return referenceBus_; }
    
    // Build admittance matrix (for CPU reference)
    void buildAdmittanceMatrix(std::vector<Complex>& Y, std::vector<Index>& rowPtr, 
                               std::vector<Index>& colInd) const;
    
    // Real-time updates
    void updateBus(BusId id, const Bus& busData);
    void updateBranch(BranchId id, const Branch& branchData);
    void removeBus(BusId id);
    void removeBranch(BranchId id);
    void invalidateAdmittanceMatrix();  // Mark matrix as needing rebuild
    
    void clear();
    
    // Get index from bus ID
    Index getBusIndex(BusId id) const;
    Index getBranchIndex(BranchId id) const;
    
    // Compute and store voltage estimates in buses
    // Computes vPU, vKV, thetaRad, thetaDeg for all buses and stores in Bus objects
    // useGPU: if true, uses GPU acceleration (requires CUDA)
    void computeVoltEstimates(const StateVector& state, bool useGPU = false);
    
    // Compute and store power injections in buses
    // Computes P, Q, MW, MVAR injections for all buses and stores in Bus objects
    // useGPU: if true, uses GPU acceleration (requires CUDA)
    void computePowerInjections(const StateVector& state, bool useGPU = false);
    
    // Compute and store power flows in branches
    // Computes P, Q, MW, MVAR, I (amps and p.u.) for all branches and stores in Branch objects
    // useGPU: if true, uses GPU acceleration (requires CUDA)
    void computePowerFlows(const StateVector& state, bool useGPU = false);
    
    // Legacy methods (for backward compatibility) - return vectors instead of storing
    void computePowerInjections(const StateVector& state,
                               std::vector<Real>& pInjection, std::vector<Real>& qInjection,
                               bool useGPU = false) const;
    
    void computePowerFlows(const StateVector& state,
                          std::vector<Real>& pFlow, std::vector<Real>& qFlow,
                          bool useGPU = false) const;
    
private:
#ifdef USE_CUDA
    // GPU memory pool for performance (reuse allocations)
    struct CudaMemoryPool;
    mutable std::unique_ptr<CudaMemoryPool> gpuMemoryPool_;
#endif
    
#ifdef USE_CUDA
    // Cached device data structures (updated incrementally)
    mutable std::vector<sle::cuda::DeviceBus> cachedDeviceBuses_;
    mutable std::vector<sle::cuda::DeviceBranch> cachedDeviceBranches_;
    // CSR format adjacency lists for GPU (column indices)
    mutable std::vector<Index> cachedBranchFromBus_;  // CSR column indices
    mutable std::vector<Index> cachedBranchToBus_;    // CSR column indices
    // CSR format row pointers (nBuses+1 elements)
    mutable std::vector<Index> cachedBranchFromBusRowPtr_;  // CSR row pointers
    mutable std::vector<Index> cachedBranchToBusRowPtr_;    // CSR row pointers
    mutable bool deviceDataDirty_;
#endif
    
    // Adjacency lists for O(1) branch queries (updated on changes)
    mutable std::vector<std::vector<Index>> branchesFromBus_;  // branchesFromBus_[busIdx] = branch indices
    mutable std::vector<std::vector<Index>> branchesToBus_;    // branchesToBus_[busIdx] = branch indices
    mutable bool adjacencyDirty_;
    
    // Cached CPU vectors for power computations (reused across calls, only resized on network changes)
    mutable std::vector<Real> cachedPInjection_;
    mutable std::vector<Real> cachedQInjection_;
    mutable std::vector<Real> cachedPFlow_;
    mutable std::vector<Real> cachedQFlow_;
    
    // Helper methods
#ifdef USE_CUDA
    void ensureGPUCapacity(size_t nBuses, size_t nBranches) const;
    void updateDeviceData() const;
#endif
    void updateAdjacencyLists() const;
    void invalidateCaches();
    
    std::vector<std::unique_ptr<Bus>> buses_;
    std::vector<std::unique_ptr<Branch>> branches_;
    std::unordered_map<BusId, Index> busIndexMap_;
    std::unordered_map<BranchId, Index> branchIndexMap_;
    std::unordered_map<std::string, Index> busNameMap_;  // Name -> index mapping for O(1) lookup
    
    Real baseMVA_;
    BusId referenceBus_;
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_NETWORKMODEL_H

