/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_NETWORKMODEL_H
#define SLE_MODEL_NETWORKMODEL_H

#include <sle/model/Bus.h>
#include <sle/model/Branch.h>
#include <sle/model/CircuitBreaker.h>
#include <sle/Types.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <sle/cuda/CudaPowerFlow.h>

// Forward declaration
namespace sle {
namespace cuda {
    class CudaDataManager;
}
}

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
    std::vector<const Branch*> getBranchesFromBus(BusId busId) const;
    std::vector<const Branch*> getBranchesToBus(BusId busId) const;
    
    // OPTIMIZED: Get branch by from/to bus IDs (O(1) lookup)
    Branch* getBranchByBuses(BusId fromBus, BusId toBus);
    const Branch* getBranchByBuses(BusId fromBus, BusId toBus) const;
    
    // OPTIMIZED: Direct access to internal branch indices (no vector allocation)
    const std::vector<Index>& getBranchIndicesFromBus(BusId busId) const;
    const std::vector<Index>& getBranchIndicesToBus(BusId busId) const;
    const Branch* getBranchByIndex(Index index) const;
    
    // Base values
    void setBaseMVA(Real baseMVA) { baseMVA_ = baseMVA; }
    Real getBaseMVA() const { return baseMVA_; }
    
    // Reference bus
    void setReferenceBus(BusId busId);
    BusId getReferenceBus() const { return referenceBus_; }
    
    // Real-time updates
    void updateBus(BusId id, const Bus& busData);
    void updateBranch(BranchId id, const Branch& branchData);
    void removeBus(BusId id);
    void removeBranch(BranchId id);
    void clear();
    
    // Circuit breaker management
    CircuitBreaker* addCircuitBreaker(const std::string& id, BranchId branchId, BusId fromBus, BusId toBus, const std::string& name = "");
    CircuitBreaker* getCircuitBreaker(const std::string& id);
    const CircuitBreaker* getCircuitBreaker(const std::string& id) const;
    CircuitBreaker* getCircuitBreakerByBranch(BranchId branchId);
    const CircuitBreaker* getCircuitBreakerByBranch(BranchId branchId) const;
    std::vector<CircuitBreaker*> getCircuitBreakers();
    std::vector<const CircuitBreaker*> getCircuitBreakers() const;
    size_t getCircuitBreakerCount() const { return circuitBreakers_.size(); }
    
    // Set callback for topology change notifications (called when circuit breaker status changes)
    // Callback signature: void()
    using TopologyChangeCallback = std::function<void()>;
    void setTopologyChangeCallback(TopologyChangeCallback callback) { topologyChangeCallback_ = callback; }
    
    // Get index from bus ID
    Index getBusIndex(BusId id) const;
    Index getBranchIndex(BranchId id) const;
    
    // CUDA-EXCLUSIVE: Compute and store power injections in buses
    // Computes P, Q, MW, MVAR injections for all buses and stores in Bus objects
    // Used by LoadFlow solver
    void computePowerInjections(const StateVector& state);
    
    // Legacy method (for backward compatibility) - return vectors instead of storing
    // CUDA-EXCLUSIVE: All computations on GPU
    // Used internally by computePowerInjections(state) and by LoadFlow
    // Optional dataManager: if provided, reuses existing GPU data; otherwise creates temporary
    void computePowerInjections(const StateVector& state,
                               std::vector<Real>& pInjection, std::vector<Real>& qInjection,
                               sle::cuda::CudaDataManager* dataManager = nullptr) const;
    
    // Note: computePowerFlows and computeVoltEstimates methods removed
    // They are replaced by Solver::storeComputedValues() which reuses GPU-computed values
    // This eliminates redundant GPU computations and host-device transfers
    
private:
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
    
    // Internal data manager for GPU operations when no external one is provided
    mutable std::shared_ptr<sle::cuda::CudaDataManager> internalDataManager_;
    
    // Adjacency lists for O(1) branch queries (updated on changes)
    mutable std::vector<std::vector<Index>> branchesFromBus_;  // branchesFromBus_[busIdx] = branch indices
    mutable std::vector<std::vector<Index>> branchesToBus_;    // branchesToBus_[busIdx] = branch indices
    mutable bool adjacencyDirty_;
    
    // Cached CPU vectors for power injection computations (reused across calls, only resized on network changes)
    // Note: Power flows are computed directly in Solver::storeComputedValues, no caching needed
    // Mutable allows these to be updated in const methods (like computePowerInjections)
    mutable std::vector<Real> cachedPInjection_;
    mutable std::vector<Real> cachedQInjection_;
    
    // Helper methods
    void updateDeviceData() const;
    sle::cuda::CudaDataManager* getInternalDataManager() const;
    void updateAdjacencyLists() const;
    void invalidateCaches();
    
    std::vector<std::unique_ptr<Bus>> buses_;
    std::vector<std::unique_ptr<Branch>> branches_;
    std::vector<std::unique_ptr<CircuitBreaker>> circuitBreakers_;
    std::unordered_map<BusId, Index> busIndexMap_;
    std::unordered_map<BranchId, Index> branchIndexMap_;
    std::unordered_map<std::string, Index> busNameMap_;  // Name -> index mapping for O(1) lookup
    std::unordered_map<std::string, Index> circuitBreakerIndexMap_;  // CB ID -> index mapping
    std::unordered_map<BranchId, Index> branchToCircuitBreakerMap_;  // Branch ID -> CB index mapping
    
    // OPTIMIZATION: Fast branch lookup by (fromBus, toBus) pair (O(1) instead of O(avg_degree))
    // Key: pair<fromBus, toBus>, Value: branch index
    // Note: std::pair has a default hash function in C++11+, but we use a custom one for better distribution
    struct BusPairHash {
        std::size_t operator()(const std::pair<BusId, BusId>& p) const {
            // Combine both IDs with a simple hash (works well for power system IDs)
            return std::hash<BusId>{}(p.first) ^ (std::hash<BusId>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<BusId, BusId>, Index, BusPairHash> branchBusPairMap_;
    
    TopologyChangeCallback topologyChangeCallback_;  // Called when circuit breaker status changes
    
    Real baseMVA_;
    BusId referenceBus_;
    
    // Internal callback handler for circuit breaker status changes
    void onCircuitBreakerStatusChanged(BranchId branchId, bool newStatus);
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_NETWORKMODEL_H
