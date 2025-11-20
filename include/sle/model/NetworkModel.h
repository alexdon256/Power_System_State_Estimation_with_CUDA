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

namespace sle {
namespace model {

class NetworkModel {
public:
    NetworkModel();
    ~NetworkModel();
    
    // Bus management
    Bus* addBus(BusId id, const std::string& name = "");
    Bus* getBus(BusId id);
    const Bus* getBus(BusId id) const;
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
    
private:
    std::vector<std::unique_ptr<Bus>> buses_;
    std::vector<std::unique_ptr<Branch>> branches_;
    std::unordered_map<BusId, Index> busIndexMap_;
    std::unordered_map<BranchId, Index> branchIndexMap_;
    
    Real baseMVA_;
    BusId referenceBus_;
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_NETWORKMODEL_H

