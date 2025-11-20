/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/NetworkModel.h>
#include <algorithm>
#include <cmath>

namespace sle {
namespace model {

NetworkModel::NetworkModel() : baseMVA_(100.0), referenceBus_(-1) {
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
    std::vector<Branch*> result;
    for (auto& branch : branches_) {
        if (branch->getFromBus() == busId) {
            result.push_back(branch.get());
        }
    }
    return result;
}

std::vector<Branch*> NetworkModel::getBranchesToBus(BusId busId) {
    std::vector<Branch*> result;
    for (auto& branch : branches_) {
        if (branch->getToBus() == busId) {
            result.push_back(branch.get());
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
            
            if (i == j) {
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
    }
}

void NetworkModel::invalidateAdmittanceMatrix() {
    // Mark that admittance matrix needs to be rebuilt
    // This would be used by solvers to know when to recompute
}

void NetworkModel::clear() {
    buses_.clear();
    branches_.clear();
    busIndexMap_.clear();
    branchIndexMap_.clear();
    referenceBus_ = -1;
}

} // namespace model
} // namespace sle

