/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Utility functions for converting network data to GPU device structures
 * Eliminates code duplication between MeasurementFunctions and JacobianMatrix
 */

#include <sle/cuda/CudaNetworkUtils.h>
#include <sle/model/NetworkModel.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <sle/Types.h>
#include <vector>

namespace sle {
namespace cuda {

void buildDeviceBuses(const model::NetworkModel& network, 
                     std::vector<DeviceBus>& deviceBuses) {
    size_t nBuses = network.getBusCount();
    deviceBuses.clear();
    deviceBuses.reserve(nBuses);
    
    auto buses = network.getBuses();
    for (const auto* bus : buses) {
        DeviceBus db;
        db.baseKV = bus->getBaseKV();
        db.gShunt = bus->getGShunt();
        db.bShunt = bus->getBShunt();
        deviceBuses.push_back(db);
    }
}

void buildDeviceBranches(const model::NetworkModel& network,
                        std::vector<DeviceBranch>& deviceBranches) {
    size_t nBuses = network.getBusCount();
    size_t nBranches = network.getBranchCount();
    deviceBranches.clear();
    
    auto branches = network.getBranches();
    for (const auto* branch : branches) {
        if (!branch->isOn()) continue; // Skip offline branches
        
        DeviceBranch db;
        Index fromIdx = network.getBusIndex(branch->getFromBus());
        Index toIdx = network.getBusIndex(branch->getToBus());
        db.fromBus = (fromIdx >= 0) ? fromIdx : -1;
        db.toBus = (toIdx >= 0) ? toIdx : -1;
        db.r = branch->getR();
        db.x = branch->getX();
        db.b = branch->getB();
        db.tapRatio = branch->getTapRatio();
        db.phaseShift = branch->getPhaseShift();
        deviceBranches.push_back(db);
    }
}

void buildCSRAdjacencyLists(const model::NetworkModel& network,
                            std::vector<Index>& branchFromBus,
                            std::vector<Index>& branchFromBusRowPtr,
                            std::vector<Index>& branchToBus,
                            std::vector<Index>& branchToBusRowPtr) {
    size_t nBuses = network.getBusCount();
    
    branchFromBus.clear();
    branchFromBusRowPtr.clear();
    branchFromBusRowPtr.resize(nBuses + 1, 0);
    branchToBus.clear();
    branchToBusRowPtr.clear();
    branchToBusRowPtr.resize(nBuses + 1, 0);
    
    auto buses = network.getBuses();
    for (size_t i = 0; i < nBuses; ++i) {
        BusId busId = buses[i]->getId();
        auto branchesFrom = network.getBranchesFromBus(busId);
        auto branchesTo = network.getBranchesToBus(busId);
        
        branchFromBusRowPtr[i + 1] = branchFromBusRowPtr[i] + branchesFrom.size();
        for (const auto* br : branchesFrom) {
            Index brIdx = network.getBranchIndex(br->getId());
            if (brIdx >= 0) branchFromBus.push_back(static_cast<Index>(brIdx));
        }
        
        branchToBusRowPtr[i + 1] = branchToBusRowPtr[i] + branchesTo.size();
        for (const auto* br : branchesTo) {
            Index brIdx = network.getBranchIndex(br->getId());
            if (brIdx >= 0) branchToBus.push_back(static_cast<Index>(brIdx));
        }
    }
}

Index mapMeasurementTypeToIndex(MeasurementType type) {
    switch (type) {
        case MeasurementType::P_FLOW: return 0;
        case MeasurementType::Q_FLOW: return 1;
        case MeasurementType::P_INJECTION: return 2;
        case MeasurementType::Q_INJECTION: return 3;
        case MeasurementType::V_MAGNITUDE: return 4;
        case MeasurementType::I_MAGNITUDE: return 5;
        default: return -1;
    }
}

Index findBranchIndex(const model::NetworkModel& network, 
                     BusId fromBus, BusId toBus) {
    if (fromBus < 0 || toBus < 0) return -1;
    
    // OPTIMIZATION: Use O(1) lookup via getBranchByBuses instead of O(n) linear search
    // Check both directions (fromBus->toBus and toBus->fromBus)
    const model::Branch* branch = network.getBranchByBuses(fromBus, toBus);
    if (branch) {
        return network.getBranchIndex(branch->getId());
    }
    
    // Try reverse direction (some branches may be stored in reverse)
    branch = network.getBranchByBuses(toBus, fromBus);
    if (branch) {
        return network.getBranchIndex(branch->getId());
    }
    
    return -1;
}

} // namespace cuda
} // namespace sle

