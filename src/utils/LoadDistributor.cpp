/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/utils/LoadDistributor.h>
#include <sle/model/NetworkModel.h>

namespace sle {
namespace utils {

void LoadDistributor::distributeLoadsProportional(
    model::NetworkModel& network,
    const std::vector<Real>& totalLoad) {
    
    auto buses = network.getBuses();
    Real totalP = 0.0;
    Real totalQ = 0.0;
    
    // Calculate total existing load
    for (auto* bus : buses) {
        totalP += bus->getPLoad();
        totalQ += bus->getQLoad();
    }
    
    // Distribute proportionally
    if (totalP > 1e-6) {
        Real factorP = totalLoad[0] / totalP;
        for (auto* bus : buses) {
            bus->setLoad(bus->getPLoad() * factorP, bus->getQLoad() * factorP);
        }
    }
}

void LoadDistributor::distributeLoadsHistorical(
    model::NetworkModel& network,
    const std::vector<Real>& historicalFactors) {
    
    auto buses = network.getBuses();
    
    for (size_t i = 0; i < buses.size() && i < historicalFactors.size(); ++i) {
        Real factor = historicalFactors[i];
        Real pLoad = buses[i]->getPLoad() * factor;
        Real qLoad = buses[i]->getQLoad() * factor;
        buses[i]->setLoad(pLoad, qLoad);
    }
}

} // namespace utils
} // namespace sle

