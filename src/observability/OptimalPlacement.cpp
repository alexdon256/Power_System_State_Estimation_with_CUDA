/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/observability/OptimalPlacement.h>
#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <algorithm>
#include <cmath>

namespace sle {
namespace observability {

OptimalPlacement::OptimalPlacement() {
}

std::vector<MeasurementPlacement> OptimalPlacement::findOptimalPlacement(
    const model::NetworkModel& network,
    const std::set<BusId>& existingMeasurements,
    size_t maxMeasurements,
    Real budget) {
    
    if (budget > 0.0) {
        return integerProgrammingPlacement(network, existingMeasurements,
                                          maxMeasurements, budget);
    } else {
        return greedyPlacement(network, existingMeasurements, maxMeasurements);
    }
}

std::vector<MeasurementPlacement> OptimalPlacement::greedyPlacement(
    const model::NetworkModel& network,
    const std::set<BusId>& existingMeasurements,
    size_t maxMeasurements) {
    
    std::vector<MeasurementPlacement> placements;
    std::set<BusId> currentMeasurements = existingMeasurements;
    
    // Check if already observable
    if (isObservableWithMeasurements(network, currentMeasurements)) {
        return placements;
    }
    
    auto buses = network.getBuses();
    std::vector<std::pair<Real, BusId>> candidates;
    
    // Evaluate all candidate buses
    for (auto* bus : buses) {
        if (currentMeasurements.count(bus->getId()) > 0) {
            continue;  // Already has measurement
        }
        
        Real gain = computeObservabilityGain(network, currentMeasurements,
                                            bus->getId(), MeasurementType::V_MAGNITUDE);
        candidates.push_back({gain, bus->getId()});
    }
    
    // Sort by gain (descending)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Greedily add measurements
    for (size_t i = 0; i < std::min(maxMeasurements, candidates.size()); ++i) {
        BusId busId = candidates[i].second;
        currentMeasurements.insert(busId);
        
        MeasurementPlacement placement;
        placement.busId = busId;
        placement.type = MeasurementType::V_MAGNITUDE;
        placement.cost = 1.0;  // Default cost
        placement.priority = static_cast<int>(i);
        placements.push_back(placement);
        
        // Check if observable now
        if (isObservableWithMeasurements(network, currentMeasurements)) {
            break;
        }
    }
    
    return placements;
}

std::vector<MeasurementPlacement> OptimalPlacement::integerProgrammingPlacement(
    const model::NetworkModel& network,
    const std::set<BusId>& existingMeasurements,
    size_t maxMeasurements,
    Real budget) {
    
    // Simplified integer programming approach
    // Full implementation would use a proper IP solver
    
    // For now, use greedy with budget constraint
    auto candidates = greedyPlacement(network, existingMeasurements, maxMeasurements);
    
    // Filter by budget
    Real totalCost = 0.0;
    std::vector<MeasurementPlacement> result;
    
    for (const auto& placement : candidates) {
        if (totalCost + placement.cost <= budget) {
            result.push_back(placement);
            totalCost += placement.cost;
        }
    }
    
    return result;
}

std::set<BusId> OptimalPlacement::identifyCriticalMeasurements(
    const model::NetworkModel& network,
    const std::set<BusId>& measurements) {
    
    std::set<BusId> critical;
    
    // A measurement is critical if removing it makes system unobservable
    for (BusId measId : measurements) {
        std::set<BusId> testMeasurements = measurements;
        testMeasurements.erase(measId);
        
        if (!isObservableWithMeasurements(network, testMeasurements)) {
            critical.insert(measId);
        }
    }
    
    return critical;
}

Real OptimalPlacement::computeObservabilityGain(
    const model::NetworkModel& network,
    const std::set<BusId>& currentMeasurements,
    BusId candidateBus,
    MeasurementType type) {
    
    // Compute how many additional buses become observable
    std::set<BusId> testMeasurements = currentMeasurements;
    testMeasurements.insert(candidateBus);
    
    if (!isObservableWithMeasurements(network, currentMeasurements) &&
        isObservableWithMeasurements(network, testMeasurements)) {
        return 1000.0;  // High gain if it makes system observable
    }
    
    // Count newly observable buses
    // Simplified - would properly compute observability gain
    return 1.0;
}

bool OptimalPlacement::isObservableWithMeasurements(
    const model::NetworkModel& network,
    const std::set<BusId>& measurements) {
    
    // Build observability matrix
    std::vector<std::vector<bool>> obsMatrix;
    buildObservabilityMatrix(network, measurements, obsMatrix);
    
    // Check rank (simplified - would use proper rank computation)
    size_t rank = 0;
    for (const auto& row : obsMatrix) {
        bool hasOne = false;
        for (bool val : row) {
            if (val) {
                hasOne = true;
                break;
            }
        }
        if (hasOne) rank++;
    }
    
    size_t nBuses = network.getBusCount();
    return rank >= nBuses;  // Simplified observability check
}

void OptimalPlacement::buildObservabilityMatrix(
    const model::NetworkModel& network,
    const std::set<BusId>& measurements,
    std::vector<std::vector<bool>>& obsMatrix) {
    
    size_t nBuses = network.getBusCount();
    obsMatrix.assign(measurements.size(), std::vector<bool>(nBuses, false));
    
    size_t row = 0;
    for (BusId measId : measurements) {
        Index busIdx = network.getBusIndex(measId);
        if (busIdx >= 0 && static_cast<size_t>(busIdx) < nBuses) {
            obsMatrix[row][busIdx] = true;
        }
        row++;
    }
}

} // namespace observability
} // namespace sle

