/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_OBSERVABILITY_OPTIMALPLACEMENT_H
#define SLE_OBSERVABILITY_OPTIMALPLACEMENT_H

#include <sle/Export.h>
#include <sle/Types.h>
#include <vector>
#include <set>
#include <string>

// Forward declaration
namespace sle {
namespace model {
    class NetworkModel;
}
}

namespace sle {
namespace observability {

struct SLE_API MeasurementPlacement {
    BusId busId;
    MeasurementType type;
    Real cost;
    int priority;
};

class SLE_API OptimalPlacement {
public:
    OptimalPlacement();
    
    // Find optimal measurement placement for observability
    std::vector<MeasurementPlacement> findOptimalPlacement(
        const model::NetworkModel& network,
        const std::set<BusId>& existingMeasurements,
        size_t maxMeasurements,
        Real budget = 0.0);
    
    // Greedy algorithm for measurement placement
    std::vector<MeasurementPlacement> greedyPlacement(
        const model::NetworkModel& network,
        const std::set<BusId>& existingMeasurements,
        size_t maxMeasurements);
    
    // Integer programming approach (simplified)
    std::vector<MeasurementPlacement> integerProgrammingPlacement(
        const model::NetworkModel& network,
        const std::set<BusId>& existingMeasurements,
        size_t maxMeasurements,
        Real budget);
    
    // Critical measurement identification
    std::set<BusId> identifyCriticalMeasurements(
        const model::NetworkModel& network,
        const std::set<BusId>& measurements);
    
    // Compute observability gain for a measurement
    Real computeObservabilityGain(const model::NetworkModel& network,
                                  const std::set<BusId>& currentMeasurements,
                                  BusId candidateBus,
                                  MeasurementType type);
    
private:
    // Graph-based observability checking
    bool isObservableWithMeasurements(const model::NetworkModel& network,
                                     const std::set<BusId>& measurements);
    
    // Build observability matrix
    void buildObservabilityMatrix(const model::NetworkModel& network,
                                  const std::set<BusId>& measurements,
                                  std::vector<std::vector<bool>>& obsMatrix);
};

} // namespace observability
} // namespace sle

#endif // SLE_OBSERVABILITY_OPTIMALPLACEMENT_H

