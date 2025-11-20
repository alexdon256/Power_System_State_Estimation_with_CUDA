/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_OBSERVABILITY_OBSERVABILITYANALYZER_H
#define SLE_OBSERVABILITY_OBSERVABILITYANALYZER_H

#include <sle/model/NetworkModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/Types.h>
#include <vector>
#include <set>

namespace sle {
namespace observability {

struct ObservableSubsystem {
    std::set<BusId> buses;
    std::set<BranchId> branches;
    bool isObservable;
};

class ObservabilityAnalyzer {
public:
    ObservabilityAnalyzer();
    
    // Analyze observability of the entire system
    std::vector<ObservableSubsystem> analyzeObservability(
        const model::NetworkModel& network,
        const model::TelemetryData& telemetry);
    
    // Check if system is fully observable
    bool isFullyObservable(const model::NetworkModel& network,
                           const model::TelemetryData& telemetry);
    
    // Find minimum measurement placement for observability
    std::vector<BusId> findMinimumMeasurements(
        const model::NetworkModel& network);
    
    // Get observable buses
    std::set<BusId> getObservableBuses(const model::NetworkModel& network,
                                       const model::TelemetryData& telemetry);
    
    // Get non-observable buses
    std::set<BusId> getNonObservableBuses(const model::NetworkModel& network,
                                         const model::TelemetryData& telemetry);
    
private:
    // Graph-based observability checking
    void buildObservabilityGraph(const model::NetworkModel& network,
                                const model::TelemetryData& telemetry,
                                std::vector<std::vector<Index>>& graph);
    
    // Depth-first search for connected components
    void dfs(Index node, const std::vector<std::vector<Index>>& graph,
            std::vector<bool>& visited, std::set<BusId>& component);
};

} // namespace observability
} // namespace sle

#endif // SLE_OBSERVABILITY_OBSERVABILITYANALYZER_H

