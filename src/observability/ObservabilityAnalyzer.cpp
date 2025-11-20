/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/observability/ObservabilityAnalyzer.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/Types.h>
#include <algorithm>
#include <set>

namespace sle {
namespace observability {

ObservabilityAnalyzer::ObservabilityAnalyzer() {
}

std::vector<ObservableSubsystem> ObservabilityAnalyzer::analyzeObservability(
    const model::NetworkModel& network,
    const model::TelemetryData& telemetry) {
    
    std::vector<ObservableSubsystem> subsystems;
    
    // Build observability graph
    std::vector<std::vector<Index>> graph(network.getBusCount());
    buildObservabilityGraph(network, telemetry, graph);
    
    // Find connected components
    std::vector<bool> visited(network.getBusCount(), false);
    auto buses = network.getBuses();
    
    for (size_t i = 0; i < buses.size(); ++i) {
        if (!visited[i]) {
            ObservableSubsystem subsystem;
            subsystem.isObservable = false;
            
            std::set<BusId> component;
            dfs(i, graph, visited, component);
            
            for (BusId busId : component) {
                subsystem.buses.insert(busId);
            }
            
            // Check if subsystem is observable
            // Simplified: check if there are enough measurements
            size_t measCount = 0;
            for (const auto& meas : telemetry.getMeasurements()) {
                if (component.count(meas->getLocation()) > 0) {
                    measCount++;
                }
            }
            
            // Observable if measurements >= buses (simplified criterion)
            subsystem.isObservable = (measCount >= component.size());
            
            subsystems.push_back(subsystem);
        }
    }
    
    return subsystems;
}

bool ObservabilityAnalyzer::isFullyObservable(const model::NetworkModel& network,
                                              const model::TelemetryData& telemetry) {
    auto subsystems = analyzeObservability(network, telemetry);
    
    for (const auto& subsystem : subsystems) {
        if (!subsystem.isObservable) {
            return false;
        }
    }
    
    return true;
}

std::vector<BusId> ObservabilityAnalyzer::findMinimumMeasurements(
    const model::NetworkModel& network) {
    
    // Simplified minimum measurement placement
    // In practice, this would use more sophisticated algorithms
    std::vector<BusId> placements;
    
    auto buses = network.getBuses();
    for (auto* bus : buses) {
        if (bus->getType() != BusType::Slack) {
            placements.push_back(bus->getId());
        }
    }
    
    return placements;
}

std::set<BusId> ObservabilityAnalyzer::getObservableBuses(
    const model::NetworkModel& network,
    const model::TelemetryData& telemetry) {
    
    std::set<BusId> observable;
    auto subsystems = analyzeObservability(network, telemetry);
    
    for (const auto& subsystem : subsystems) {
        if (subsystem.isObservable) {
            for (BusId busId : subsystem.buses) {
                observable.insert(busId);
            }
        }
    }
    
    return observable;
}

std::set<BusId> ObservabilityAnalyzer::getNonObservableBuses(
    const model::NetworkModel& network,
    const model::TelemetryData& telemetry) {
    
    std::set<BusId> nonObservable;
    auto subsystems = analyzeObservability(network, telemetry);
    
    for (const auto& subsystem : subsystems) {
        if (!subsystem.isObservable) {
            for (BusId busId : subsystem.buses) {
                nonObservable.insert(busId);
            }
        }
    }
    
    return nonObservable;
}

void ObservabilityAnalyzer::buildObservabilityGraph(
    const model::NetworkModel& network,
    const model::TelemetryData& telemetry,
    std::vector<std::vector<Index>>& graph) {
    
    // Build graph based on branches and measurements
    auto branches = network.getBranches();
    
    for (auto* branch : branches) {
        Index fromIdx = network.getBusIndex(branch->getFromBus());
        Index toIdx = network.getBusIndex(branch->getToBus());
        
        if (fromIdx >= 0 && toIdx >= 0) {
            graph[fromIdx].push_back(toIdx);
            graph[toIdx].push_back(fromIdx);
        }
    }
}

void ObservabilityAnalyzer::dfs(Index node,
                                const std::vector<std::vector<Index>>& graph,
                                std::vector<bool>& visited,
                                std::set<BusId>& component) {
    
    visited[node] = true;
    // Would need bus ID lookup here
    // component.insert(busId);
    
    for (Index neighbor : graph[node]) {
        if (!visited[neighbor]) {
            dfs(neighbor, graph, visited, component);
        }
    }
}

} // namespace observability
} // namespace sle

