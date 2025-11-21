/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/multiarea/MultiAreaEstimator.h>
#include <sle/interface/StateEstimator.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/StateVector.h>
#include <sle/Types.h>
#include <set>
#include <string>
#include <map>

namespace sle {
namespace multiarea {

MultiAreaEstimator::MultiAreaEstimator() = default;

void MultiAreaEstimator::addZone(const Zone& zone) {
    zones_.push_back(zone);
    auto estimator = std::make_shared<interface::StateEstimator>();
    estimator->setNetwork(zone.network);
    estimator->setTelemetryData(zone.telemetry);
    zoneEstimators_[zone.name] = estimator;
}

void MultiAreaEstimator::addArea(const Area& area) {
    areas_.push_back(area);
    
    // Create estimator for this area
    auto estimator = std::make_shared<interface::StateEstimator>();
    estimator->setNetwork(area.network);
    estimator->setTelemetryData(area.telemetry);
    
    areaEstimators_[area.name] = estimator;
}

void MultiAreaEstimator::addRegion(const Region& region) {
    regions_.push_back(region);
    
    auto estimator = std::make_shared<interface::StateEstimator>();
    estimator->setNetwork(region.network);
    estimator->setTelemetryData(region.telemetry);
    
    regionEstimators_[region.name] = estimator;
}

void MultiAreaEstimator::setTieLineMeasurements(
    const std::vector<TieLineMeasurement>& measurements) {
    tieLineMeasurements_ = measurements;
}

MultiAreaEstimator::MultiAreaResult MultiAreaEstimator::estimate() {
    // Use hierarchical estimation by default
    return estimateHierarchical();
}

MultiAreaEstimator::MultiAreaResult MultiAreaEstimator::estimateHierarchical() {
    MultiAreaResult result;
    result.converged = true;
    result.totalIterations = 0;
    
    // Estimate each area independently
    for (const auto& area : areas_) {
        auto it = areaEstimators_.find(area.name);
        if (it != areaEstimators_.end()) {
            auto areaResult = it->second->estimate();
            // Access values before move
            result.totalIterations += areaResult.iterations;
            if (!areaResult.converged) {
                result.converged = false;
            }
            // Use emplace with move to avoid copying unique_ptr
            result.areaResults.emplace(area.name, std::move(areaResult));
        }
    }
    
    // Coordinate tie line flows
    coordinateTieLineFlows();
    
    result.message = result.converged ? "All areas converged" : "Some areas did not converge";
    
    return result;
}

MultiAreaEstimator::MultiAreaResult MultiAreaEstimator::estimateDistributed(
    int maxCoordinationIterations) {
    
    MultiAreaResult result;
    result.converged = false;
    result.totalIterations = 0;
    
    // Initial estimation
    auto initialResult = estimateHierarchical();
    // Move results instead of copying (StateEstimationResult contains unique_ptr)
    result.areaResults = std::move(initialResult.areaResults);
    
    // Iterative coordination
    for (int coordIter = 0; coordIter < maxCoordinationIterations; ++coordIter) {
        bool converged = true;
        
        // Update boundary conditions and re-estimate
        for (const auto& area : areas_) {
            // Get neighbor states and update boundary
            for (const auto& tieLine : tieLineMeasurements_) {
                if (area.tieLines.count(tieLine.branchId) > 0) {
                    // Find neighbor area
                    for (const auto& neighborArea : areas_) {
                        if (neighborArea.name != area.name &&
                            neighborArea.tieLines.count(tieLine.branchId) > 0) {
                            auto neighborState = getAreaState(neighborArea.name);
                            if (neighborState) {
                                updateBoundaryConditions(area.name, *neighborState);
                            }
                        }
                    }
                }
            }
            
            // Re-estimate area
            auto it = areaEstimators_.find(area.name);
            if (it != areaEstimators_.end()) {
                auto areaResult = it->second->estimateIncremental();
                // Access values before move
                if (!areaResult.converged) {
                    converged = false;
                }
                // Use emplace with move to avoid copying unique_ptr
                result.areaResults.emplace(area.name, std::move(areaResult));
            }
        }
        
        result.totalIterations++;
        
        if (converged) {
            result.converged = true;
            result.message = "Distributed estimation converged";
            break;
        }
    }
    
    if (!result.converged) {
        result.message = "Distributed estimation did not converge";
    }
    
    return result;
}

std::shared_ptr<model::StateVector> MultiAreaEstimator::getAreaState(
    const std::string& areaName) {
    
    auto it = areaEstimators_.find(areaName);
    if (it != areaEstimators_.end()) {
        return it->second->getCurrentState();
    }
    return nullptr;
}

std::shared_ptr<model::StateVector> MultiAreaEstimator::getZoneState(
    const std::string& zoneName) {
    auto it = zoneEstimators_.find(zoneName);
    if (it != zoneEstimators_.end()) {
        return it->second->getCurrentState();
    }
    return nullptr;
}

std::shared_ptr<model::StateVector> MultiAreaEstimator::getRegionState(
    const std::string& regionName) {
    auto it = regionEstimators_.find(regionName);
    if (it != regionEstimators_.end()) {
        return it->second->getCurrentState();
    }
    return nullptr;
}

void MultiAreaEstimator::coordinateTieLineFlows() {
    // Coordinate tie line power flows between areas
    // Ensure consistency at boundaries
    
    for (const auto& tieLine : tieLineMeasurements_) {
        // Find areas connected by this tie line
        std::vector<std::string> connectedAreas;
        
        for (const auto& area : areas_) {
            if (area.tieLines.count(tieLine.branchId) > 0) {
                connectedAreas.push_back(area.name);
            }
        }
        
        if (connectedAreas.size() >= 2) {
            // Coordinate flow measurements
            // Simplified - would properly coordinate boundary states
        }
    }
}

void MultiAreaEstimator::updateBoundaryConditions(
    const std::string& areaName,
    const model::StateVector& neighborState) {
    (void)areaName;  // May be used in future implementation
    (void)neighborState;  // May be used in future implementation
    
    // Update boundary bus voltages/angles based on neighbor state
    // This would update the area's network model with boundary conditions
    
    auto it = areaEstimators_.find(areaName);
    if (it != areaEstimators_.end()) {
        // Update boundary conditions in the area's network model
        // Simplified - would properly update boundary buses
    }
}

void MultiAreaEstimator::updateZoneBoundaryConditions(
    const std::string& zoneName,
    const model::StateVector& neighborState) {
    (void)zoneName;
    (void)neighborState;
}

void MultiAreaEstimator::updateRegionBoundaryConditions(
    const std::string& regionName,
    const model::StateVector& neighborState) {
    (void)regionName;
    (void)neighborState;
}

MultiAreaEstimator::HierarchyLevel MultiAreaEstimator::getHierarchyLevel(
    const std::string& name) const {
    if (zoneEstimators_.count(name)) {
        return HierarchyLevel::ZONE;
    }
    if (areaEstimators_.count(name)) {
        return HierarchyLevel::AREA;
    }
    return HierarchyLevel::REGION;
}

} // namespace multiarea
} // namespace sle

