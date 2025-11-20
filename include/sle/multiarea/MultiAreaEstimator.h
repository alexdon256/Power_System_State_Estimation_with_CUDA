/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MULTIAREA_MULTIAREAESTIMATOR_H
#define SLE_MULTIAREA_MULTIAREAESTIMATOR_H

#include <sle/Export.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/interface/StateEstimator.h>
#include <sle/Types.h>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>

namespace sle {
namespace multiarea {

// Zone definition (lowest level: subdivisions within an area)
// Zones are typically transmission zones, load zones, or operational subdivisions
// within a control area. Used for fine-grained parallelization and organization.
struct SLE_API Zone {
    std::string name;                    // Zone identifier (e.g., "Zone1", "NorthZone")
    std::string areaName;                // Parent area this zone belongs to
    std::set<BusId> buses;               // Buses in this zone
    std::set<BranchId> internalBranches; // Branches within the zone
    std::set<BranchId> tieLines;         // Branches connecting to other zones/areas
    std::shared_ptr<model::NetworkModel> network;
    std::shared_ptr<model::TelemetryData> telemetry;
};

// Area definition (middle level: control areas, ISOs, RTOs)
// Areas are typically independent system operators (ISOs) or control areas
// within a region. Examples: PJM, NYISO, CAISO, ERCOT
struct SLE_API Area {
    std::string name;                    // Area identifier (e.g., "PJM", "NYISO")
    std::string regionName;              // Parent region this area belongs to (optional)
    std::set<BusId> buses;               // Buses in this area
    std::set<BranchId> internalBranches; // Branches within the area
    std::set<BranchId> tieLines;         // Branches connecting to other areas/regions
    std::shared_ptr<model::NetworkModel> network;
    std::shared_ptr<model::TelemetryData> telemetry;
    std::vector<Zone> zones;             // Optional: zones within this area
};

// Region definition (highest level: large-scale interconnections)
// Regions are typically major power system interconnections spanning multiple
// control areas. Examples: Eastern Interconnection, Western Interconnection
struct SLE_API Region {
    std::string name;                    // Region identifier (e.g., "Eastern", "Western")
    std::set<BusId> buses;              // All buses in the region
    std::set<BranchId> internalBranches; // Branches within the region
    std::set<BranchId> tieLines;         // Branches connecting to other regions
    std::shared_ptr<model::NetworkModel> network;
    std::shared_ptr<model::TelemetryData> telemetry;
    std::vector<Area> areas;             // Areas within this region
};

// Tie line measurement
struct SLE_API TieLineMeasurement {
    BranchId branchId;
    BusId fromArea;
    BusId toArea;
    MeasurementType type;
    Real value;
    Real stdDev;
};

class SLE_API MultiAreaEstimator {
public:
    MultiAreaEstimator();
    
    // Add zone (lowest level: within an area)
    // Zones are optional - use for fine-grained organization within areas
    void addZone(const Zone& zone);
    
    // Add area (middle level: control areas, ISOs)
    // Areas can be used independently or as part of a region
    void addArea(const Area& area);
    
    // Add region (highest level: large interconnections)
    // Regions contain multiple areas, which may contain zones
    void addRegion(const Region& region);
    
    // Set tie line measurements
    // Tie lines connect different areas/zones/regions
    void setTieLineMeasurements(const std::vector<TieLineMeasurement>& measurements);
    
    // Run multi-area state estimation
    struct SLE_API MultiAreaResult {
        bool converged;
        int totalIterations;
        std::map<std::string, interface::StateEstimationResult> areaResults;
        std::map<std::string, interface::StateEstimationResult> zoneResults;  // Zone-level results
        std::map<std::string, interface::StateEstimationResult> regionResults; // Region-level results
        std::vector<Real> tieLineFlows;
        std::string message;
    };
    
    MultiAreaResult estimate();
    
    // Hierarchical estimation (estimate independently, then coordinate)
    // For 3-level hierarchy: estimate zones → areas → regions
    // For 2-level: estimate zones → areas, or areas → regions
    // For 1-level: estimate areas only
    MultiAreaResult estimateHierarchical();
    
    // Distributed estimation (iterative coordination)
    // Iteratively coordinates between levels until convergence
    MultiAreaResult estimateDistributed(int maxCoordinationIterations = 10);
    
    // Get state at different hierarchy levels
    std::shared_ptr<model::StateVector> getZoneState(const std::string& zoneName);
    std::shared_ptr<model::StateVector> getAreaState(const std::string& areaName);
    std::shared_ptr<model::StateVector> getRegionState(const std::string& regionName);
    
private:
    // Storage for all hierarchy levels
    std::vector<Zone> zones_;           // Zones (lowest level)
    std::vector<Area> areas_;           // Areas (middle level)
    std::vector<Region> regions_;       // Regions (highest level)
    
    std::vector<TieLineMeasurement> tieLineMeasurements_;
    
    // Estimators for each hierarchy level
    std::map<std::string, std::shared_ptr<interface::StateEstimator>> zoneEstimators_;
    std::map<std::string, std::shared_ptr<interface::StateEstimator>> areaEstimators_;
    std::map<std::string, std::shared_ptr<interface::StateEstimator>> regionEstimators_;
    
    // Coordinate tie line flows between hierarchy levels
    void coordinateTieLineFlows();
    
    // Update boundary conditions at different levels
    void updateBoundaryConditions(const std::string& areaName,
                                  const model::StateVector& neighborState);
    void updateZoneBoundaryConditions(const std::string& zoneName,
                                      const model::StateVector& neighborState);
    void updateRegionBoundaryConditions(const std::string& regionName,
                                        const model::StateVector& neighborState);
    
    // Helper: Determine hierarchy level from name
    enum class HierarchyLevel { ZONE, AREA, REGION };
    HierarchyLevel getHierarchyLevel(const std::string& name) const;
};

} // namespace multiarea
} // namespace sle

#endif // SLE_MULTIAREA_MULTIAREAESTIMATOR_H

