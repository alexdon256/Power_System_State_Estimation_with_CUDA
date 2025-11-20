/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Advanced Features Example
 * 
 * This example demonstrates advanced features of the state estimation library:
 * 1. Robust Estimation: M-estimators (Huber, Bi-square, Cauchy, Welsch) for bad data handling
 * 2. Load Flow: Power flow solution for initial state or validation
 * 3. Optimal Measurement Placement: Find best locations for new meters
 * 4. Transformer Modeling: Tap ratios and phase shifts for accurate transformer representation
 * 5. PMU Support: Phasor Measurement Unit data integration
 * 6. Multi-Area Estimation: Hierarchical estimation for large interconnected systems
 * 
 * Use cases:
 * - Systems with bad data requiring robust estimation
 * - Planning studies for meter placement
 * - Systems with transformers requiring accurate modeling
 * - PMU integration for enhanced observability
 * - Large-scale systems requiring distributed estimation
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/math/RobustEstimator.h>
#include <sle/math/LoadFlow.h>
#include <sle/observability/OptimalPlacement.h>
#include <sle/multiarea/MultiAreaEstimator.h>
#include <sle/io/PMUData.h>
#include <sle/model/Branch.h>
#include <sle/model/NetworkModel.h>
#include <iostream>
#include <cmath>

int main() {
    try {
        std::cout << "=== Advanced Features Example ===\n\n";
        
        // ========================================================================
        // STEP 1: Load Network and Measurements
        // ========================================================================
        auto network = sle::interface::ModelLoader::loadFromIEEE("examples/ieee14/network.dat");
        auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
            "examples/ieee14/measurements.csv", *network);
        
        // ========================================================================
        // STEP 2: Robust Estimation (M-estimators)
        // ========================================================================
        // Robust estimation uses M-estimators to handle bad data automatically
        // Instead of rejecting bad measurements, it down-weights them iteratively
        //
        // M-estimators:
        // - Huber: Good for moderate outliers (tuning constant = 1.345)
        // - Bi-square: Good for severe outliers (tuning constant = 4.685)
        // - Cauchy: Very robust, slower convergence (tuning constant = 2.385)
        // - Welsch: Good for symmetric outliers (tuning constant = 2.985)
        //
        // Algorithm: Iteratively Reweighted Least Squares (IRLS)
        // 1. Solve WLS with current weights
        // 2. Compute residuals
        // 3. Update weights based on residual magnitude (large residuals → low weights)
        // 4. Repeat until convergence
        //
        // Performance: 2-5x slower than WLS, but handles bad data automatically
        std::cout << "1. Robust Estimation (Huber M-estimator)...\n";
        sle::math::RobustEstimator robust;
        sle::math::RobustEstimatorConfig robustConfig;
        robustConfig.weightFunction = sle::math::RobustWeightFunction::HUBER;  // M-estimator type
        robustConfig.tuningConstant = 1.345;  // Tuning constant (controls outlier sensitivity)
                                                // Lower = more robust (down-weights more aggressively)
                                                // Higher = less robust (closer to WLS)
        robust.setConfig(robustConfig);
        
        // Initialize state vector (voltage magnitudes and angles)
        sle::model::StateVector state(network->getBusCount());
        state.initializeFromNetwork(*network);  // Flat start or load flow solution
        
        // Run robust estimation
        auto robustResult = robust.estimate(state, *network, *telemetry);
        std::cout << "   Converged: " << (robustResult.converged ? "Yes" : "No") << "\n";
        std::cout << "   Iterations: " << robustResult.iterations << "\n\n";
        
        // ========================================================================
        // STEP 3: Load Flow Analysis
        // ========================================================================
        // Load flow solves the power flow equations to find steady-state operating point
        // Used for:
        // - Initial state for state estimation (better than flat start)
        // - Validation of state estimation results
        // - Planning studies
        //
        // Algorithm: Newton-Raphson method
        // Solves: P(θ,V) = P_specified, Q(θ,V) = Q_specified
        std::cout << "2. Load Flow Analysis...\n";
        sle::math::LoadFlow loadflow;
        sle::math::LoadFlowConfig lfConfig;
        lfConfig.tolerance = 1e-6;        // Convergence tolerance (power mismatch in p.u.)
        lfConfig.maxIterations = 100;     // Maximum Newton-Raphson iterations
        loadflow.setConfig(lfConfig);
        
        // Solve load flow
        auto lfResult = loadflow.solve(*network);
        std::cout << "   Converged: " << (lfResult.converged ? "Yes" : "No") << "\n";
        std::cout << "   Final mismatch: " << lfResult.finalMismatch << "\n\n";
        
        // ========================================================================
        // STEP 4: Optimal Measurement Placement
        // ========================================================================
        // Find optimal locations for new measurements to maximize:
        // - Observability (ensure system can be estimated)
        // - Redundancy (improve estimation accuracy)
        // - Bad data detection capability
        //
        // Algorithms: Greedy, Genetic Algorithm, or Integer Programming
        // Objective: Minimize number of new measurements while maximizing benefits
        std::cout << "3. Optimal Measurement Placement...\n";
        sle::observability::OptimalPlacement placement;
        std::set<sle::BusId> existing = {1, 2, 3};  // Existing measurement locations
        // Find optimal placement for 5 additional measurements
        auto placements = placement.findOptimalPlacement(*network, existing, 5);
        std::cout << "   Recommended placements: " << placements.size() << "\n";
        for (const auto& p : placements) {
            std::cout << "     Bus " << p.busId << " (" 
                      << static_cast<int>(p.type) << ")\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 5: Current and Power Measurements
        // ========================================================================
        // The library supports comprehensive power and current measurements:
        // 
        // Power Measurements:
        // - P_INJECTION, Q_INJECTION: Net power at buses (generation - load)
        // - P_FLOW, Q_FLOW: Power flow on transmission lines/transformers
        //
        // Current Measurements:
        // - I_MAGNITUDE: Current magnitude from current transformers (CTs)
        //   Format in CSV: I_MAGNITUDE,CT_001,1,2,0.25,0.02
        //                  (Type,DeviceId,BusId,FromBus,ToBus,Value,StdDev)
        // - I_PHASOR: Current phasor from PMUs (magnitude and angle)
        //
        // All measurement types are automatically processed during state estimation
        std::cout << "5. Current and Power Measurements...\n";
        std::cout << "   Supported measurement types:\n";
        std::cout << "   - Power: P_INJECTION, Q_INJECTION, P_FLOW, Q_FLOW\n";
        std::cout << "   - Current: I_MAGNITUDE, I_PHASOR (PMU)\n";
        std::cout << "   - Voltage: V_MAGNITUDE, V_PHASOR (PMU)\n\n";
        
        // ========================================================================
        // STEP 6: Transformer Tap Ratio Configuration
        // ========================================================================
        // Transformers require accurate modeling with tap ratios and phase shifts
        // Tap ratio: Voltage transformation ratio (V_to = tap × V_from)
        // Phase shift: Phase angle shift introduced by transformer
        //
        // Types:
        // - Regular transformer: tap ≠ 1.0, phase shift = 0.0
        // - Phase-shifting transformer: tap = 1.0, phase shift ≠ 0.0
        // - Transmission line: tap = 1.0, phase shift = 0.0
        //
        // Impact on state estimation:
        // - Affects measurement functions (power flow equations)
        // - Must be accurately modeled for correct estimation
        // - Tap ratios can be estimated if treated as state variables
        std::cout << "6. Transformer Tap Ratio Configuration...\n";
        std::cout << "   Configuring transformer branches with tap ratios...\n";
        
        // Get all branches and configure transformers
        auto branches = network->getBranches();
        int transformerCount = 0;
        for (auto* branch : branches) {
            // Example: Set tap ratio for transformer between bus 4 and 5 (if exists)
            if (branch->getFromBus() == 4 && branch->getToBus() == 5) {
                // Set tap ratio to 1.05 (5% boost transformer)
                // Tap ratio > 1.0 increases voltage at "to" bus relative to "from" bus
                // Typical range: 0.9 to 1.1 (10% variation)
                branch->setTapRatio(1.05);
                branch->setPhaseShift(0.0);  // No phase shift for regular transformer
                std::cout << "   Branch " << branch->getId() 
                          << " (Bus " << branch->getFromBus() 
                          << " -> Bus " << branch->getToBus() 
                          << "): Tap ratio = " << branch->getTapRatio() << "\n";
                transformerCount++;
            }
            // Example: Set phase-shifting transformer
            else if (branch->getFromBus() == 6 && branch->getToBus() == 7) {
                // Phase-shifting transformer: tap ratio = 1.0, phase shift = 0.1 rad (~5.7 degrees)
                // Phase shift controls power flow direction (used for flow control)
                // Typical range: -0.3 to +0.3 radians (-17° to +17°)
                branch->setTapRatio(1.0);
                branch->setPhaseShift(0.1);  // 0.1 radians phase shift
                std::cout << "   Branch " << branch->getId() 
                          << " (Bus " << branch->getFromBus() 
                          << " -> Bus " << branch->getToBus() 
                          << "): Phase shift = " << branch->getPhaseShift() 
                          << " rad (" << (branch->getPhaseShift() * 180.0 / 3.14159) << " deg)\n";
                transformerCount++;
            }
            
            // Check if branch is a transformer
            // A branch is a transformer if tap ≠ 1.0 OR phase shift ≠ 0.0
            if (branch->isTransformer()) {
                std::cout << "   Branch " << branch->getId() << " is a transformer:\n";
                std::cout << "     - Tap ratio: " << branch->getTapRatio() << "\n";
                std::cout << "     - Phase shift: " << branch->getPhaseShift() << " rad\n";
                std::cout << "     - Impedance: R = " << branch->getR() 
                          << " p.u., X = " << branch->getX() << " p.u.\n";
            }
        }
        std::cout << "   Total transformers configured: " << transformerCount << "\n";
        std::cout << "   Note: Tap ratio = 1.0 and phase shift = 0.0 indicates a transmission line\n";
        std::cout << "         Tap ratio ≠ 1.0 or phase shift ≠ 0.0 indicates a transformer\n\n";
        
        // ========================================================================
        // STEP 7: PMU (Phasor Measurement Unit) Support
        // ========================================================================
        // PMUs provide synchronized phasor measurements (voltage and current phasors)
        // Advantages:
        // - Direct measurement of voltage/current phasors (magnitude and angle)
        // - Synchronized measurements (GPS time synchronization)
        // - High update rate (30-120 Hz vs 1-10 Hz for SCADA)
        // - High accuracy (0.1% vs 1-2% for SCADA)
        //
        // PMU data format: IEEE C37.118 (binary or CSV)
        // Measurements: Voltage phasor (V∠θ), current phasor (I∠φ), frequency
        std::cout << "7. PMU Data Processing...\n";
        try {
            // Parse PMU data file (IEEE C37.118 format)
            auto pmuFrames = sle::io::pmu::PMUParser::parseFromFile("pmu_data.bin");
            std::cout << "   Parsed " << pmuFrames.size() << " PMU frames\n";
            if (!pmuFrames.empty()) {
                // Convert PMU frame to measurement format
                // PMU measurements are treated as V_MAGNITUDE and V_ANGLE measurements
                auto pmuMeas = sle::io::pmu::PMUParser::convertToMeasurement(pmuFrames[0], 1);
                std::cout << "   Voltage phasor: " << pmuMeas.voltagePhasor << "\n";
                std::cout << "   Frequency: " << pmuMeas.frequency << " Hz\n";
            }
        } catch (...) {
            std::cout << "   PMU file not found (expected)\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 8: Multi-Area State Estimation (3-Level Hierarchy)
        // ========================================================================
        // Multi-area estimation divides large systems into hierarchical levels:
        //
        // Hierarchy Levels:
        // 1. REGION (highest): Large-scale interconnections
        //    - Examples: Eastern Interconnection, Western Interconnection, ERCOT
        //    - Contains multiple control areas (ISOs/RTOs)
        //    - Used for inter-regional coordination
        //
        // 2. AREA (middle): Control areas, ISOs, RTOs
        //    - Examples: PJM, NYISO, CAISO, ERCOT
        //    - Independent system operators or control areas
        //    - Contains multiple zones (optional)
        //    - Used for inter-area coordination
        //
        // 3. ZONE (lowest): Subdivisions within an area
        //    - Examples: Transmission zones, load zones, operational subdivisions
        //    - Fine-grained organization within a control area
        //    - Used for parallel processing and detailed analysis
        //
        // Algorithm: Hierarchical estimation
        // 1. Estimate zones independently (parallel) - lowest level
        // 2. Coordinate zones within each area (exchange tie-line flows)
        // 3. Estimate areas independently (parallel) - middle level
        // 4. Coordinate areas within each region (exchange tie-line flows)
        // 5. Estimate regions independently (parallel) - highest level
        // 6. Coordinate regions (exchange inter-regional tie-line flows)
        // 7. Repeat until convergence
        //
        // Benefits:
        // - Scalability: Handle systems with 10,000+ buses
        // - Parallelization: Each zone/area/region can run on separate GPU/CPU
        // - Privacy: Different levels can be owned by different utilities
        // - Flexibility: Use 1-level (areas only), 2-level (zones+areas or areas+regions), or 3-level
        std::cout << "6. Multi-Area State Estimation (3-Level Hierarchy: Region → Area → Zone)...\n";
        sle::multiarea::MultiAreaEstimator multiArea;
        
        // ========================================================================
        // STEP 7a: Create Zones (Lowest Level)
        // ========================================================================
        // Zones are subdivisions within an area (e.g., transmission zones, load zones)
        // Used for fine-grained parallelization and organization
        std::cout << "   Creating zones (lowest level)...\n";
        
        // Zone 1: North Zone (buses 1-3) - within Area 1
        sle::multiarea::Zone zone1;
        zone1.name = "NorthZone";
        zone1.areaName = "PJM";  // Parent area
        for (sle::BusId i = 1; i <= 3; ++i) {
            zone1.buses.insert(i);
        }
        zone1.network = std::make_shared<sle::model::NetworkModel>(*network);
        zone1.telemetry = telemetry;
        multiArea.addZone(zone1);
        std::cout << "     - " << zone1.name << " (buses 1-3) in area " << zone1.areaName << "\n";
        
        // Zone 2: South Zone (buses 4-7) - within Area 1
        sle::multiarea::Zone zone2;
        zone2.name = "SouthZone";
        zone2.areaName = "PJM";  // Parent area
        for (sle::BusId i = 4; i <= 7; ++i) {
            zone2.buses.insert(i);
        }
        zone2.network = std::make_shared<sle::model::NetworkModel>(*network);
        zone2.telemetry = telemetry;
        multiArea.addZone(zone2);
        std::cout << "     - " << zone2.name << " (buses 4-7) in area " << zone2.areaName << "\n";
        
        // Zone 3: East Zone (buses 8-11) - within Area 2
        sle::multiarea::Zone zone3;
        zone3.name = "EastZone";
        zone3.areaName = "NYISO";  // Parent area
        for (sle::BusId i = 8; i <= 11; ++i) {
            zone3.buses.insert(i);
        }
        zone3.network = std::make_shared<sle::model::NetworkModel>(*network);
        zone3.telemetry = telemetry;
        multiArea.addZone(zone3);
        std::cout << "     - " << zone3.name << " (buses 8-11) in area " << zone3.areaName << "\n";
        
        // Zone 4: West Zone (buses 12-14) - within Area 2
        sle::multiarea::Zone zone4;
        zone4.name = "WestZone";
        zone4.areaName = "NYISO";  // Parent area
        for (sle::BusId i = 12; i <= 14; ++i) {
            zone4.buses.insert(i);
        }
        zone4.network = std::make_shared<sle::model::NetworkModel>(*network);
        zone4.telemetry = telemetry;
        multiArea.addZone(zone4);
        std::cout << "     - " << zone4.name << " (buses 12-14) in area " << zone4.areaName << "\n";
        
        // ========================================================================
        // STEP 7b: Create Areas (Middle Level)
        // ========================================================================
        // Areas are control areas, ISOs, or RTOs (e.g., PJM, NYISO, CAISO)
        // Each area can contain multiple zones (optional)
        std::cout << "   Creating areas (middle level)...\n";
        
        // Area 1: PJM (buses 1-7) - contains zones 1 and 2
        sle::multiarea::Area area1;
        area1.name = "PJM";
        area1.regionName = "Eastern";  // Parent region
        for (sle::BusId i = 1; i <= 7; ++i) {
            area1.buses.insert(i);
        }
        area1.network = std::make_shared<sle::model::NetworkModel>(*network);
        area1.telemetry = telemetry;
        // Add zones to area (optional - zones can be managed separately)
        area1.zones.push_back(zone1);
        area1.zones.push_back(zone2);
        multiArea.addArea(area1);
        std::cout << "     - " << area1.name << " (buses 1-7) in region " << area1.regionName 
                  << " with " << area1.zones.size() << " zones\n";
        
        // Area 2: NYISO (buses 8-14) - contains zones 3 and 4
        sle::multiarea::Area area2;
        area2.name = "NYISO";
        area2.regionName = "Eastern";  // Parent region
        for (sle::BusId i = 8; i <= 14; ++i) {
            area2.buses.insert(i);
        }
        area2.network = std::make_shared<sle::model::NetworkModel>(*network);
        area2.telemetry = telemetry;
        // Add zones to area (optional)
        area2.zones.push_back(zone3);
        area2.zones.push_back(zone4);
        multiArea.addArea(area2);
        std::cout << "     - " << area2.name << " (buses 8-14) in region " << area2.regionName 
                  << " with " << area2.zones.size() << " zones\n";
        
        // ========================================================================
        // STEP 7c: Create Region (Highest Level)
        // ========================================================================
        // Regions are large-scale interconnections (e.g., Eastern, Western, ERCOT)
        // Each region contains multiple areas, which may contain zones
        std::cout << "   Creating region (highest level)...\n";
        
        sle::multiarea::Region region1;
        region1.name = "Eastern";
        // All buses in the region (union of all area buses)
        for (sle::BusId i = 1; i <= 14; ++i) {
            region1.buses.insert(i);
        }
        region1.network = std::make_shared<sle::model::NetworkModel>(*network);
        region1.telemetry = telemetry;
        // Add areas to region (optional - areas can be managed separately)
        region1.areas.push_back(area1);
        region1.areas.push_back(area2);
        multiArea.addRegion(region1);
        std::cout << "     - " << region1.name << " Interconnection (buses 1-14) with " 
                  << region1.areas.size() << " areas\n";
        
        // ========================================================================
        // STEP 7d: Configure Tie-Line Measurements (Optional)
        // ========================================================================
        // Tie lines connect different hierarchy levels:
        // - Zone-to-zone: Within an area
        // - Area-to-area: Within a region
        // - Region-to-region: Between interconnections
        std::cout << "   Configuring tie-line measurements...\n";
        std::vector<sle::multiarea::TieLineMeasurement> tieLineMeasurements;
        
        // Example: Tie line between zones (bus 3 to bus 4)
        sle::multiarea::TieLineMeasurement tie1;
        tie1.branchId = 1;  // Example branch ID
        tie1.fromArea = 3;  // From bus (in NorthZone)
        tie1.toArea = 4;    // To bus (in SouthZone)
        tie1.type = sle::MeasurementType::P_FLOW;
        tie1.value = 0.5;   // Measured power flow (p.u.)
        tie1.stdDev = 0.01;
        tieLineMeasurements.push_back(tie1);
        
        // Example: Tie line between areas (bus 7 to bus 8)
        sle::multiarea::TieLineMeasurement tie2;
        tie2.branchId = 2;  // Example branch ID
        tie2.fromArea = 7;  // From bus (in PJM)
        tie2.toArea = 8;    // To bus (in NYISO)
        tie2.type = sle::MeasurementType::P_FLOW;
        tie2.value = 0.3;   // Measured power flow (p.u.)
        tie2.stdDev = 0.01;
        tieLineMeasurements.push_back(tie2);
        
        multiArea.setTieLineMeasurements(tieLineMeasurements);
        std::cout << "     - Configured " << tieLineMeasurements.size() << " tie-line measurements\n";
        
        // ========================================================================
        // STEP 7e: Run Hierarchical Multi-Area Estimation
        // ========================================================================
        // Hierarchical estimation:
        // 1. Estimate zones independently (parallel) - fastest, most parallel
        // 2. Coordinate zones within areas (exchange boundary conditions)
        // 3. Estimate areas independently (parallel) - medium speed
        // 4. Coordinate areas within regions (exchange boundary conditions)
        // 5. Estimate regions independently (parallel) - slowest, least parallel
        // 6. Coordinate regions (exchange inter-regional boundary conditions)
        // 7. Repeat until convergence
        std::cout << "   Running hierarchical multi-area estimation...\n";
        auto multiResult = multiArea.estimateHierarchical();
        
        std::cout << "   Results:\n";
        std::cout << "     - Converged: " << (multiResult.converged ? "Yes" : "No") << "\n";
        std::cout << "     - Total iterations: " << multiResult.totalIterations << "\n";
        std::cout << "     - Zones estimated: " << multiResult.zoneResults.size() << "\n";
        std::cout << "     - Areas estimated: " << multiResult.areaResults.size() << "\n";
        std::cout << "     - Regions estimated: " << multiResult.regionResults.size() << "\n";
        
        // Display zone-level results
        if (!multiResult.zoneResults.empty()) {
            std::cout << "     - Zone results:\n";
            for (const auto& [zoneName, result] : multiResult.zoneResults) {
                std::cout << "       * " << zoneName << ": " << result.message 
                          << " (" << result.iterations << " iterations)\n";
            }
        }
        
        // Display area-level results
        if (!multiResult.areaResults.empty()) {
            std::cout << "     - Area results:\n";
            for (const auto& [areaName, result] : multiResult.areaResults) {
                std::cout << "       * " << areaName << ": " << result.message 
                          << " (" << result.iterations << " iterations)\n";
            }
        }
        
        // Display region-level results
        if (!multiResult.regionResults.empty()) {
            std::cout << "     - Region results:\n";
            for (const auto& [regionName, result] : multiResult.regionResults) {
                std::cout << "       * " << regionName << ": " << result.message 
                          << " (" << result.iterations << " iterations)\n";
            }
        }
        
        std::cout << "\n";
        
        std::cout << "Advanced features example completed!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

