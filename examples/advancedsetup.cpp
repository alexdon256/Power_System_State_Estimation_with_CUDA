/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Advanced Setup Example
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
#include <set>

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Advanced Features Setup ===\n\n";
        
        // ========================================================================
        // STEP 1: Load Network and Measurements
        // ========================================================================
        std::string networkFile = (argc > 1) ? argv[1] : "examples/ieee14/network.dat";
        std::string measurementFile = (argc > 2) ? argv[2] : "examples/ieee14/measurements.csv";
        
        std::cout << "Loading network model from: " << networkFile << "\n";
        auto network = sle::interface::ModelLoader::loadFromIEEE(networkFile);
        if (!network) {
            std::cerr << "ERROR: Failed to load network model\n";
            return 1;
        }
        std::cout << "  - Loaded " << network->getBusCount() << " buses, " 
                  << network->getBranchCount() << " branches\n";
        
        std::cout << "Loading measurements from: " << measurementFile << "\n";
        auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
            measurementFile, *network);
        if (!telemetry) {
            std::cerr << "ERROR: Failed to load telemetry data\n";
            return 1;
        }
        std::cout << "  - Loaded " << telemetry->getMeasurementCount() << " measurements\n\n";
        
        // ========================================================================
        // STEP 2: Robust Estimation (M-estimators)
        // ========================================================================
        std::cout << "=== Robust Estimation ===\n";
        sle::math::RobustEstimator robustEstimator;
        
        // Configure Huber M-estimator
        sle::math::RobustEstimatorConfig config;
        config.weightFunction = sle::math::RobustWeightFunction::HUBER;
        config.tuningConstant = 1.345;  // Standard Huber tuning constant
        config.tolerance = 1e-6;
        config.maxIterations = 50;
        config.useGPU = true;
        robustEstimator.setConfig(config);
        
        std::cout << "✓ Robust estimator configured (Huber M-estimator)\n";
        std::cout << "  Available M-estimators: HUBER, BISQUARE, CAUCHY, WELSCH\n\n";
        
        // ========================================================================
        // STEP 3: Load Flow
        // ========================================================================
        std::cout << "=== Load Flow ===\n";
        sle::math::LoadFlow loadflow;
        sle::math::LoadFlowConfig lfConfig;
        lfConfig.tolerance = 1e-6;
        lfConfig.maxIterations = 100;
        loadflow.setConfig(lfConfig);
        
        auto lfResult = loadflow.solve(*network);
        std::cout << "✓ Load flow: " << (lfResult.converged ? "Converged" : "Failed") << "\n";
        std::cout << "  Final mismatch: " << lfResult.finalMismatch << "\n\n";
        
        // ========================================================================
        // STEP 4: Optimal Measurement Placement
        // ========================================================================
        std::cout << "=== Optimal Measurement Placement ===\n";
        sle::observability::OptimalPlacement placement;
        std::set<sle::BusId> existing = {1, 2, 3};  // Existing measurement locations
        
        auto placements = placement.findOptimalPlacement(*network, existing, 5);
        std::cout << "✓ Recommended " << placements.size() << " new measurement placements:\n";
        for (const auto& p : placements) {
            std::cout << "  - Bus " << p.busId << " (type: " 
                      << static_cast<int>(p.type) << ")\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 5: Transformer Tap Ratio Configuration
        // ========================================================================
        std::cout << "=== Transformer Configuration ===\n";
        auto branches = network->getBranches();
        int transformerCount = 0;
        
        for (auto* branch : branches) {
            // Example: Configure transformer between bus 4 and 5 (if exists)
            if (branch->getFromBus() == 4 && branch->getToBus() == 5) {
                branch->setTapRatio(1.05);  // 5% boost transformer
                branch->setPhaseShift(0.0);
                std::cout << "✓ Branch " << branch->getId() << ": Tap ratio = " 
                          << branch->getTapRatio() << "\n";
                transformerCount++;
            }
            // Example: Phase-shifting transformer
            else if (branch->getFromBus() == 6 && branch->getToBus() == 7) {
                branch->setTapRatio(1.0);
                branch->setPhaseShift(0.1);  // 0.1 radians phase shift
                std::cout << "✓ Branch " << branch->getId() << ": Phase shift = " 
                          << branch->getPhaseShift() << " rad\n";
                transformerCount++;
            }
        }
        
        if (transformerCount == 0) {
            std::cout << "  (No transformers configured in this example)\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 6: PMU Support
        // ========================================================================
        std::cout << "=== PMU Support ===\n";
        std::cout << "✓ PMU support available (C37.118 compliant)\n";
        std::cout << "  Supported measurement types:\n";
        std::cout << "    - V_PHASOR: Voltage phasor (magnitude and angle)\n";
        std::cout << "    - I_PHASOR: Current phasor (magnitude and angle)\n";
        std::cout << "  PMU data can be loaded from binary C37.118 format\n";
        std::cout << "  or converted from PMU frames to measurements\n\n";
        
        // ========================================================================
        // STEP 7: Multi-Area Estimation
        // ========================================================================
        std::cout << "=== Multi-Area Estimation ===\n";
        sle::multiarea::MultiAreaEstimator multiArea;
        
        // Create zones (lowest level)
        sle::multiarea::Zone zone1;
        zone1.name = "NorthZone";
        zone1.areaName = "PJM";
        zone1.buses = {1, 2, 3, 4, 5};
        multiArea.addZone(zone1);
        
        sle::multiarea::Zone zone2;
        zone2.name = "SouthZone";
        zone2.areaName = "PJM";
        zone2.buses = {6, 7, 8, 9, 10};
        multiArea.addZone(zone2);
        
        // Create areas (middle level)
        sle::multiarea::Area area1;
        area1.name = "PJM";
        area1.regionName = "Eastern";
        area1.zones = {"NorthZone", "SouthZone"};
        multiArea.addArea(area1);
        
        // Create regions (highest level)
        sle::multiarea::Region region1;
        region1.name = "Eastern";
        region1.areas = {"PJM"};
        multiArea.addRegion(region1);
        
        std::cout << "✓ Multi-area hierarchy configured:\n";
        std::cout << "  - Region: Eastern\n";
        std::cout << "    - Area: PJM\n";
        std::cout << "      - Zone: NorthZone (" << zone1.buses.size() << " buses)\n";
        std::cout << "      - Zone: SouthZone (" << zone2.buses.size() << " buses)\n";
        std::cout << "  Hierarchical estimation available\n\n";
        
        // ========================================================================
        // STEP 8: Run State Estimation with Advanced Features
        // ========================================================================
        std::cout << "=== Running State Estimation ===\n";
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));
        estimator.setTelemetryData(telemetry);
        estimator.configureForOffline(1e-8, 50, true);
        
        auto result = estimator.estimate();
        std::cout << "✓ Estimation: " << (result.converged ? "Converged" : "Failed") << "\n";
        std::cout << "  Iterations: " << result.iterations << "\n";
        std::cout << "  Final norm: " << result.finalNorm << "\n\n";
        
        // ========================================================================
        // STEP 9: Compute and Extract Values
        // ========================================================================
        std::cout << "=== Computing Estimated Values ===\n";
        if (result.state) {
            bool useGPU = true;
            network->computeVoltEstimates(*result.state, useGPU);
            network->computePowerInjections(*result.state, useGPU);
            network->computePowerFlows(*result.state, useGPU);
            std::cout << "✓ All values computed and stored in Bus/Branch objects\n\n";
        }
        
        // ========================================================================
        // STEP 10: Summary
        // ========================================================================
        std::cout << "=== Advanced Features Summary ===\n";
        std::cout << "✓ Robust estimation: Available (Huber, Bi-square, Cauchy, Welsch)\n";
        std::cout << "✓ Load flow: Available for initial state or validation\n";
        std::cout << "✓ Optimal placement: Available for planning studies\n";
        std::cout << "✓ Transformer modeling: Tap ratios and phase shifts supported\n";
        std::cout << "✓ PMU support: C37.118 compliant phasor measurements\n";
        std::cout << "✓ Multi-area: 3-level hierarchy (Region → Area → Zone)\n";
        std::cout << "\n=== Advanced Setup Completed Successfully ===\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

