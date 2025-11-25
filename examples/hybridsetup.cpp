/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Hybrid Setup Example - Comprehensive Real-Time State Estimation
 * 
 * This example demonstrates the most comprehensive hybrid approach for real-time state estimation:
 * 
 * Pre-Estimation Features:
 * - Pre-Validation: Data consistency checking to identify issues before estimation
 * - Observability Analysis: Check if system is fully observable
 * - Optimal Placement: Recommend best locations for new measurements (if needed)
 * 
 * Real-Time Features:
 * - Fast standard WLS for real-time updates (every cycle) - ~10-50 ms
 * - Periodic robust estimation to detect and handle bad data (every N cycles) - ~100-500 ms
 * - Bad data detection to identify problematic measurements (every M cycles) - ~20-100 ms
 * - Computed Values: Extract voltage, power, and current from robust estimation results
 * 
 * Strategy:
 * 1. Validate data consistency and check observability before starting
 * 2. Use fast WLS for every-cycle updates (provides real-time state)
 * 3. Periodically run robust estimation (handles bad data automatically)
 * 4. Frequently check for bad data (identifies problems quickly)
 * 5. Compute and display all estimated values (voltage, power, current)
 * 
 * Use cases:
 * - Real-time applications requiring fast updates (SCADA, EMS)
 * - Systems with occasional bad data that need automatic handling
 * - When you need both speed (WLS) and accuracy (robust estimation)
 * - Production systems requiring comprehensive validation and monitoring
 * - Systems needing optimal measurement placement recommendations
 * 
 * Performance:
 * - Standard WLS: Fast (2-5 iterations), suitable for every cycle
 * - Robust estimation: Slower (20+ IRLS iterations × 10 WLS iterations), use periodically
 * - Bad data detection: Fast (statistical tests), use frequently
 * - Value computation: GPU-accelerated, ~5-20x faster than CPU
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/math/RobustEstimator.h>
#include <sle/baddata/BadDataDetector.h>
#include <sle/baddata/DataConsistencyChecker.h>
#include <sle/observability/ObservabilityAnalyzer.h>
#include <sle/observability/OptimalPlacement.h>
#include <sle/io/OutputFormatter.h>
#include <sle/Types.h>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <cmath>
#include <vector>
#include <set>

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Hybrid Robust Estimation Setup ===\n\n";
        
        // ========================================================================
        // STEP 1: Load Network Model and Measurements
        // ========================================================================
        std::string networkFile = (argc > 1) ? argv[1] : "examples/ieee14/network.dat";
        std::string measurementFile = (argc > 2) ? argv[2] : "examples/ieee14/measurements.csv";
        
        std::cout << "Loading network model from: " << networkFile << "\n";
        auto networkUnique = sle::interface::ModelLoader::loadFromIEEE(networkFile);
        if (!networkUnique) {
            std::cerr << "ERROR: Failed to load network model\n";
            return 1;
        }
        auto network = std::shared_ptr<sle::model::NetworkModel>(std::move(networkUnique));
        std::cout << "  - Loaded " << network->getBusCount() << " buses, " 
                  << network->getBranchCount() << " branches\n";
        
        std::cout << "Loading measurements from: " << measurementFile << "\n";
        auto telemetryUnique = sle::interface::MeasurementLoader::loadTelemetry(
            measurementFile, *network);
        if (!telemetryUnique) {
            std::cerr << "ERROR: Failed to load telemetry data\n";
            return 1;
        }
        auto telemetry = std::shared_ptr<sle::model::TelemetryData>(std::move(telemetryUnique));
        std::cout << "  - Loaded " << telemetry->getMeasurementCount() << " measurements\n\n";
        
        // ========================================================================
        // STEP 2: Pre-Validation (Data Consistency Check)
        // ========================================================================
        std::cout << "=== Pre-Validation: Data Consistency Check ===\n";
        sle::baddata::DataConsistencyChecker consistencyChecker;
        auto consistency = consistencyChecker.checkConsistency(*telemetry, *network);
        if (!consistency.isConsistent) {
            std::cout << "⚠ Data consistency issues found:\n";
            for (const auto& issue : consistency.inconsistencies) {
                std::cout << "  - " << issue << "\n";
            }
            std::cout << "Proceeding with estimation (robust estimator will handle issues)...\n";
        } else {
            std::cout << "✓ Data consistency check passed\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 3: Observability Analysis
        // ========================================================================
        std::cout << "=== Observability Analysis ===\n";
        sle::observability::ObservabilityAnalyzer analyzer;
        bool observable = analyzer.isFullyObservable(*network, *telemetry);
        
        if (!observable) {
            std::cout << "⚠ System is NOT fully observable\n";
            auto nonObservable = analyzer.getNonObservableBuses(*network, *telemetry);
            std::cout << "  Non-observable buses: ";
            for (auto busId : nonObservable) {
                std::cout << busId << " ";
            }
            std::cout << "\n";
            std::cout << "  Consider adding measurements or using pseudo measurements\n";
            
            // ====================================================================
            // STEP 3a: Optimal Measurement Placement (if not observable)
            // ====================================================================
            std::cout << "\n=== Optimal Measurement Placement ===\n";
            sle::observability::OptimalPlacement placement;
            
            // Get existing measurement locations
            std::set<sle::BusId> existingBuses;
            const auto& measurements = telemetry->getMeasurements();
            for (const auto& meas : measurements) {
                if (meas->getLocation() >= 0) {
                    existingBuses.insert(static_cast<sle::BusId>(meas->getLocation()));
                }
            }
            
            // Find optimal placement for 3-5 new measurements
            int numNewMeasurements = std::min(5, static_cast<int>(nonObservable.size()));
            if (numNewMeasurements > 0) {
                auto placements = placement.findOptimalPlacement(*network, existingBuses, numNewMeasurements);
                std::cout << "✓ Recommended " << placements.size() << " new measurement placements:\n";
                for (const auto& p : placements) {
                    std::cout << "  - Bus " << p.busId << " (type: " 
                              << static_cast<int>(p.type) << ")\n";
                }
            } else {
                std::cout << "  (No additional placements needed)\n";
            }
        } else {
            std::cout << "✓ System is fully observable\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 4: Configure Standard WLS Estimator (Fast)
        // ========================================================================
        std::cout << "=== Configuring Standard WLS Estimator ===\n";
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(network);
        estimator.setTelemetryData(telemetry);
        
        // Real-time mode: Fast, relaxed tolerance
        estimator.configureForRealTime(1e-5, 15, true);
        std::cout << "✓ Standard WLS configured for fast real-time updates\n\n";
        
        // ========================================================================
        // STEP 5: Configure Robust Estimator (Accurate)
        // ========================================================================
        std::cout << "=== Configuring Robust Estimator ===\n";
        sle::math::RobustEstimator robustEstimator;
        sle::math::RobustEstimatorConfig robustConfig;
        robustConfig.weightFunction = sle::math::RobustWeightFunction::HUBER;
        robustConfig.tuningConstant = 1.345;  // Standard Huber tuning constant
        robustConfig.tolerance = 1e-6;
        robustConfig.maxIterations = 20;  // IRLS iterations (each runs WLS internally)
        robustConfig.useGPU = true;
        robustEstimator.setConfig(robustConfig);
        std::cout << "✓ Robust estimator configured (Huber M-estimator)\n\n";
        
        // ========================================================================
        // STEP 6: Configure Bad Data Detector
        // ========================================================================
        std::cout << "=== Configuring Bad Data Detector ===\n";
        sle::baddata::BadDataDetector badDataDetector;
        badDataDetector.setNormalizedResidualThreshold(3.0);  // 3-sigma rule
        std::cout << "✓ Bad data detector configured\n\n";
        
       
        // ========================================================================
        // STEP 8: Initial Full Estimation (Standard WLS)
        // ========================================================================
        std::cout << "=== Initial Estimation (Standard WLS) ===\n";
        auto startTime = std::chrono::high_resolution_clock::now();
        auto result = estimator.estimate();
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime).count();
        
        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
        std::cout << "Time: " << duration << " ms\n\n";
        
        // ========================================================================
        // STEP 9: Configure Hybrid Strategy Intervals
        // ========================================================================
        const int ROBUST_CHECK_INTERVAL = 5;      // Run robust estimation every 5 cycles
        const int BAD_DATA_CHECK_INTERVAL = 3;    // Check for bad data every 3 cycles
        int cycleCount = 0;
        
        std::cout << "=== Hybrid Strategy Configuration ===\n";
        std::cout << "  - Fast WLS: Every cycle (real-time updates)\n";
        std::cout << "  - Robust estimation: Every " << ROBUST_CHECK_INTERVAL << " cycles\n";
        std::cout << "  - Bad data detection: Every " << BAD_DATA_CHECK_INTERVAL << " cycles\n\n";
        
        // ========================================================================
        // STEP 10: Real-Time Update Loop with Hybrid Strategy
        // ========================================================================
        std::cout << "=== Real-Time Update Loop ===\n";
        const int NUM_CYCLES = 20;
        
        for (int i = 0; i < NUM_CYCLES; ++i) {
            cycleCount++;
            
            // Simulate telemetry update
            sle::interface::TelemetryUpdate update;
            update.deviceId = "METER_" + std::to_string(i % 5);
            update.type = sle::MeasurementType::P_INJECTION;
            // Occasionally inject bad data for demonstration
            if (i == 8) {
                update.value = 10.0;  // Bad data: unrealistic value
                std::cout << "Cycle " << i << ": Injecting bad data (value = 10.0)\n";
            } else {
                update.value = 1.0 + (i * 0.05);  // Normal data
            }
            update.stdDev = 0.01;
            update.busId = (i % network->getBusCount()) + 1;
            update.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            estimator.getTelemetryData()->updateMeasurement(update);
            
            // ====================================================================
            // STEP 8a: Fast Real-Time Estimation (Standard WLS) - Every Cycle
            // ====================================================================
            startTime = std::chrono::high_resolution_clock::now();
            auto wlsResult = estimator.estimateIncremental();
            endTime = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                endTime - startTime).count();
            
            std::cout << "Cycle " << i << " [WLS]: " << wlsResult.message
                      << " (iterations: " << wlsResult.iterations 
                      << ", time: " << duration << " ms)\n";
            
            // ====================================================================
            // STEP 8b: Bad Data Detection - Every BAD_DATA_CHECK_INTERVAL Cycles
            // ====================================================================
            if (cycleCount % BAD_DATA_CHECK_INTERVAL == 0 && wlsResult.state) {
                auto badDataResult = badDataDetector.detectBadData(
                    *telemetry, *wlsResult.state, *network);
                
                if (badDataResult.hasBadData) {
                    std::cout << "  [Bad Data] Detected " << badDataResult.badDeviceIds.size() 
                              << " bad measurements\n";
                } else {
                    std::cout << "  [Bad Data] No bad data detected\n";
                }
            }
            
            // ====================================================================
            // STEP 10c: Robust Estimation - Every ROBUST_CHECK_INTERVAL Cycles
            // ====================================================================
            if (cycleCount % ROBUST_CHECK_INTERVAL == 0 && wlsResult.state) {
                std::cout << "  [Robust] Running robust estimation...\n";
                startTime = std::chrono::high_resolution_clock::now();
                auto robustState = std::make_unique<sle::model::StateVector>(*wlsResult.state);
                auto robustResult = robustEstimator.estimate(
                    *robustState, *network, *telemetry);
                endTime = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    endTime - startTime).count();
                
                std::cout << "  [Robust] " << robustResult.message
                          << " (iterations: " << robustResult.iterations 
                          << ", time: " << duration << " ms)\n";
                
                // ================================================================
                // STEP 10d: Compute Values from Robust Estimation
                // ================================================================
                if (robustResult.state && robustResult.converged) {
                    std::cout << "  [Values] Computed from robust estimation:\n";
                    std::cout << std::fixed << std::setprecision(4);
                    
                    // Show sample values (first 3 buses and branches)
                    auto buses = network->getBuses();
                    int busCount = 0;
                    for (auto* bus : buses) {
                        if (bus && busCount < 3) {
                            std::cout << "    Bus " << bus->getId() << ": V=" 
                                      << bus->getVPU() << " p.u., P=" 
                                      << bus->getPInjectionMW() << " MW, Q=" 
                                      << bus->getQInjectionMVAR() << " MVAR\n";
                            busCount++;
                        }
                    }
                    
                    auto branches = network->getBranches();
                    int branchCount = 0;
                    for (auto* branch : branches) {
                        if (branch && branchCount < 3) {
                            std::cout << "    Branch " << branch->getId() << ": P=" 
                                      << branch->getPMW() << " MW, Q=" 
                                      << branch->getQMVAR() << " MVAR, I=" 
                                      << branch->getIAmps() << " A\n";
                            branchCount++;
                        }
                    }
                    if (busCount > 0 || branchCount > 0) {
                        std::cout << "    ... (showing first 3 buses and branches)\n";
                    }
                }
            }
            
            // Simulate real-time delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // ========================================================================
        // STEP 11: Final Full Estimation and Computed Values
        // ========================================================================
        std::cout << "\n=== Final Full Estimation ===\n";
        result = estimator.estimate();
        std::cout << "✓ Final estimation: " << result.message 
                  << " (iterations: " << result.iterations << ")\n";
        
        // Compute all final values
        if (result.state && result.converged) {
            std::cout << "\n=== Computing Final Estimated Values ===\n";
            std::cout << "✓ Final estimated values ready (solver already populated bus/branch metrics)\n";
            
            // Display summary of computed values
            std::cout << "\n=== Computed Values Summary ===\n";
            std::cout << std::fixed << std::setprecision(4);
            
            // Voltage summary
            auto buses = network->getBuses();
            std::cout << "Voltage Estimates (sample):\n";
            std::cout << "Bus ID | V (p.u.) | V (kV)  | P (MW)   | Q (MVAR)\n";
            std::cout << "-------|----------|---------|----------|----------\n";
            int displayCount = 0;
            for (auto* bus : buses) {
                if (bus && displayCount < 5) {
                    std::cout << std::setw(6) << bus->getId() << " | "
                              << std::setw(8) << bus->getVPU() << " | "
                              << std::setw(7) << bus->getVKV() << " | "
                              << std::setw(9) << bus->getPInjectionMW() << " | "
                              << std::setw(9) << bus->getQInjectionMVAR() << "\n";
                    displayCount++;
                }
            }
            if (displayCount < static_cast<int>(buses.size())) {
                std::cout << "... (" << (buses.size() - displayCount) << " more buses)\n";
            }
            
            // Power flow summary
            auto branches = network->getBranches();
            std::cout << "\nPower Flows (sample):\n";
            std::cout << "Branch | From | To  | P (MW)   | Q (MVAR) | I (A)\n";
            std::cout << "-------|------|-----|----------|----------|-------\n";
            displayCount = 0;
            for (auto* branch : branches) {
                if (branch && displayCount < 5) {
                    std::cout << std::setw(6) << branch->getId() << " | "
                              << std::setw(4) << branch->getFromBus() << " | "
                              << std::setw(3) << branch->getToBus() << " | "
                              << std::setw(9) << branch->getPMW() << " | "
                              << std::setw(9) << branch->getQMVAR() << " | "
                              << std::setw(6) << branch->getIAmps() << "\n";
                    displayCount++;
                }
            }
            if (displayCount < static_cast<int>(branches.size())) {
                std::cout << "... (" << (branches.size() - displayCount) << " more branches)\n";
            }
            std::cout << "\n";
        }
        
        std::cout << "\n=== Hybrid Setup Completed Successfully ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

