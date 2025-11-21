/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Production Real-Time Setup Example
 * 
 * This example demonstrates a complete production-ready real-time workflow:
 * 1. Load network model and measurements
 * 2. Pre-estimation validation (data consistency)
 * 3. Configure estimator for real-time operation
 * 4. Start real-time telemetry processing
 * 5. Run initial state estimation
 * 6. Real-time update loop with incremental estimation
 * 7. System monitoring (voltage violations, branch overloads)
 * 8. Bad data detection
 * 9. Compute and extract all estimated values
 * 10. Generate comprehensive reports
 * 
 * Features:
 * - Asynchronous telemetry processing (thread-safe update queue)
 * - Incremental estimation (faster convergence using previous state)
 * - Real-time measurement updates without full reload
 * - GPU acceleration for fast performance
 * - Comprehensive monitoring and violation detection
 * - Bad data detection and reporting
 * - Production-ready code structure
 * 
 * Use cases:
 * - SCADA systems with continuous telemetry streams
 * - Energy Management Systems (EMS) requiring real-time state
 * - PMU data integration with high update rates
 * - Dynamic systems with frequent measurement changes
 * - Production deployment with monitoring and reporting
 * 
 * Configuration: Real-time mode (fast, relaxed tolerance)
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/io/OutputFormatter.h>
#include <sle/io/ComparisonReport.h>
#include <sle/baddata/BadDataDetector.h>
#include <sle/baddata/DataConsistencyChecker.h>
#include <sle/Types.h>
#include <sle/model/Bus.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>

using sle::Real;

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Production Real-Time State Estimation Setup ===\n\n";
        
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
        auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
            measurementFile, *network);
        if (!telemetry) {
            std::cerr << "ERROR: Failed to load telemetry data\n";
            return 1;
        }
        std::cout << "  - Loaded " << telemetry->getMeasurementCount() << " measurements\n\n";
        auto telemetryShared = std::shared_ptr<sle::model::TelemetryData>(std::move(telemetry));
        
        // ========================================================================
        // STEP 2: Pre-Estimation Validation
        // ========================================================================
        std::cout << "=== Pre-Estimation Validation ===\n";
        
        // Data consistency check
        sle::baddata::DataConsistencyChecker consistencyChecker;
        auto consistency = consistencyChecker.checkConsistency(*telemetryShared, *network);
        if (!consistency.isConsistent) {
            std::cout << "⚠ Data consistency issues found:\n";
            for (const auto& issue : consistency.inconsistencies) {
                std::cout << "  - " << issue << "\n";
            }
            std::cout << "Proceeding with estimation...\n";
        } else {
            std::cout << "✓ Data consistency check passed\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 3: Configure Estimator for Real-Time Operation
        // ========================================================================
        std::cout << "=== Configuring Estimator ===\n";
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(network);
        estimator.setTelemetryData(telemetryShared);
        
        // Real-time mode: Fast, relaxed tolerance
        estimator.configureForRealTime(1e-5, 15, true);  // tolerance, maxIter, useGPU
        std::cout << "✓ Configured for real-time operation\n";
        std::cout << "  - Tolerance: 1e-5\n";
        std::cout << "  - Max iterations: 15\n";
        std::cout << "  - GPU acceleration: Enabled\n\n";
        
        // ========================================================================
        // STEP 4: Real-Time Telemetry Handling
        // ========================================================================
        std::cout << "=== Real-Time Telemetry Handling ===\n";
        std::cout << "Telemetry updates are applied immediately (single-threaded processing).\n\n";
        
        // ========================================================================
        // STEP 5: Initial Full Estimation
        // ========================================================================
        std::cout << "=== Initial Estimation ===\n";
        auto startTime = std::chrono::high_resolution_clock::now();
        auto result = estimator.estimate();
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime).count();
        
        if (!result.converged) {
            std::cerr << "ERROR: Initial estimation failed\n";
            std::cerr << "Message: " << result.message << "\n";
            return 1;
        }
        
        std::cout << "✓ Initial estimation converged\n";
        std::cout << "  - Iterations: " << result.iterations << "\n";
        std::cout << "  - Final norm: " << std::scientific << std::setprecision(3) 
                  << result.finalNorm << "\n";
        std::cout << "  - Computation time: " << duration << " ms\n\n";
        
        // ========================================================================
        // STEP 6: Real-Time Update Loop
        // ========================================================================
        std::cout << "=== Real-Time Update Loop ===\n";
        std::cout << "Simulating real-time telemetry updates...\n";
        std::cout << "  (In production, updates come from SCADA/PMU systems)\n\n";
        
        const int NUM_UPDATES = 10;
        const int UPDATE_INTERVAL_MS = 100;  // 10 Hz update rate
        int voltageViolations = 0;
        int overloads = 0;
        
        for (int i = 0; i < NUM_UPDATES; ++i) {
            // Simulate telemetry update (in production, this comes from SCADA/PMU)
            sle::interface::TelemetryUpdate update;
            update.deviceId = "METER_" + std::to_string(i % 5);
            update.type = sle::MeasurementType::P_INJECTION;
            update.value = 1.0 + (i * 0.05);  // Gradual change
            update.stdDev = 0.01;
            update.busId = (i % network->getBusCount()) + 1;
            update.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Update measurement in real-time (thread-safe)
            estimator.getTelemetryProcessor().updateMeasurement(update);
            
            // Run incremental estimation periodically
            // Incremental uses previous state as initial guess (faster convergence)
            if (i % 5 == 0) {
                startTime = std::chrono::high_resolution_clock::now();
                auto incResult = estimator.estimateIncremental();
                endTime = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    endTime - startTime).count();
                
                std::cout << "Update " << i << ": " << incResult.message
                          << " (iterations: " << incResult.iterations 
                          << ", time: " << duration << " ms)\n";
                
                // Compute and extract values for monitoring
                if (incResult.state) {
                    
                    // Monitor voltage violations
                    auto buses = network->getBuses();
                    for (auto* bus : buses) {
                        if (bus) {
                            Real vPU = bus->getVPU();
                            if (vPU < 0.95 || vPU > 1.05) {
                                voltageViolations++;
                                std::cout << "  ⚠ Voltage violation at Bus " << bus->getId() 
                                          << ": " << bus->getVKV() << " kV (" << vPU << " p.u.)\n";
                            }
                        }
                    }
                    
                    // Monitor branch overloads
                    auto branches = network->getBranches();
                    for (auto* branch : branches) {
                        if (branch) {
                            Real pMW = branch->getPMW();
                            Real qMVAR = branch->getQMVAR();
                            Real sFlow = std::sqrt(pMW * pMW + qMVAR * qMVAR);
                            Real rating = branch->getRating();
                            
                            if (rating > 0 && sFlow > rating * 0.9) {
                                overloads++;
                                std::cout << "  ⚠ Branch overload: Branch " << branch->getId() 
                                          << " (" << branch->getFromBus() << " -> " 
                                          << branch->getToBus() << "): " 
                                          << sFlow << " MVA (rating: " << rating << " MVA)\n";
                            }
                        }
                    }
                }
            }
            
            // Simulate real-time delay
            std::this_thread::sleep_for(std::chrono::milliseconds(UPDATE_INTERVAL_MS));
        }
        
        // ========================================================================
        // STEP 7: Final Full Estimation
        // ========================================================================
        std::cout << "\n=== Final Full Estimation ===\n";
        result = estimator.estimate();
        std::cout << "✓ Final estimation: " << result.message 
                  << " (iterations: " << result.iterations << ")\n\n";
        
        // ========================================================================
        // STEP 8: Compute All Estimated Values
        // ========================================================================
        std::cout << "=== Computing Estimated Values ===\n";
        if (result.state) {
            std::cout << "✓ Estimated values ready (solver already populated bus/branch metrics)\n\n";
        }
        
        // ========================================================================
        // STEP 9: System Monitoring
        // ========================================================================
        std::cout << "=== System Monitoring ===\n";
        std::cout << std::fixed << std::setprecision(3);
        
        voltageViolations = 0;
        overloads = 0;
        auto buses = network->getBuses();
        auto branches = network->getBranches();
        
        // Voltage violations
        for (auto* bus : buses) {
            if (bus) {
                Real vPU = bus->getVPU();
                if (vPU < 0.95 || vPU > 1.05) {
                    voltageViolations++;
                    std::cout << "⚠ Voltage violation at Bus " << bus->getId() 
                              << ": " << bus->getVKV() << " kV (" << vPU << " p.u.)\n";
                }
            }
        }
        if (voltageViolations == 0) {
            std::cout << "✓ All voltages within acceptable range (0.95 - 1.05 p.u.)\n";
        } else {
            std::cout << "⚠ Total voltage violations: " << voltageViolations << "\n";
        }
        
        // Branch overloads
        for (auto* branch : branches) {
            if (branch) {
                Real pMW = branch->getPMW();
                Real qMVAR = branch->getQMVAR();
                Real sFlow = std::sqrt(pMW * pMW + qMVAR * qMVAR);
                Real rating = branch->getRating();
                
                if (rating > 0 && sFlow > rating * 0.9) {
                    overloads++;
                    std::cout << "⚠ Branch overload: Branch " << branch->getId() 
                              << " (" << branch->getFromBus() << " -> " 
                              << branch->getToBus() << "): " 
                              << sFlow << " MVA (rating: " << rating << " MVA)\n";
                }
            }
        }
        if (overloads == 0) {
            std::cout << "✓ No branch overloads detected\n";
        } else {
            std::cout << "⚠ Total branch overloads: " << overloads << "\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 10: Bad Data Detection
        // ========================================================================
        std::cout << "=== Bad Data Detection ===\n";
        sle::baddata::BadDataDetector badDataDetector;
        badDataDetector.setNormalizedResidualThreshold(3.0);  // 3-sigma threshold
        auto badDataResult = badDataDetector.detectBadData(
            *telemetryShared, *result.state, *network);
        
        if (badDataResult.hasBadData) {
            std::cout << "⚠ Bad data detected in " << badDataResult.badDeviceIds.size() 
                      << " measurements:\n";
            for (const auto& deviceId : badDataResult.badDeviceIds) {
                std::cout << "  - " << deviceId << "\n";
            }
            std::cout << "Chi-square statistic: " << badDataResult.chiSquareStatistic << "\n";
        } else {
            std::cout << "✓ No bad data detected (chi-square: " 
                      << badDataResult.chiSquareStatistic << ")\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 11: Generate Reports
        // ========================================================================
        std::cout << "=== Generating Reports ===\n";
        
        // JSON results
        sle::interface::Results results(result);
        sle::io::OutputFormatter::writeToFile("realtime_results.json", results, "json");
        std::cout << "✓ Results saved to: realtime_results.json\n";
        
        // Comparison report
        auto comparisons = sle::io::ComparisonReport::compare(
            *telemetryShared, *result.state, *network);
        sle::io::ComparisonReport::writeReport("realtime_comparison.txt", comparisons);
        std::cout << "✓ Comparison report saved to: realtime_comparison.txt\n";
        
        // System summary
        std::ofstream summaryFile("realtime_summary.txt");
        if (summaryFile.is_open()) {
            summaryFile << "=== Real-Time State Estimation Summary ===\n\n";
            summaryFile << "Network: " << network->getBusCount() << " buses, " 
                       << network->getBranchCount() << " branches\n";
            summaryFile << "Measurements: " << telemetryShared->getMeasurementCount() << "\n";
            summaryFile << "Estimation: " << (result.converged ? "Converged" : "Failed") 
                       << " in " << result.iterations << " iterations\n";
            summaryFile << "Computation time: " << duration << " ms\n\n";
            
            summaryFile << "=== Bus Summary ===\n";
            summaryFile << std::fixed << std::setprecision(6);
            for (auto* bus : buses) {
                if (bus) {
                    summaryFile << "Bus " << bus->getId() << ": "
                               << "V = " << bus->getVPU() << " p.u. = " 
                               << bus->getVKV() << " kV, "
                               << "θ = " << bus->getThetaDeg() << " deg, "
                               << "P = " << bus->getPInjectionMW() << " MW, "
                               << "Q = " << bus->getQInjectionMVAR() << " MVAR\n";
                }
            }
            
            summaryFile << "\n=== Branch Summary ===\n";
            summaryFile << std::setprecision(3);
            for (auto* branch : branches) {
                if (branch) {
                    summaryFile << "Branch " << branch->getId() << " (" 
                               << branch->getFromBus() << " -> " 
                               << branch->getToBus() << "): "
                               << "P = " << branch->getPMW() << " MW, "
                               << "Q = " << branch->getQMVAR() << " MVAR, "
                               << "I = " << branch->getIAmps() << " A\n";
                }
            }
            
            summaryFile << "\n=== Violations ===\n";
            summaryFile << "Voltage violations: " << voltageViolations << "\n";
            summaryFile << "Branch overloads: " << overloads << "\n";
            summaryFile << "Bad measurements: " << badDataResult.badDeviceIds.size() << "\n";
            
            summaryFile.close();
            std::cout << "✓ System summary saved to: realtime_summary.txt\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 12: Display Key Values
        // ========================================================================
        std::cout << "=== Key System Values ===\n";
        std::cout << std::fixed << std::setprecision(3);
        
        // Display top 5 buses by power injection
        std::cout << "\nTop 5 Buses by Power Injection:\n";
        std::vector<std::pair<Real, sle::BusId>> busPower;
        for (auto* bus : buses) {
            if (bus) {
                Real sInj = std::sqrt(bus->getPInjectionMW() * bus->getPInjectionMW() + 
                                     bus->getQInjectionMVAR() * bus->getQInjectionMVAR());
                busPower.push_back({sInj, bus->getId()});
            }
        }
        std::sort(busPower.rbegin(), busPower.rend());
        for (size_t i = 0; i < std::min(5UL, busPower.size()); ++i) {
            auto* bus = network->getBus(busPower[i].second);
            if (bus) {
                std::cout << "  " << (i+1) << ". Bus " << bus->getId() << ": "
                         << bus->getPInjectionMW() << " MW, " 
                         << bus->getQInjectionMVAR() << " MVAR\n";
            }
        }
        
        // Display top 5 branches by power flow
        std::cout << "\nTop 5 Branches by Power Flow:\n";
        std::vector<std::pair<Real, sle::BranchId>> branchPower;
        for (auto* branch : branches) {
            if (branch) {
                Real sFlow = std::sqrt(branch->getPMW() * branch->getPMW() + 
                                      branch->getQMVAR() * branch->getQMVAR());
                branchPower.push_back({sFlow, branch->getId()});
            }
        }
        std::sort(branchPower.rbegin(), branchPower.rend());
        for (size_t i = 0; i < std::min(5UL, branchPower.size()); ++i) {
            // Find branch by ID
            sle::model::Branch* branch = nullptr;
            for (auto* b : branches) {
                if (b && b->getId() == branchPower[i].second) {
                    branch = b;
                    break;
                }
            }
            if (branch) {
                std::cout << "  " << (i+1) << ". Branch " << branch->getId() << ": "
                         << branch->getPMW() << " MW, " 
                         << branch->getQMVAR() << " MVAR, "
                         << branch->getIAmps() << " A\n";
            }
        }
        
        std::cout << "\n=== Production Real-Time Setup Completed Successfully ===\n";
        std::cout << "All estimated values are available via Bus/Branch getters:\n";
        std::cout << "  - Bus::getVPU(), getVKV(), getThetaDeg()\n";
        std::cout << "  - Bus::getPInjectionMW(), getQInjectionMVAR()\n";
        std::cout << "  - Branch::getPMW(), getQMVAR(), getIAmps()\n";
        std::cout << "\nReports generated:\n";
        std::cout << "  - realtime_results.json\n";
        std::cout << "  - realtime_comparison.txt\n";
        std::cout << "  - realtime_summary.txt\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
