/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Offline Setup Example
 * 
 * This example demonstrates offline state estimation for analysis and planning:
 * - High accuracy with tight convergence tolerance
 * - Comprehensive validation (observability, data consistency, bad data)
 * - Detailed computed values extraction (voltage, power, current)
 * - System monitoring (violations, overloads)
 * - Complete reporting (JSON, comparison, summary)
 * 
 * Use cases:
 * - Planning studies and analysis
 * - Offline validation and testing
 * - Historical data analysis
 * - Research and development
 * - System validation before deployment
 * 
 * Configuration: Offline mode (high accuracy, relaxed timing)
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/io/OutputFormatter.h>
#include <sle/io/ComparisonReport.h>
#include <sle/observability/ObservabilityAnalyzer.h>
#include <sle/baddata/BadDataDetector.h>
#include <sle/baddata/DataConsistencyChecker.h>
#include <sle/Types.h>
#include <sle/model/Bus.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>

using sle::Real;

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Offline State Estimation Setup ===\n\n";
        
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
        // STEP 2: Observability Analysis
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
        } else {
            std::cout << "✓ System is fully observable\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 3: Data Consistency Check
        // ========================================================================
        std::cout << "=== Data Consistency Check ===\n";
        sle::baddata::DataConsistencyChecker consistencyChecker;
        auto consistency = consistencyChecker.checkConsistency(*telemetry, *network);
        if (!consistency.isConsistent) {
            std::cout << "⚠ Data consistency issues found:\n";
            for (const auto& issue : consistency.inconsistencies) {
                std::cout << "  - " << issue << "\n";
            }
        } else {
            std::cout << "✓ Data is consistent\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 4: Configure Estimator for Offline Analysis
        // ========================================================================
        std::cout << "=== Configuring Estimator ===\n";
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(network);
        estimator.setTelemetryData(telemetry);
        
        // Offline mode: High accuracy, relaxed timing
        estimator.configureForOffline(1e-8, 50, true);  // tolerance, maxIter, useGPU
        std::cout << "✓ Configured for offline analysis (high accuracy)\n";
        std::cout << "  - Tolerance: 1e-8\n";
        std::cout << "  - Max iterations: 50\n";
        std::cout << "  - GPU acceleration: Enabled\n\n";
        
        // ========================================================================
        // STEP 5: Run State Estimation
        // ========================================================================
        std::cout << "=== Running State Estimation ===\n";
        auto startTime = std::chrono::high_resolution_clock::now();
        auto result = estimator.estimate();
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime).count();
        
        if (!result.converged) {
            std::cerr << "ERROR: State estimation failed to converge\n";
            std::cerr << "Message: " << result.message << "\n";
            return 1;
        }
        
        std::cout << "✓ Estimation converged successfully\n";
        std::cout << "  - Iterations: " << result.iterations << "\n";
        std::cout << "  - Final norm: " << std::scientific << std::setprecision(3) 
                  << result.finalNorm << "\n";
        std::cout << "  - Objective value: " << result.objectiveValue << "\n";
        std::cout << "  - Computation time: " << duration << " ms\n\n";
        
        // ========================================================================
        // STEP 6: Compute All Estimated Values
        // ========================================================================
        std::cout << "=== Computing Estimated Values ===\n";
        std::cout << "✓ Estimated values ready (solver already populated bus/branch metrics)\n\n";
        
        // ========================================================================
        // STEP 7: System Monitoring
        // ========================================================================
        std::cout << "=== System Monitoring ===\n";
        std::cout << std::fixed << std::setprecision(3);
        
        // Voltage violations
        int voltageViolations = 0;
        auto buses = network->getBuses();
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
        int overloads = 0;
        auto branches = network->getBranches();
        for (auto* branch : branches) {
            if (!branch) continue;
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
        if (overloads == 0) {
            std::cout << "✓ No branch overloads detected\n";
        } else {
            std::cout << "⚠ Total branch overloads: " << overloads << "\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 8: Bad Data Detection
        // ========================================================================
        std::cout << "=== Bad Data Detection ===\n";
        sle::baddata::BadDataDetector badDataDetector;
        badDataDetector.setNormalizedResidualThreshold(3.0);
        auto badDataResult = badDataDetector.detectBadData(
            *telemetry, *result.state, *network);
            auto badDataResult = badDataDetector.detectBadData(
                *telemetry, *result.state, *network);
        
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
        // STEP 9: Generate Reports
        // ========================================================================
        std::cout << "=== Generating Reports ===\n";
        
        // JSON results
        sle::interface::Results results(result);
        sle::io::OutputFormatter::writeToFile("offline_results.json", results, "json");
        std::cout << "✓ Results saved to: offline_results.json\n";
        
        // Comparison report
        auto comparisons = sle::io::ComparisonReport::compare(
            *telemetry, *result.state, *network);
        sle::io::ComparisonReport::writeReport("offline_comparison.txt", comparisons);
        std::cout << "✓ Comparison report saved to: offline_comparison.txt\n";
        
        // System summary
        std::ofstream summaryFile("offline_summary.txt");
        if (summaryFile.is_open()) {
            summaryFile << "=== Offline State Estimation Summary ===\n\n";
            summaryFile << "Network: " << network->getBusCount() << " buses, " 
                       << network->getBranchCount() << " branches\n";
            summaryFile << "Measurements: " << telemetry->getMeasurementCount() << "\n";
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
            std::cout << "✓ System summary saved to: offline_summary.txt\n";
        }
        
        std::cout << "\n=== Offline Setup Completed Successfully ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

