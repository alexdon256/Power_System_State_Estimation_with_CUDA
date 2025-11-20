/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/math/RobustEstimator.h>
#include <sle/baddata/BadDataDetector.h>
#include <sle/io/OutputFormatter.h>
#include <sle/Types.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

// Hybrid approach: Real-time WLS with periodic robust estimation
//
// This example demonstrates the optimal strategy for real-time state estimation:
// 1. Use fast standard WLS for real-time updates (every cycle) - ~10-50 ms
// 2. Periodically run robust estimation to detect and handle bad data (every N cycles) - ~100-500 ms
// 3. Use bad data detection to identify problematic measurements (every M cycles) - ~20-100 ms
//
// When to use this approach:
// - Real-time applications requiring fast updates (SCADA, EMS)
// - Systems with occasional bad data that need automatic handling
// - When you need both speed (WLS) and accuracy (robust estimation)
// - Production systems where bad data detection alone isn't sufficient
//
// Performance characteristics:
// - Standard WLS: Fast (2-5 iterations), suitable for every cycle
// - Robust estimation: Slower (20+ IRLS iterations × 10 WLS iterations), use periodically
// - Bad data detection: Fast (statistical tests), use frequently
//
// Trade-offs:
// - Speed: Standard WLS every cycle provides fast real-time updates
// - Accuracy: Periodic robust estimation ensures data quality
// - Overhead: Robust estimation adds 2-5x computation time when run

int main() {
    try {
        std::cout << "=== Hybrid Robust Estimation Example ===\n\n";
        
        // ========================================================================
        // STEP 1: Load Network Model and Measurements
        // ========================================================================
        std::cout << "Loading network model...\n";
        auto network = sle::interface::ModelLoader::loadFromIEEE("examples/ieee14/network.dat");
        if (!network) {
            std::cerr << "Failed to load network\n";
            return 1;
        }
        std::cout << "Loaded network with " << network->getBusCount() 
                  << " buses and " << network->getBranchCount() << " branches\n\n";
        
        std::cout << "Loading measurements...\n";
        auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
            "examples/ieee14/measurements.csv", *network);
        std::cout << "Loaded " << telemetry->getMeasurementCount() << " measurements\n\n";
        
        // ========================================================================
        // STEP 2: Create Standard WLS Estimator (Fast Real-Time)
        // ========================================================================
        // Standard WLS is fast and suitable for every-cycle real-time updates
        // Uses Gauss-Newton method with 2-5 iterations typically
        std::cout << "Creating real-time estimator (standard WLS)...\n";
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));
        estimator.setTelemetryData(telemetry);
        
        // Configure for fast real-time operation
        sle::math::SolverConfig config;
        config.tolerance = 1e-5;        // Slightly relaxed for speed (vs 1e-8 for offline)
                                        // Real-time systems prioritize speed over extreme accuracy
        config.maxIterations = 15;       // Fewer iterations for real-time (vs 50+ for offline)
                                        // Incremental estimation typically converges in 2-5 iterations
        config.useGPU = true;            // GPU acceleration for speed (5-100x speedup)
        estimator.setSolverConfig(config);
        
        // ========================================================================
        // STEP 3: Create Robust Estimator (Periodic Bad Data Handling)
        // ========================================================================
        // Robust estimation is slower but handles bad data automatically
        // Use periodically (every N cycles) to validate and correct for bad data
        // Algorithm: Iteratively Reweighted Least Squares (IRLS)
        // - Each IRLS iteration runs multiple WLS iterations
        // - Weights are updated based on residual magnitudes
        // - Large residuals → low weights (bad data down-weighted)
        std::cout << "Creating robust estimator (for periodic checks)...\n";
        sle::math::RobustEstimator robustEstimator;
        sle::math::RobustEstimatorConfig robustConfig;
        robustConfig.weightFunction = sle::math::RobustWeightFunction::HUBER;  // Good balance
                                                                                  // Huber: Moderate outliers, standard tuning
                                                                                  // Alternatives: Bi-square (severe outliers), Cauchy (very robust)
        robustConfig.tuningConstant = 1.345;  // Standard Huber tuning constant
                                                // Controls outlier sensitivity (lower = more robust)
        robustConfig.tolerance = 1e-6;         // Convergence tolerance for IRLS
        robustConfig.maxIterations = 20;        // Maximum IRLS iterations (each runs WLS internally)
                                                // Fewer for faster periodic checks (vs 50+ for offline)
        robustConfig.useGPU = true;            // GPU acceleration (robust estimation benefits from GPU)
        robustEstimator.setConfig(robustConfig);
        
        // ========================================================================
        // STEP 4: Create Bad Data Detector
        // ========================================================================
        // Bad data detection uses statistical tests (normalized residuals, chi-square)
        // Fast (20-100 ms) - can run more frequently than robust estimation
        // Identifies problematic measurements but doesn't handle them automatically
        sle::baddata::BadDataDetector badDataDetector;
        badDataDetector.setNormalizedResidualThreshold(3.0);  // 3-sigma rule
                                                               // Measurements with |normalized residual| > 3.0 are flagged
                                                               // Standard threshold: 3.0-4.0 (higher = less sensitive)
        
        // ========================================================================
        // STEP 5: Start Real-Time Processing
        // ========================================================================
        // TelemetryProcessor enables asynchronous measurement updates
        // Thread-safe update queue for concurrent access from multiple sources
        std::cout << "Starting real-time processing...\n";
        estimator.getTelemetryProcessor().startRealTimeProcessing();
        
        // ========================================================================
        // STEP 6: Initial Full Estimation (Standard WLS)
        // ========================================================================
        // Establish baseline state from initial measurements
        // Uses flat start (V=1.0 p.u., θ=0.0 rad) or load flow solution
        std::cout << "\n=== Initial Estimation (Standard WLS) ===\n";
        auto startTime = std::chrono::high_resolution_clock::now();
        auto result = estimator.estimate();
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
        std::cout << "Final norm: " << result.finalNorm << "\n";
        std::cout << "Time: " << duration.count() << " ms\n\n";
        
        // ========================================================================
        // STEP 7: Configure Hybrid Approach Intervals
        // ========================================================================
        // Hybrid strategy balances speed and accuracy:
        // - Fast WLS: Every cycle (10-50 ms) - provides real-time updates
        // - Bad data detection: Every 3 cycles (20-100 ms) - identifies problems
        // - Robust estimation: Every 5 cycles (100-500 ms) - handles bad data
        //
        // Tuning guidelines:
        // - Increase ROBUST_CHECK_INTERVAL if system is stable (less frequent checks)
        // - Decrease ROBUST_CHECK_INTERVAL if bad data is common (more frequent checks)
        // - Adjust BAD_DATA_CHECK_INTERVAL based on measurement quality
        const int ROBUST_CHECK_INTERVAL = 5;      // Run robust estimation every 5 cycles
        const int BAD_DATA_CHECK_INTERVAL = 3;    // Check for bad data every 3 cycles
        int cycleCount = 0;
        
        std::cout << "=== Real-Time Update Loop ===\n";
        std::cout << "Strategy: Fast WLS every cycle, Robust estimation every " 
                  << ROBUST_CHECK_INTERVAL << " cycles\n";
        std::cout << "Bad data detection every " << BAD_DATA_CHECK_INTERVAL << " cycles\n\n";
        
        // ========================================================================
        // STEP 8: Real-Time Update Loop
        // ========================================================================
        // Simulate real-time telemetry updates from SCADA/PMU systems
        // In production, this would be driven by actual telemetry streams
        for (int i = 0; i < 20; ++i) {
            cycleCount++;
            
            // Simulate telemetry update
            sle::interface::TelemetryUpdate update;
            update.deviceId = "METER_" + std::to_string(i % 5);
            update.type = sle::MeasurementType::P_INJECTION;
            // Occasionally inject bad data for demonstration
            if (i == 8) {
                update.value = 10.0;  // Bad data: unrealistic value (should be ~1.0 p.u.)
                std::cout << "Cycle " << i << ": Injecting bad data (value = 10.0)\n";
            } else {
                update.value = 1.0 + (i * 0.05);  // Normal data: gradual change
            }
            update.stdDev = 0.01;     // Measurement uncertainty
            update.busId = (i % network->getBusCount()) + 1;
            update.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Update measurement in real-time (thread-safe)
            estimator.getTelemetryProcessor().updateMeasurement(update);
            
            // ====================================================================
            // STEP 8a: Fast Real-Time Estimation (Standard WLS) - Every Cycle
            // ====================================================================
            // Incremental estimation uses previous state as initial guess
            // Typically converges in 2-5 iterations (vs 10-20 for full estimation)
            // Fast enough for every-cycle updates (10-50 ms depending on system size)
            startTime = std::chrono::high_resolution_clock::now();
            auto wlsResult = estimator.estimateIncremental();
            endTime = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            std::cout << "Cycle " << i << " [WLS]: " << wlsResult.message
                      << " (iterations: " << wlsResult.iterations 
                      << ", time: " << duration.count() << " ms)\n";
            
            // ====================================================================
            // STEP 8b: Periodic Bad Data Detection - Every N Cycles
            // ====================================================================
            // Bad data detection uses normalized residual test (chi-square test)
            // Fast statistical test (20-100 ms) - can run more frequently
            // Identifies problematic measurements but doesn't handle them automatically
            if (cycleCount % BAD_DATA_CHECK_INTERVAL == 0 && wlsResult.state) {
                std::cout << "  -> Bad data check...\n";
                // Compute normalized residuals and test against threshold
                auto badDataResult = badDataDetector.detectBadData(
                    *telemetry, *wlsResult.state, *network);
                
                if (badDataResult.hasBadData) {
                    std::cout << "  -> Bad data detected in " 
                              << badDataResult.badDeviceIds.size() << " measurements:\n";
                    for (const auto& deviceId : badDataResult.badDeviceIds) {
                        std::cout << "     - " << deviceId << "\n";
                    }
                    // Optionally remove bad measurements automatically
                    // badDataDetector.removeBadMeasurements(*telemetry, badDataResult);
                } else {
                    std::cout << "  -> No bad data detected\n";
                }
            }
            
            // ====================================================================
            // STEP 8c: Periodic Robust Estimation - Every N Cycles
            // ====================================================================
            // Robust estimation handles bad data automatically by down-weighting outliers
            // Slower (100-500 ms) but more accurate - use periodically
            // IRLS algorithm: Iteratively reweights measurements based on residuals
            if (cycleCount % ROBUST_CHECK_INTERVAL == 0 && wlsResult.state) {
                std::cout << "  -> Running robust estimation (periodic check)...\n";
                
                // Use current WLS state as initial guess for robust estimation
                // This speeds up convergence (robust estimation starts near solution)
                sle::model::StateVector robustState = *wlsResult.state;
                
                startTime = std::chrono::high_resolution_clock::now();
                // Robust estimation runs IRLS: each IRLS iteration runs multiple WLS iterations
                auto robustResult = robustEstimator.estimate(robustState, *network, *telemetry);
                endTime = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
                
                std::cout << "  -> Robust estimation: " << robustResult.message
                          << " (IRLS iterations: " << robustResult.iterations
                          << ", time: " << duration.count() << " ms)\n";
                
                // Compare WLS and robust estimation results
                // Significant difference indicates bad data was handled by robust estimation
                if (robustResult.state) {
                    Real stateDiff = 0.0;
                    for (size_t j = 0; j < wlsResult.state->size(); ++j) {
                        Real vDiff = std::abs(wlsResult.state->getVoltageMagnitude(j) - 
                                            robustResult.state->getVoltageMagnitude(j));
                        Real aDiff = std::abs(wlsResult.state->getVoltageAngle(j) - 
                                            robustResult.state->getVoltageAngle(j));
                        stateDiff += vDiff + aDiff;
                    }
                    std::cout << "  -> State difference (WLS vs Robust): " 
                              << stateDiff << " p.u./rad\n";
                    
                    // If significant difference, robust estimation found and handled bad data
                    // Threshold: 0.01 p.u./rad (adjust based on system requirements)
                    if (stateDiff > 0.01) {
                        std::cout << "  -> WARNING: Significant difference detected!\n";
                        std::cout << "     Robust estimation suggests bad data may be present.\n";
                        std::cout << "     Consider removing bad measurements or using robust weights.\n";
                    }
                }
            }
            
            // Simulate real-time delay (50ms = 20 Hz update rate)
            // Typical SCADA: 1-10 Hz, PMU: 30-120 Hz
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        // ========================================================================
        // STEP 9: Final Comprehensive Check
        // ========================================================================
        // Run final robust estimation to ensure accuracy after all updates
        // This provides the most accurate final state estimate
        std::cout << "\n=== Final Comprehensive Check ===\n";
        std::cout << "Running final robust estimation...\n";
        
        auto finalState = estimator.getCurrentState();
        if (finalState) {
            sle::model::StateVector robustState = *finalState;
            auto finalRobustResult = robustEstimator.estimate(robustState, *network, *telemetry);
            
            std::cout << "Final robust estimation: " << finalRobustResult.message << "\n";
            std::cout << "IRLS iterations: " << finalRobustResult.iterations << "\n";
            std::cout << "Final norm: " << finalRobustResult.finalNorm << "\n";
            
            // Output final results to file
            if (finalRobustResult.state) {
                sle::interface::StateEstimationResult finalResult;
                finalResult.converged = finalRobustResult.converged;
                finalResult.iterations = finalRobustResult.iterations;
                finalResult.finalNorm = finalRobustResult.finalNorm;
                finalResult.objectiveValue = finalRobustResult.objectiveValue;
                finalResult.state = std::move(finalRobustResult.state);
                finalResult.message = finalRobustResult.message;
                
                sle::interface::Results results(finalResult);
                sle::io::OutputFormatter::writeToFile("hybrid_results.json", results, "json");
                std::cout << "Results written to hybrid_results.json\n";
            }
        }
        
        // ========================================================================
        // STEP 10: Stop Real-Time Processing
        // ========================================================================
        estimator.getTelemetryProcessor().stopRealTimeProcessing();
        
        // ========================================================================
        // STEP 11: Summary
        // ========================================================================
        std::cout << "\n=== Summary ===\n";
        std::cout << "Hybrid approach benefits:\n";
        std::cout << "  - Fast real-time updates: Standard WLS every cycle (~" 
                  << "10-50 ms depending on system size)\n";
        std::cout << "  - Periodic accuracy checks: Robust estimation every " 
                  << ROBUST_CHECK_INTERVAL << " cycles (~" 
                  << "100-500 ms depending on system size)\n";
        std::cout << "  - Bad data detection: Every " << BAD_DATA_CHECK_INTERVAL 
                  << " cycles (~" << "20-100 ms)\n";
        std::cout << "  - Best of both worlds: Speed for real-time, accuracy for validation\n";
        std::cout << "\nHybrid example completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

