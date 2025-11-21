/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Hybrid Setup Example
 * 
 * This example demonstrates the optimal hybrid approach for real-time state estimation:
 * - Fast standard WLS for real-time updates (every cycle) - ~10-50 ms
 * - Periodic robust estimation to detect and handle bad data (every N cycles) - ~100-500 ms
 * - Bad data detection to identify problematic measurements (every M cycles) - ~20-100 ms
 * 
 * Strategy:
 * 1. Use fast WLS for every-cycle updates (provides real-time state)
 * 2. Periodically run robust estimation (handles bad data automatically)
 * 3. Frequently check for bad data (identifies problems quickly)
 * 
 * Use cases:
 * - Real-time applications requiring fast updates (SCADA, EMS)
 * - Systems with occasional bad data that need automatic handling
 * - When you need both speed (WLS) and accuracy (robust estimation)
 * - Production systems where bad data detection alone isn't sufficient
 * 
 * Performance:
 * - Standard WLS: Fast (2-5 iterations), suitable for every cycle
 * - Robust estimation: Slower (20+ IRLS iterations × 10 WLS iterations), use periodically
 * - Bad data detection: Fast (statistical tests), use frequently
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
#include <vector>

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Hybrid Robust Estimation Setup ===\n\n";
        
        // ========================================================================
        // STEP 1: Load Network Model and Measurements
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
        // STEP 2: Configure Standard WLS Estimator (Fast)
        // ========================================================================
        std::cout << "=== Configuring Standard WLS Estimator ===\n";
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));
        estimator.setTelemetryData(telemetry);
        
        // Real-time mode: Fast, relaxed tolerance
        estimator.configureForRealTime(1e-5, 15, true);
        std::cout << "✓ Standard WLS configured for fast real-time updates\n\n";
        
        // ========================================================================
        // STEP 3: Configure Robust Estimator (Accurate)
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
        // STEP 4: Configure Bad Data Detector
        // ========================================================================
        std::cout << "=== Configuring Bad Data Detector ===\n";
        sle::baddata::BadDataDetector badDataDetector;
        badDataDetector.setNormalizedResidualThreshold(3.0);  // 3-sigma rule
        std::cout << "✓ Bad data detector configured\n\n";
        
        // ========================================================================
        // STEP 5: Start Real-Time Processing
        // ========================================================================
        std::cout << "=== Starting Real-Time Processing ===\n";
        estimator.getTelemetryProcessor().startRealTimeProcessing();
        std::cout << "✓ Real-time processing started\n\n";
        
        // ========================================================================
        // STEP 6: Initial Full Estimation (Standard WLS)
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
        // STEP 7: Configure Hybrid Strategy Intervals
        // ========================================================================
        const int ROBUST_CHECK_INTERVAL = 5;      // Run robust estimation every 5 cycles
        const int BAD_DATA_CHECK_INTERVAL = 3;    // Check for bad data every 3 cycles
        int cycleCount = 0;
        
        std::cout << "=== Hybrid Strategy Configuration ===\n";
        std::cout << "  - Fast WLS: Every cycle (real-time updates)\n";
        std::cout << "  - Robust estimation: Every " << ROBUST_CHECK_INTERVAL << " cycles\n";
        std::cout << "  - Bad data detection: Every " << BAD_DATA_CHECK_INTERVAL << " cycles\n\n";
        
        // ========================================================================
        // STEP 8: Real-Time Update Loop with Hybrid Strategy
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
            
            estimator.getTelemetryProcessor().updateMeasurement(update);
            
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
            // STEP 8c: Robust Estimation - Every ROBUST_CHECK_INTERVAL Cycles
            // ====================================================================
            if (cycleCount % ROBUST_CHECK_INTERVAL == 0 && wlsResult.state) {
                std::cout << "  [Robust] Running robust estimation...\n";
                startTime = std::chrono::high_resolution_clock::now();
                auto robustResult = robustEstimator.estimate(
                    *wlsResult.state, *network, *telemetry);
                endTime = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    endTime - startTime).count();
                
                std::cout << "  [Robust] " << robustResult.message
                          << " (iterations: " << robustResult.iterations 
                          << ", time: " << duration << " ms)\n";
            }
            
            // Simulate real-time delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // ========================================================================
        // STEP 9: Final Full Estimation
        // ========================================================================
        std::cout << "\n=== Final Full Estimation ===\n";
        result = estimator.estimate();
        std::cout << "✓ Final estimation: " << result.message 
                  << " (iterations: " << result.iterations << ")\n";
        
        // ========================================================================
        // STEP 10: Stop Real-Time Processing
        // ========================================================================
        estimator.getTelemetryProcessor().stopRealTimeProcessing();
        std::cout << "\n✓ Real-time processing stopped\n";
        
        std::cout << "\n=== Hybrid Setup Completed Successfully ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

