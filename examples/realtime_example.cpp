/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Real-Time State Estimation Example
 * 
 * This example demonstrates real-time state estimation with on-the-fly measurement updates:
 * 1. Load network model and initial measurements
 * 2. Start real-time telemetry processing (asynchronous updates)
 * 3. Update measurements dynamically without full reload
 * 4. Run incremental estimation (faster than full estimation)
 * 5. Process telemetry updates in real-time loop
 * 
 * Key features:
 * - TelemetryProcessor: Thread-safe measurement update queue
 * - Incremental estimation: Uses previous state as initial guess (faster convergence)
 * - Real-time updates: Measurements can be updated while estimation is running
 * - GPU acceleration: Enabled for fast real-time performance
 * 
 * Use cases:
 * - SCADA systems with continuous telemetry streams
 * - Energy Management Systems (EMS) requiring real-time state
 * - PMU data integration with high update rates
 * - Dynamic systems with frequent measurement changes
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/io/OutputFormatter.h>
#include <sle/io/ComparisonReport.h>
#include <sle/Types.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>

int main() {
    try {
        // ========================================================================
        // STEP 1: Load Network Model and Initial Measurements
        // ========================================================================
        // Load the power system network topology (buses, branches, transformers)
        // This is typically done once at startup, then updated incrementally
        std::cout << "Loading network model...\n";
        auto network = sle::interface::ModelLoader::loadFromIEEE("network.dat");
        if (!network) {
            std::cerr << "Failed to load network\n";
            return 1;
        }
        
        // Load initial telemetry measurements
        // These provide the baseline state before real-time updates begin
        std::cout << "Loading measurements...\n";
        auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
            "measurements.csv", *network);
        
        // ========================================================================
        // STEP 2: Create State Estimator and Configure for Real-Time
        // ========================================================================
        // StateEstimator manages the estimation process and coordinates updates
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));
        estimator.setTelemetryData(telemetry);
        
        // Configure solver for real-time operation (optimized for speed)
        sle::math::SolverConfig config;
        config.tolerance = 1e-5;        // Slightly relaxed tolerance for faster convergence
                                        // Real-time systems prioritize speed over extreme accuracy
                                        // Typical: 1e-4 to 1e-6 (vs 1e-8 for offline analysis)
        config.maxIterations = 15;       // Fewer iterations for speed
                                        // Real-time: 10-20 iterations (vs 50+ for offline)
                                        // Incremental estimation typically converges in 2-5 iterations
        config.useGPU = true;           // Enable GPU acceleration for real-time performance
                                        // GPU provides 5-100x speedup depending on system size
        estimator.setSolverConfig(config);
        
        // ========================================================================
        // STEP 3: Start Real-Time Telemetry Processing
        // ========================================================================
        // TelemetryProcessor enables asynchronous measurement updates
        // - Thread-safe update queue for concurrent access
        // - Background processing thread for handling updates
        // - Automatic timestamp tracking for stale data detection
        // - Batch update support for efficient processing
        std::cout << "Starting real-time processing...\n";
        estimator.getTelemetryProcessor().startRealTimeProcessing();
        
        // ========================================================================
        // STEP 4: Run Initial Full Estimation
        // ========================================================================
        // Initial estimation establishes baseline state from initial measurements
        // Uses flat start (V=1.0 p.u., θ=0.0 rad) or load flow solution
        // Subsequent updates use incremental estimation (faster)
        std::cout << "Running initial estimation...\n";
        auto result = estimator.estimate();
        std::cout << "Initial estimation: " << result.message 
                  << " (iterations: " << result.iterations << ")\n";
        
        // ========================================================================
        // STEP 5: Real-Time Update Loop
        // ========================================================================
        // Simulate real-time telemetry updates from SCADA/PMU systems
        // In production, this would be driven by actual telemetry streams
        std::cout << "\nEntering real-time update loop...\n";
        for (int i = 0; i < 10; ++i) {
            // Create a telemetry update structure
            // In production, this would come from SCADA/PMU data streams
            sle::interface::TelemetryUpdate update;
            update.deviceId = "METER_" + std::to_string(i % 5);  // Device identifier
            update.type = sle::MeasurementType::P_INJECTION;     // Measurement type
            update.value = 1.0 + (i * 0.05);                     // New measurement value (p.u.)
            update.stdDev = 0.01;                                // Measurement uncertainty
            update.busId = (i % network->getBusCount()) + 1;      // Bus location
            update.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();  // Unix timestamp (ms)
            
            // Update measurement in real-time (thread-safe)
            // This updates existing measurement if deviceId matches, or adds new measurement
            // Can be called from multiple threads (e.g., SCADA thread, PMU thread)
            estimator.getTelemetryProcessor().updateMeasurement(update);
            
            // Run incremental estimation periodically
            // Incremental estimation uses previous state as initial guess:
            // - Faster convergence (typically 2-5 iterations vs 10-20 for full estimation)
            // - Suitable for small state changes between updates
            // - Not run every cycle to balance accuracy and performance
            if (i % 5 == 0) {
                auto incResult = estimator.estimateIncremental();
                std::cout << "Update " << i << ": " << incResult.message
                          << " (iterations: " << incResult.iterations << ")\n";
            }
            
            // Simulate real-time delay (100ms = 10 Hz update rate)
            // Typical SCADA update rates: 1-10 Hz
            // PMU update rates: 30-120 Hz (phasor data)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // ========================================================================
        // STEP 6: Final Full Estimation
        // ========================================================================
        // Run final full estimation to ensure accuracy after all updates
        // Full estimation resets initial guess and ensures convergence
        std::cout << "\nRunning final estimation...\n";
        result = estimator.estimate();
        
        // ========================================================================
        // STEP 7: Compare Measured vs Estimated Values
        // ========================================================================
        // ComparisonReport compares measured values with estimated values
        // - Residuals: Difference between measured and estimated
        // - Normalized residuals: Residuals normalized by standard deviation
        // - Bad data detection: Flags measurements with large normalized residuals
        // - Used for validation, bad data detection, and measurement accuracy assessment
        std::cout << "\nComparing measured vs estimated values...\n";
        if (result.state && telemetry) {
            // Compare all measurements
            auto comparisons = sle::io::ComparisonReport::compare(
                *telemetry, *result.state, *network);
            
            std::cout << "Comparison Results:\n";
            std::cout << "==================\n";
            std::cout << std::fixed << std::setprecision(6);
            
            // Display summary statistics
            int badCount = 0;
            Real maxResidual = 0.0;
            Real maxNormalizedResidual = 0.0;
            
            for (const auto& comp : comparisons) {
                if (comp.isBad) badCount++;
                maxResidual = std::max(maxResidual, std::abs(comp.residual));
                maxNormalizedResidual = std::max(maxNormalizedResidual, comp.normalizedResidual);
            }
            
            std::cout << "Total measurements: " << comparisons.size() << "\n";
            std::cout << "Bad measurements: " << badCount << " (normalized residual > 3.0)\n";
            std::cout << "Max residual: " << maxResidual << " p.u.\n";
            std::cout << "Max normalized residual: " << maxNormalizedResidual << "\n\n";
            
            // Display detailed comparison for first 10 measurements
            std::cout << "Sample comparisons (first 10):\n";
            std::cout << std::setw(12) << "Device ID" 
                      << std::setw(12) << "Measured" 
                      << std::setw(12) << "Estimated" 
                      << std::setw(12) << "Residual" 
                      << std::setw(18) << "Norm. Residual" 
                      << std::setw(8) << "Status" << "\n";
            std::cout << std::string(80, '-') << "\n";
            
            size_t displayCount = std::min(static_cast<size_t>(10), comparisons.size());
            for (size_t i = 0; i < displayCount; ++i) {
                const auto& comp = comparisons[i];
                std::cout << std::setw(12) << comp.deviceId
                          << std::setw(12) << comp.measuredValue
                          << std::setw(12) << comp.estimatedValue
                          << std::setw(12) << comp.residual
                          << std::setw(18) << comp.normalizedResidual
                          << std::setw(8) << (comp.isBad ? "BAD" : "OK") << "\n";
            }
            
            // Write full comparison report to file
            sle::io::ComparisonReport::writeReport("comparison_report.txt", comparisons);
            std::cout << "\nFull comparison report written to comparison_report.txt\n";
        }
        
        // ========================================================================
        // STEP 8: Output Results
        // ========================================================================
        // Save final state estimate to file for analysis
        sle::interface::Results results(result);
        sle::io::OutputFormatter::writeToFile("results.json", results, "json");
        std::cout << "Results written to results.json\n";
        
        // ========================================================================
        // STEP 9: Stop Real-Time Processing
        // ========================================================================
        // Clean shutdown: stop background processing thread
        estimator.getTelemetryProcessor().stopRealTimeProcessing();
        
        std::cout << "\nReal-time example completed successfully.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

