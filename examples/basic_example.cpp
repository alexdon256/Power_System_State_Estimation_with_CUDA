/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Basic State Estimation Example
 * 
 * This example demonstrates the complete workflow for state estimation:
 * 1. Load network model (buses, branches, transformers)
 * 2. Load telemetry measurements (power, voltage, current)
 * 3. Check observability (ensure system can be estimated)
 * 4. Validate data consistency (pre-processing checks)
 * 5. Run state estimation (Weighted Least Squares with Newton-Raphson)
 * 6. Detect bad data (post-estimation validation)
 * 7. Display and save results (voltage magnitudes and angles)
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/io/OutputFormatter.h>
#include <sle/observability/ObservabilityAnalyzer.h>
#include <sle/baddata/BadDataDetector.h>
#include <sle/baddata/DataConsistencyChecker.h>
#include <iostream>

int main() {
    try {
        std::cout << "=== Basic State Estimation Example ===\n\n";
        
        // ========================================================================
        // STEP 1: Load Network Model
        // ========================================================================
        // The network model contains the power system topology:
        // - Buses: Network nodes with voltage, load, and generation
        // - Branches: Transmission lines and transformers connecting buses
        // - Parameters: Impedances, tap ratios, ratings, etc.
        // 
        // IEEE Common Format is the standard format for power system data files
        std::cout << "Loading network model...\n";
        auto network = sle::interface::ModelLoader::loadFromIEEE("examples/ieee14/network.dat");
        if (!network) {
            std::cerr << "Failed to load network model\n";
            return 1;
        }
        std::cout << "Loaded network with " << network->getBusCount() 
                  << " buses and " << network->getBranchCount() << " branches\n\n";
        
        // ========================================================================
        // STEP 2: Load Telemetry Measurements
        // ========================================================================
        // Measurements are the actual sensor readings from the power system:
        // - Power injections: Active (P) and reactive (Q) power at buses
        // - Power flows: Active and reactive power on branches
        // - Voltage magnitudes: Bus voltage measurements
        // - Current magnitudes: Branch current measurements (from current transformers)
        // - PMU phasors: Synchronized voltage and current phasors (if available)
        //
        // Each measurement includes:
        // - Value: The measured quantity (MW, MVAR, p.u., Amperes, etc.)
        // - Standard deviation: Measurement uncertainty (used for weighting)
        // - Device ID: Identifier of the measuring device
        // - Location: Bus or branch where measurement is taken
        std::cout << "Loading measurements...\n";
        auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
            "examples/ieee14/measurements.csv", *network);
        std::cout << "Loaded " << telemetry->getMeasurementCount() << " measurements\n\n";
        
        // ========================================================================
        // STEP 3: Check Observability
        // ========================================================================
        // Observability analysis determines if there are enough measurements
        // to estimate all bus voltages. A system is observable if:
        // - The measurement Jacobian matrix has full rank
        // - All buses have sufficient measurement redundancy
        //
        // If not observable, we can add:
        // - Virtual measurements: Zero injection constraints (Kirchhoff's law)
        // - Pseudo measurements: Load forecasts or historical patterns
        std::cout << "Checking observability...\n";
        sle::observability::ObservabilityAnalyzer obsAnalyzer;
        bool observable = obsAnalyzer.isFullyObservable(*network, *telemetry);
        std::cout << "System is " << (observable ? "observable" : "not observable") << "\n";
        
        if (!observable) {
            // Add virtual measurements (zero injection constraints)
            // These enforce Kirchhoff's current law: sum of injections = 0
            // at buses with no load and no generation
            std::cout << "Adding virtual measurements...\n";
            sle::interface::MeasurementLoader::addVirtualMeasurements(*telemetry, *network);
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 4: Check Data Consistency
        // ========================================================================
        // Pre-estimation validation to catch obvious errors:
        // - Invalid bus/branch IDs
        // - Out-of-range measurement values
        // - Missing required data
        // - Unit inconsistencies
        //
        // This is faster than running estimation and catching errors later
        std::cout << "Checking data consistency...\n";
        sle::baddata::DataConsistencyChecker consistencyChecker;
        auto consistency = consistencyChecker.checkConsistency(*telemetry, *network);
        if (!consistency.isConsistent) {
            std::cout << "Warnings found:\n";
            for (const auto& issue : consistency.inconsistencies) {
                std::cout << "  - " << issue << "\n";
            }
        } else {
            std::cout << "Data is consistent\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 5: Create State Estimator and Configure Solver
        // ========================================================================
        // StateEstimator is the main API class that coordinates:
        // - Network model management
        // - Measurement processing
        // - Solver execution
        // - Result formatting
        std::cout << "Creating state estimator...\n";
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));
        estimator.setTelemetryData(telemetry);
        
        // Configure the Newton-Raphson solver
        sle::math::SolverConfig config;
        config.tolerance = 1e-6;        // Convergence tolerance: maximum allowed residual norm
                                        // Smaller = more accurate but slower (typical: 1e-4 to 1e-8)
        config.maxIterations = 50;      // Maximum Newton-Raphson iterations
                                        // Prevents infinite loops if convergence fails
        config.useGPU = true;           // Enable CUDA GPU acceleration
                                        // true: Use GPU for measurement functions, Jacobian, solving
                                        // false: CPU-only (slower but more compatible)
        config.verbose = true;          // Print iteration progress
        estimator.setSolverConfig(config);
        
        // ========================================================================
        // STEP 6: Run State Estimation
        // ========================================================================
        // State estimation uses Weighted Least Squares (WLS) method:
        // - Minimizes weighted sum of squared residuals
        // - Uses Newton-Raphson iterative solver
        // - Estimates voltage magnitudes and angles for all buses
        //
        // The algorithm:
        // 1. Initialize state (flat start: V=1.0 p.u., θ=0.0 rad)
        // 2. Evaluate measurement functions h(x)
        // 3. Compute residuals r = z - h(x)
        // 4. Build Jacobian matrix H = ∂h/∂x
        // 5. Solve normal equations: H^T W H Δx = H^T W r
        // 6. Update state: x = x + Δx
        // 7. Repeat until convergence (||r|| < tolerance)
        std::cout << "Running state estimation...\n";
        auto result = estimator.estimate();
        
        // Display estimation results
        std::cout << "\n=== Results ===\n";
        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
        std::cout << "Final norm: " << result.finalNorm << "\n";  // Final residual norm
        std::cout << "Objective value: " << result.objectiveValue << "\n";  // WLS objective function
        std::cout << "Message: " << result.message << "\n\n";
        
        // ========================================================================
        // STEP 7: Bad Data Detection
        // ========================================================================
        // Post-estimation validation to identify erroneous measurements:
        // - Chi-square test: Overall measurement quality
        // - Largest normalized residual (LNR): Identifies specific bad measurements
        //
        // Normalized residual = residual / sqrt(variance)
        // Measurements with |normalized residual| > threshold are flagged as bad
        // Typical threshold: 3.0 (3-sigma rule)
        if (result.state) {
            std::cout << "Checking for bad data...\n";
            sle::baddata::BadDataDetector badDataDetector;
            badDataDetector.setNormalizedResidualThreshold(3.0);  // 3-sigma threshold
            auto badDataResult = badDataDetector.detectBadData(
                *telemetry, *result.state, *network);
            
            if (badDataResult.hasBadData) {
                std::cout << "Bad data detected in " << badDataResult.badDeviceIds.size() 
                          << " measurements\n";
                for (const auto& deviceId : badDataResult.badDeviceIds) {
                    std::cout << "  - " << deviceId << "\n";
                }
                // Optionally remove bad measurements and re-estimate:
                // badDataDetector.removeBadMeasurements(*telemetry, badDataResult);
                // result = estimator.estimate();
            } else {
                std::cout << "No bad data detected\n";
            }
            std::cout << "\n";
        }
        
        // ========================================================================
        // STEP 8: Output Results
        // ========================================================================
        // Save results to file in JSON format for further analysis
        sle::interface::Results results(result);
        sle::io::OutputFormatter::writeToFile("results.json", results, "json");
        std::cout << "Results written to results.json\n";
        
        // Display voltage estimates for all buses
        // Voltage magnitude: in per-unit (1.0 p.u. = nominal voltage)
        // Voltage angle: in radians (relative to slack bus, typically 0.0)
        if (result.state) {
            std::cout << "\n=== Voltage Estimates ===\n";
            auto voltages = results.getVoltages();  // Voltage magnitudes [V₁, V₂, ..., Vₙ]
            auto angles = results.getAngles();       // Voltage angles [θ₁, θ₂, ..., θₙ]
            for (size_t i = 0; i < voltages.size(); ++i) {
                std::cout << "Bus " << (i+1) << ": V = " << voltages[i] 
                          << " p.u., θ = " << angles[i] << " rad\n";
            }
        }
        
        std::cout << "\nExample completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

