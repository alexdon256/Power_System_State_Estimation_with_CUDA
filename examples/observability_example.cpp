/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Observability Analysis Example
 * 
 * This example demonstrates observability analysis and restoration techniques:
 * 1. Check if system is fully observable (can estimate all bus voltages)
 * 2. Identify non-observable buses (buses that cannot be estimated)
 * 3. Restore observability using pseudo measurements (load forecasts)
 * 4. Restore observability using pseudo measurements (load forecasts)
 * 5. Find optimal measurement placement for observability
 * 
 * Observability Concepts:
 * - A system is observable if the measurement Jacobian matrix has full rank
 * - Observable buses: Can be estimated from available measurements
 * - Non-observable buses: Cannot be estimated (need additional measurements)
 * - Pseudo measurements: Forecasted/estimated values (less accurate than real measurements)
 * 
 * Use cases:
 * - Pre-estimation validation (ensure system can be estimated)
 * - Measurement planning (determine where to place new meters)
 * - Observability restoration (add pseudo measurements automatically)
 * - Redundancy analysis (identify critical measurements)
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/observability/ObservabilityAnalyzer.h>
#include <sle/measurements/PseudoMeasurementGenerator.h>
#include <iostream>

int main() {
    try {
        std::cout << "=== Observability Analysis Example ===\n\n";
        
        // ========================================================================
        // STEP 1: Load Network and Measurements
        // ========================================================================
        // Load network topology (buses, branches, transformers)
        auto network = sle::interface::ModelLoader::loadFromIEEE("examples/ieee14/network.dat");
        
        // Load measurements (may be partial - simulate non-observable case)
        // In practice, partial measurements can occur due to:
        // - Meter failures
        // - Communication outages
        // - Insufficient meter placement
        auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
            "examples/ieee14/measurements.csv", *network);
        
        // ========================================================================
        // STEP 2: Initial Observability Check
        // ========================================================================
        // ObservabilityAnalyzer determines if the system can be fully estimated
        // Uses numerical rank analysis of the measurement Jacobian matrix
        sle::observability::ObservabilityAnalyzer analyzer;
        
        std::cout << "Initial observability check...\n";
        // Check if all buses can be estimated from available measurements
        // Returns true if measurement Jacobian has full rank (rank = 2N-1 for N buses)
        // Full rank means: rank(H) = 2N-1 (N voltage magnitudes + N-1 angles, slack angle fixed)
        bool observable = analyzer.isFullyObservable(*network, *telemetry);
        std::cout << "System observable: " << (observable ? "Yes" : "No") << "\n\n";
        
        // ========================================================================
        // STEP 3: Identify Non-Observable Buses
        // ========================================================================
        // If system is not observable, identify which buses cannot be estimated
        // These buses need additional measurements or pseudo measurements
        if (!observable) {
            std::cout << "Non-observable buses:\n";
            // Get list of bus IDs that cannot be estimated
            // Non-observable buses form unobservable islands in the network
            auto nonObsBuses = analyzer.getNonObservableBuses(*network, *telemetry);
            for (auto busId : nonObsBuses) {
                std::cout << "  Bus " << busId << "\n";
            }
            std::cout << "\n";
            
            // ========================================================================
            // STEP 4: Restore Observability with Pseudo Measurements (if needed)
            // ========================================================================
            // If system is still not observable, use pseudo measurements
            // Pseudo measurements are forecasted/estimated values:
            // - Load forecasts from historical patterns
            // - Estimated values from similar buses
            // - Values from previous state estimation
            //
            // Pseudo measurements have lower weight (higher stdDev) than real measurements
            // They help restore observability but are less accurate
            if (!observable) {
                std::cout << "Adding pseudo measurements (load forecasts)...\n";
                // Create load forecasts for each bus (in practice, from historical data)
                // forecasts[i] = forecasted load at bus i (in p.u.)
                std::vector<sle::Real> forecasts(network->getBusCount(), 1.0);
                // Generate pseudo measurements from load forecasts
                // These are treated as P_INJECTION measurements with lower accuracy
                sle::measurements::PseudoMeasurementGenerator::generateFromLoadForecasts(
                    *telemetry, *network, forecasts);
                
                // Final observability check
                observable = analyzer.isFullyObservable(*network, *telemetry);
                std::cout << "After pseudo measurements: " 
                          << (observable ? "Observable" : "Still not observable") << "\n\n";
            }
        }
        
        // ========================================================================
        // STEP 6: Optimal Measurement Placement
        // ========================================================================
        // Find minimum set of additional measurements needed for full observability
        // Uses optimization algorithms (greedy, genetic algorithm, or integer programming)
        // Results can be used for:
        // - Meter placement planning
        // - Budget allocation for new meters
        // - Redundancy analysis
        std::cout << "Finding minimum measurement placement...\n";
        // Returns recommended bus locations for new measurements
        // Each placement specifies: bus ID, measurement type, priority
        auto placements = analyzer.findMinimumMeasurements(*network);
        std::cout << "Recommended measurement locations:\n";
        for (auto busId : placements) {
            std::cout << "  Bus " << busId << "\n";
        }
        
        std::cout << "\nObservability example completed!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

