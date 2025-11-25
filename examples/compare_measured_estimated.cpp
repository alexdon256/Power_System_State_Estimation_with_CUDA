/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Compare Measured vs Estimated Values Example (Optimized)
 * 
 * This example demonstrates how to compare real measured values (from measurement devices)
 * with estimated values (from state estimation) on buses and branches.
 * 
 * OPTIMIZATIONS:
 * - Pre-computes all estimated values once (fast extraction)
 * - Caches device associations to avoid repeated queries
 * - Batch processing for efficient computation
 * - Efficient data structures for O(1) lookups
 * - Telemetry updates without device modification
 * 
 * Features:
 * - Load network, measurements, and devices
 * - Run state estimation
 * - Update telemetry without modifying devices
 * - Compare measured vs estimated values for buses (voltage, power injections)
 * - Compare measured vs estimated values for branches (power flow, current)
 * - Calculate summary statistics (average/max errors)
 * - Device-level analysis with normalized residuals
 * - Bad data detection flags
 * 
 * Usage:
 *   ./compare_measured_estimated network.dat measurements.csv [devices.csv]
 * 
 * Output:
 *   - Detailed comparison for each bus and branch with measurements
 *   - Summary statistics
 *   - Device-level analysis
 *   - Telemetry update demonstration
 */

#include <sle/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>
#include <sle/interface/TelemetryProcessor.h>
#include <sle/model/Bus.h>
#include <sle/model/Branch.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementDevice.h>
#include <sle/Types.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <limits>

using sle::Real;
using namespace std::chrono;

// Helper function to format comparison results
void printComparison(const std::string& label, Real measured, Real estimated, 
                     Real difference, Real percentError) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  " << std::setw(25) << std::left << label << ": ";
    std::cout << "Measured=" << std::setw(10) << measured;
    std::cout << "  Estimated=" << std::setw(10) << estimated;
    std::cout << "  Diff=" << std::setw(10) << difference;
    std::cout << "  Error=" << std::setw(8) << std::setprecision(2) << percentError << "%" << std::endl;
}

// Helper function to get estimated value for a measurement
Real getEstimatedValue(const sle::model::MeasurementModel* measurement,
                      sle::interface::StateEstimator& estimator,
                      sle::model::NetworkModel* network) {
    Real estimatedValue = 0.0;
    
    switch (measurement->getType()) {
        case sle::MeasurementType::V_MAGNITUDE: {
            BusId busId = measurement->getLocation();
            if (busId >= 0) {
                estimatedValue = estimator.getVoltageMagnitude(busId);
            }
            break;
        }
        case sle::MeasurementType::P_FLOW: {
            BusId fromBus = measurement->getFromBus();
            BusId toBus = measurement->getToBus();
            if (fromBus >= 0 && toBus >= 0) {
                const sle::model::Branch* branch = network->getBranchByBuses(fromBus, toBus);
                if (branch) {
                    estimatedValue = branch->getPFlow();
                }
            }
            break;
        }
        case sle::MeasurementType::Q_FLOW: {
            BusId fromBus = measurement->getFromBus();
            BusId toBus = measurement->getToBus();
            if (fromBus >= 0 && toBus >= 0) {
                const sle::model::Branch* branch = network->getBranchByBuses(fromBus, toBus);
                if (branch) {
                    estimatedValue = branch->getQFlow();
                }
            }
            break;
        }
        case sle::MeasurementType::P_INJECTION: {
            BusId busId = measurement->getLocation();
            if (busId >= 0) {
                const sle::model::Bus* bus = network->getBus(busId);
                if (bus) {
                    estimatedValue = bus->getPInjection();
                }
            }
            break;
        }
        case sle::MeasurementType::Q_INJECTION: {
            BusId busId = measurement->getLocation();
            if (busId >= 0) {
                const sle::model::Bus* bus = network->getBus(busId);
                if (bus) {
                    estimatedValue = bus->getQInjection();
                }
            }
            break;
        }
        case sle::MeasurementType::I_MAGNITUDE: {
            BusId fromBus = measurement->getFromBus();
            BusId toBus = measurement->getToBus();
            if (fromBus >= 0 && toBus >= 0) {
                const sle::model::Branch* branch = network->getBranchByBuses(fromBus, toBus);
                if (branch) {
                    estimatedValue = branch->getIPU();
                }
            }
            break;
        }
        default:
            break;
    }
    
    return estimatedValue;
}

// Hash function for BusId pairs (same as NetworkModel)
struct BusPairHash {
    std::size_t operator()(const std::pair<BusId, BusId>& p) const {
        return std::hash<BusId>{}(p.first) ^ (std::hash<BusId>{}(p.second) << 1);
    }
};

// Optimized comparison structure: Pre-computed estimated values
struct ComparisonCache {
    std::unordered_map<BusId, Real> busVoltages;
    std::unordered_map<BusId, std::pair<Real, Real>> busPowerInjections; // P, Q
    std::unordered_map<std::pair<BusId, BusId>, std::pair<Real, Real>, BusPairHash> branchPowerFlows; // P, Q
    std::unordered_map<std::pair<BusId, BusId>, Real, BusPairHash> branchCurrents;
    
    // Cache device associations
    std::unordered_map<BusId, std::vector<const sle::model::MeasurementDevice*>> busDevices;
    std::unordered_map<std::pair<BusId, BusId>, std::vector<const sle::model::MeasurementDevice*>, BusPairHash> branchDevices;
};

// Fast extraction: Pre-compute all estimated values
void buildComparisonCache(ComparisonCache& cache,
                         sle::interface::StateEstimator& estimator,
                         sle::model::NetworkModel* network,
                         sle::model::TelemetryData* telemetry) {
    auto buses = network->getBuses();
    auto branches = network->getBranches();
    
    // Pre-compute bus values
    for (const auto* bus : buses) {
        BusId busId = bus->getId();
        cache.busVoltages[busId] = estimator.getVoltageMagnitude(busId);
        cache.busPowerInjections[busId] = {bus->getPInjection(), bus->getQInjection()};
        
        // Cache device associations
        auto devices = bus->getAssociatedDevices(*telemetry);
        if (!devices.empty()) {
            cache.busDevices[busId] = devices;
        }
    }
    
    // Pre-compute branch values
    for (const auto* branch : branches) {
        std::pair<BusId, BusId> key = {branch->getFromBus(), branch->getToBus()};
        cache.branchPowerFlows[key] = {branch->getPFlow(), branch->getQFlow()};
        cache.branchCurrents[key] = branch->getIPU();
        
        // Cache device associations
        auto devices = branch->getAssociatedDevices(*telemetry);
        if (!devices.empty()) {
            cache.branchDevices[key] = devices;
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Measured vs Estimated Comparison ===\n\n";
        
        // ========================================================================
        // STEP 1: Load Network and Measurements
        // ========================================================================
        std::string networkFile = (argc > 1) ? argv[1] : "examples/ieee14/network.dat";
        std::string measurementFile = (argc > 2) ? argv[2] : "examples/ieee14/measurements.csv";
        std::string deviceFile = (argc > 3) ? argv[3] : "";
        
        std::cout << "Loading network from: " << networkFile << "\n";
        auto network = sle::interface::ModelLoader::loadFromIEEE(networkFile);
        if (!network) {
            std::cerr << "ERROR: Failed to load network\n";
            return 1;
        }
        std::cout << "  - Loaded " << network->getBusCount() << " buses, "
                  << network->getBranchCount() << " branches\n";
        
        std::cout << "Loading measurements from: " << measurementFile << "\n";
        auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
            measurementFile, *network);
        if (!telemetry) {
            std::cerr << "ERROR: Failed to load telemetry\n";
            return 1;
        }
        std::cout << "  - Loaded " << telemetry->getMeasurementCount() << " measurements\n";
        
        // Load devices if provided
        if (!deviceFile.empty()) {
            std::cout << "Loading devices from: " << deviceFile << "\n";
            try {
                sle::interface::MeasurementLoader::loadDevices(
                    deviceFile, *telemetry, *network);
                std::cout << "  - Loaded " << telemetry->getDevices().size() << " devices\n";
            } catch (const std::exception& e) {
                std::cerr << "  Warning: " << e.what() << "\n";
            }
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 2: Run State Estimation
        // ========================================================================
        std::cout << "=== Running State Estimation ===\n";
        sle::interface::StateEstimator estimator;
        estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));
        estimator.setTelemetryData(telemetry);
        estimator.configureForOffline();  // High accuracy for comparison
        
        auto result = estimator.estimate();
        if (!result.converged) {
            std::cerr << "ERROR: State estimation did not converge\n";
            return 1;
        }
        
        std::cout << "  - Converged in " << result.iterations << " iterations\n";
        std::cout << "  - Final residual norm: " << result.finalNorm << "\n";
        std::cout << "  - Objective value: " << result.objectiveValue << "\n\n";
        
        // ========================================================================
        // STEP 2.5: Build Comparison Cache (Fast Extraction)
        // ========================================================================
        auto startCache = high_resolution_clock::now();
        ComparisonCache cache;
        buildComparisonCache(cache, estimator, network.get(), telemetry.get());
        auto endCache = high_resolution_clock::now();
        auto cacheDuration = duration_cast<microseconds>(endCache - startCache).count();
        std::cout << "=== Cache Built ===\n";
        std::cout << "  - Cache build time: " << cacheDuration << " μs\n\n";
        
        // ========================================================================
        // STEP 2.6: Update Telemetry Without Modifying Devices
        // ========================================================================
        std::cout << "=== Updating Telemetry (Without Device Changes) ===\n";
        sle::interface::TelemetryProcessor processor;
        processor.setTelemetryData(telemetry.get());
        processor.setNetworkModel(network.get());
        
        // Example: Update some measurements
        std::vector<sle::interface::TelemetryUpdate> updates;
        
        // Update voltage measurement (device stays the same, only value changes)
        sle::interface::TelemetryUpdate voltageUpdate;
        voltageUpdate.deviceId = "VM-001";  // Existing device ID
        voltageUpdate.type = sle::MeasurementType::V_MAGNITUDE;
        voltageUpdate.value = 1.06;  // New value
        voltageUpdate.stdDev = 0.01;
        voltageUpdate.busId = 1;
        voltageUpdate.timestamp = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count();
        updates.push_back(voltageUpdate);
        
        // Batch update measurements (fast)
        processor.updateMeasurements(updates);
        std::cout << "  - Updated " << updates.size() << " measurements\n";
        std::cout << "  - Devices unchanged (only measurement values updated)\n";
        
        // Verify update is immediately visible
        const sle::model::Bus* bus1 = network->getBus(1);
        if (bus1) {
            Real updatedVoltage = bus1->getCurrentVoltageMeasurement(*telemetry);
            std::cout << "  - Bus 1 now sees updated voltage: " << updatedVoltage << " p.u.\n";
        }
        std::cout << "\n";
        
        // ========================================================================
        // STEP 3: Compare Bus Measurements vs Estimates (Optimized)
        // ========================================================================
        std::cout << "=== Bus Comparison: Measured vs Estimated ===\n\n";
        
        auto startComparison = high_resolution_clock::now();
        auto buses = network->getBuses();
        for (const auto* bus : buses) {
            BusId busId = bus->getId();
            
            // OPTIMIZED: Get measured voltage from cached devices
            Real measuredVoltage = std::numeric_limits<Real>::quiet_NaN();
            auto busDevicesIt = cache.busDevices.find(busId);
            if (busDevicesIt != cache.busDevices.end()) {
                for (const auto* device : busDevicesIt->second) {
                    const auto& measurements = device->getMeasurements();
                    for (const auto* m : measurements) {
                        if (m->getType() == sle::MeasurementType::V_MAGNITUDE) {
                            measuredVoltage = m->getValue();
                            break;
                        }
                    }
                    if (!std::isnan(measuredVoltage)) break;
                }
            }
            
            // OPTIMIZED: Get estimated voltage from cache
            Real estimatedVoltage = cache.busVoltages[busId];
            
            // Skip if no measurement available
            if (std::isnan(measuredVoltage)) {
                continue;
            }
            
            // Calculate differences
            Real voltageDiff = estimatedVoltage - measuredVoltage;
            Real voltageErrorPercent = (voltageDiff / measuredVoltage) * 100.0;
            
            std::cout << "Bus " << busId << " (" << bus->getName() << "):\n";
            printComparison("Voltage Magnitude (p.u.)", 
                          measuredVoltage, estimatedVoltage, voltageDiff, voltageErrorPercent);
            
            // OPTIMIZED: Get measured power injections from cached devices
            Real measuredP = 0.0, measuredQ = 0.0;
            bool hasPowerMeasurements = false;
            if (busDevicesIt != cache.busDevices.end()) {
                for (const auto* device : busDevicesIt->second) {
                    const auto& measurements = device->getMeasurements();
                    for (const auto* m : measurements) {
                        if (m->getType() == sle::MeasurementType::P_INJECTION) {
                            measuredP = m->getValue();
                            hasPowerMeasurements = true;
                        } else if (m->getType() == sle::MeasurementType::Q_INJECTION) {
                            measuredQ = m->getValue();
                            hasPowerMeasurements = true;
                        }
                    }
                }
            }
            
            if (hasPowerMeasurements) {
                // OPTIMIZED: Get estimated power injections from cache
                auto powerIt = cache.busPowerInjections.find(busId);
                Real estimatedP = powerIt->second.first;
                Real estimatedQ = powerIt->second.second;
                
                Real pDiff = estimatedP - measuredP;
                Real qDiff = estimatedQ - measuredQ;
                Real pErrorPercent = (measuredP != 0.0) ? (pDiff / std::abs(measuredP)) * 100.0 : 0.0;
                Real qErrorPercent = (measuredQ != 0.0) ? (qDiff / std::abs(measuredQ)) * 100.0 : 0.0;
                
                printComparison("P Injection (p.u.)", measuredP, estimatedP, pDiff, pErrorPercent);
                printComparison("Q Injection (p.u.)", measuredQ, estimatedQ, qDiff, qErrorPercent);
            }
            
            std::cout << "\n";
        }
        auto endComparison = high_resolution_clock::now();
        auto comparisonDuration = duration_cast<microseconds>(endComparison - startComparison).count();
        std::cout << "  - Comparison time: " << comparisonDuration << " μs\n\n";
        
        // ========================================================================
        // STEP 4: Compare Branch Measurements vs Estimates (Optimized)
        // ========================================================================
        std::cout << "=== Branch Comparison: Measured vs Estimated ===\n\n";
        
        auto startBranchComparison = high_resolution_clock::now();
        auto branches = network->getBranches();
        for (const auto* branch : branches) {
            BranchId branchId = branch->getId();
            BusId fromBus = branch->getFromBus();
            BusId toBus = branch->getToBus();
            std::pair<BusId, BusId> key = {fromBus, toBus};
            
            // OPTIMIZED: Get measured values from cached devices
            Real measuredPFlow = 0.0, measuredQFlow = 0.0;
            Real measuredCurrent = std::numeric_limits<Real>::quiet_NaN();
            bool hasFlowMeasurements = false;
            
            auto branchDevicesIt = cache.branchDevices.find(key);
            if (branchDevicesIt != cache.branchDevices.end()) {
                for (const auto* device : branchDevicesIt->second) {
                    const auto& measurements = device->getMeasurements();
                    for (const auto* m : measurements) {
                        if (m->getType() == sle::MeasurementType::P_FLOW) {
                            measuredPFlow = m->getValue();
                            hasFlowMeasurements = true;
                        } else if (m->getType() == sle::MeasurementType::Q_FLOW) {
                            measuredQFlow = m->getValue();
                            hasFlowMeasurements = true;
                        } else if (m->getType() == sle::MeasurementType::I_MAGNITUDE) {
                            measuredCurrent = m->getValue();
                        }
                    }
                }
            }
            
            // Skip if no measurements available
            if (!hasFlowMeasurements && std::isnan(measuredCurrent)) {
                continue;
            }
            
            // OPTIMIZED: Get estimated values from cache
            auto flowIt = cache.branchPowerFlows.find(key);
            Real estimatedPFlow = flowIt->second.first;
            Real estimatedQFlow = flowIt->second.second;
            Real estimatedCurrent = cache.branchCurrents[key];
            
            std::cout << "Branch " << branchId << " (Bus " << fromBus 
                      << " -> Bus " << toBus << "):\n";
            
            if (hasFlowMeasurements) {
                Real pDiff = estimatedPFlow - measuredPFlow;
                Real qDiff = estimatedQFlow - measuredQFlow;
                Real pErrorPercent = (measuredPFlow != 0.0) ? 
                    (pDiff / std::abs(measuredPFlow)) * 100.0 : 0.0;
                Real qErrorPercent = (measuredQFlow != 0.0) ? 
                    (qDiff / std::abs(measuredQFlow)) * 100.0 : 0.0;
                
                printComparison("P Flow (p.u.)", measuredPFlow, estimatedPFlow, pDiff, pErrorPercent);
                printComparison("Q Flow (p.u.)", measuredQFlow, estimatedQFlow, qDiff, qErrorPercent);
            }
            
            if (!std::isnan(measuredCurrent)) {
                Real currentDiff = estimatedCurrent - measuredCurrent;
                Real currentErrorPercent = (measuredCurrent != 0.0) ? 
                    (currentDiff / std::abs(measuredCurrent)) * 100.0 : 0.0;
                
                printComparison("Current (p.u.)", measuredCurrent, estimatedCurrent, 
                              currentDiff, currentErrorPercent);
            }
            
            std::cout << "\n";
        }
        auto endBranchComparison = high_resolution_clock::now();
        auto branchComparisonDuration = duration_cast<microseconds>(
            endBranchComparison - startBranchComparison).count();
        std::cout << "  - Branch comparison time: " << branchComparisonDuration << " μs\n\n";
        
        // ========================================================================
        // STEP 5: Summary Statistics (Optimized)
        // ========================================================================
        std::cout << "=== Summary Statistics ===\n\n";
        
        auto startStats = high_resolution_clock::now();
        int busComparisons = 0;
        int branchComparisons = 0;
        double totalVoltageError = 0.0;
        double totalPFlowError = 0.0;
        double totalQFlowError = 0.0;
        double maxVoltageError = 0.0;
        double maxPFlowError = 0.0;
        double maxQFlowError = 0.0;
        
        // OPTIMIZED: Calculate statistics using cache
        for (const auto& [busId, devices] : cache.busDevices) {
            Real measuredVoltage = std::numeric_limits<Real>::quiet_NaN();
            for (const auto* device : devices) {
                const auto& measurements = device->getMeasurements();
                for (const auto* m : measurements) {
                    if (m->getType() == sle::MeasurementType::V_MAGNITUDE) {
                        measuredVoltage = m->getValue();
                        break;
                    }
                }
                if (!std::isnan(measuredVoltage)) break;
            }
            
            if (!std::isnan(measuredVoltage)) {
                Real estimatedVoltage = cache.busVoltages[busId];
                Real error = std::abs(estimatedVoltage - measuredVoltage);
                totalVoltageError += error;
                maxVoltageError = std::max(maxVoltageError, error);
                busComparisons++;
            }
        }
        
        // OPTIMIZED: Calculate statistics for branches using cache
        for (const auto& [key, devices] : cache.branchDevices) {
            Real measuredP = 0.0, measuredQ = 0.0;
            bool hasMeasurements = false;
            
            for (const auto* device : devices) {
                const auto& measurements = device->getMeasurements();
                for (const auto* m : measurements) {
                    if (m->getType() == sle::MeasurementType::P_FLOW) {
                        measuredP = m->getValue();
                        hasMeasurements = true;
                    } else if (m->getType() == sle::MeasurementType::Q_FLOW) {
                        measuredQ = m->getValue();
                        hasMeasurements = true;
                    }
                }
            }
            
            if (hasMeasurements) {
                auto flowIt = cache.branchPowerFlows.find(key);
                Real errorP = std::abs(flowIt->second.first - measuredP);
                Real errorQ = std::abs(flowIt->second.second - measuredQ);
                totalPFlowError += errorP;
                totalQFlowError += errorQ;
                maxPFlowError = std::max(maxPFlowError, errorP);
                maxQFlowError = std::max(maxQFlowError, errorQ);
                branchComparisons++;
            }
        }
        
        auto endStats = high_resolution_clock::now();
        auto statsDuration = duration_cast<microseconds>(endStats - startStats).count();
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Bus Comparisons: " << busComparisons << "\n";
        if (busComparisons > 0) {
            std::cout << "  Average Voltage Error: " 
                      << (totalVoltageError / busComparisons) << " p.u.\n";
            std::cout << "  Maximum Voltage Error: " << maxVoltageError << " p.u.\n";
        }
        
        std::cout << "\nBranch Comparisons: " << branchComparisons << "\n";
        if (branchComparisons > 0) {
            std::cout << "  Average P Flow Error: " 
                      << (totalPFlowError / branchComparisons) << " p.u.\n";
            std::cout << "  Average Q Flow Error: " 
                      << (totalQFlowError / branchComparisons) << " p.u.\n";
            std::cout << "  Maximum P Flow Error: " << maxPFlowError << " p.u.\n";
            std::cout << "  Maximum Q Flow Error: " << maxQFlowError << " p.u.\n";
        }
        
        std::cout << "\n  - Statistics computation time: " << statsDuration << " μs\n";
        std::cout << "\n";
        
        // ========================================================================
        // STEP 6: Device-Level Comparison
        // ========================================================================
        std::cout << "=== Device-Level Comparison ===\n\n";
        
        const auto& devices = telemetry->getDevices();
        for (const auto& pair : devices) {
            if (!pair.second) continue;
            const auto* device = pair.second.get();
            std::cout << "Device: " << device->getId() << " (" << device->getName() << ")\n";
            std::cout << "  Type: " << device->getDeviceType() << "\n";
            std::cout << "  Status: " << static_cast<int>(device->getStatus()) << "\n";
            std::cout << "  Accuracy: " << device->getAccuracy() << "\n";
            
            // Get measurements from this device
            const auto& measurements = device->getMeasurements();
            std::cout << "  Measurements (" << measurements.size() << "):\n";
            
            for (const auto* measurement : measurements) {
                Real measuredValue = measurement->getValue();
                Real stdDev = measurement->getStdDev();
                
                std::cout << "    Type: " << static_cast<int>(measurement->getType()) << "\n";
                std::cout << "    Measured Value: " << measuredValue << " ± " << stdDev << "\n";
                
                // Get corresponding estimated value
                Real estimatedValue = getEstimatedValue(measurement, estimator, network.get());
                
                if (estimatedValue != 0.0 || stdDev > 0.0) {
                    Real difference = estimatedValue - measuredValue;
                    Real normalizedResidual = (stdDev > 0.0) ? (difference / stdDev) : 0.0;
                    
                    std::cout << "    Estimated Value: " << estimatedValue << "\n";
                    std::cout << "    Difference: " << difference << "\n";
                    std::cout << "    Normalized Residual: " << normalizedResidual << "\n";
                    
                    // Flag potential bad data (normalized residual > 3)
                    if (std::abs(normalizedResidual) > 3.0) {
                        std::cout << "    ⚠ WARNING: Large normalized residual - possible bad data!\n";
                    }
                }
                
                std::cout << "\n";
            }
            
            std::cout << "\n";
        }
        
        std::cout << "=== Comparison Complete ===\n";
        
        // Performance summary
        auto totalTime = cacheDuration + comparisonDuration + branchComparisonDuration + statsDuration;
        std::cout << "\n=== Performance Summary ===\n";
        std::cout << "  Cache build: " << cacheDuration << " μs\n";
        std::cout << "  Bus comparison: " << comparisonDuration << " μs\n";
        std::cout << "  Branch comparison: " << branchComparisonDuration << " μs\n";
        std::cout << "  Statistics: " << statsDuration << " μs\n";
        std::cout << "  Total extraction time: " << totalTime << " μs\n";
        std::cout << "  Average per bus: " << (cacheDuration + comparisonDuration) / buses.size() << " μs\n";
        std::cout << "  Average per branch: " << branchComparisonDuration / branches.size() << " μs\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

