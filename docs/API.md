# State Estimation API Documentation

## Overview

The State Estimation library provides a comprehensive API for power system state estimation with CUDA acceleration and real-time update capabilities.

**Library Type:** The library is built as a **shared library (DLL on Windows, .so on Linux)** with all public APIs properly exported using the `SLE_API` macro. This ensures proper symbol visibility and allows dynamic linking.

**Latest Updates:**
- **Shared Library (DLL/.so)**: Project now builds as a DLL/shared library with proper symbol export
- Convenience methods for easier usage (`configureForRealTime()`, `loadFromFiles()`, etc.)
- Comparison reports for measured vs estimated values
- 3-level multi-area hierarchy (Region → Area → Zone)
- Enhanced documentation and examples

## Core Classes

### StateEstimator

Main class for state estimation operations.

```cpp
#include <sle/interface/StateEstimator.h>

// Create state estimator instance
// This is the main API class for performing state estimation
sle::interface::StateEstimator estimator;

// Load network model from IEEE Common Format file
// network: Shared pointer to NetworkModel containing buses, branches, and transformers
//          Includes topology, impedances, tap ratios, and all network parameters
auto network = sle::interface::ModelLoader::loadFromIEEE("network.dat");

// Set the network model in the estimator
// The estimator uses this network for admittance matrix construction and power flow calculations
estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));

// Load telemetry measurements from CSV file
// telemetry: Shared pointer to TelemetryData containing all measurement data
//            Includes measurement values, standard deviations, device IDs, and timestamps
//            Measurements are automatically matched to network buses/branches
auto telemetry = sle::interface::MeasurementLoader::loadTelemetry("measurements.csv", *network);

// Set telemetry data in the estimator
// The estimator uses these measurements for the weighted least squares (WLS) estimation
estimator.setTelemetryData(telemetry);

// Configure the Newton-Raphson solver parameters
sle::math::SolverConfig config;
config.tolerance = 1e-6;        // Convergence tolerance: maximum allowed residual norm
                                // Smaller values = more accurate but slower convergence
                                // Typical range: 1e-4 to 1e-8
config.maxIterations = 50;      // Maximum Newton-Raphson iterations
                                // Prevents infinite loops if convergence fails
                                // Typical range: 20-100 iterations
config.useGPU = true;           // Enable CUDA GPU acceleration
                                // true: Use GPU for measurement functions, Jacobian, and solving
                                // false: Use CPU-only implementation (slower but more compatible)

// Apply solver configuration
estimator.setSolverConfig(config);

// Convenience methods for easier configuration
// Configure for real-time operation (fast, relaxed tolerance)
// tolerance: Convergence tolerance (default: 1e-5)
// maxIterations: Maximum iterations (default: 15)
// useGPU: Enable GPU acceleration (default: true)
estimator.configureForRealTime(1e-5, 15, true);

// Configure for offline analysis (accurate, tight tolerance)
// tolerance: Convergence tolerance (default: 1e-8)
// maxIterations: Maximum iterations (default: 50)
// useGPU: Enable GPU acceleration (default: true)
estimator.configureForOffline(1e-8, 50, true);

// Quick setup: load network and telemetry from files in one call
// networkFile: Path to network file (IEEE or JSON format)
// telemetryFile: Path to telemetry file (CSV format)
// Returns: true if successful, false on error
bool success = estimator.loadFromFiles("network.dat", "measurements.csv");

// Check if estimator is ready (has network and telemetry)
if (estimator.isReady()) {
    // Ready to run estimation
}

// Run state estimation
// result: StateEstimationResult containing:
//         - converged: Boolean indicating if estimation converged
//         - iterations: Number of Newton-Raphson iterations performed
//         - finalNorm: Final residual norm (should be < tolerance if converged)
//         - objectiveValue: Final weighted least squares objective function value
//         - state: Estimated state vector (voltage magnitudes and angles for all buses)
//         - message: Human-readable status message
//         - timestamp: Estimation timestamp
auto result = estimator.estimate();
```

### Real-Time Updates

```cpp
// Get reference to telemetry processor for real-time updates
// processor: TelemetryProcessor instance that handles asynchronous measurement updates
//           Provides thread-safe measurement updates without full network reload
auto& processor = estimator.getTelemetryProcessor();

// Start real-time processing thread
// Enables asynchronous measurement updates and background processing
// Measurements can be updated while estimation is running
processor.startRealTimeProcessing();

// Create a real-time measurement update structure
sle::interface::TelemetryUpdate update;
update.deviceId = "METER_001";                    // Device identifier: unique string ID for the measuring device
                                                 // Used to match updates to existing measurements
update.type = sle::MeasurementType::P_INJECTION; // Measurement type: P_INJECTION, Q_INJECTION, V_MAGNITUDE, etc.
                                                 // Determines which measurement function to use
update.value = 1.5;                              // New measurement value in per-unit or engineering units
                                                 // Must match the units expected for the measurement type
update.stdDev = 0.01;                            // Standard deviation of measurement error
                                                 // Used to calculate weight: weight = 1 / (stdDev²)
                                                 // Smaller stdDev = higher weight = more trusted measurement
update.busId = 1;                                // Bus identifier where measurement is located
                                                 // Must match an existing bus ID in the network
update.timestamp = getCurrentTimestamp();        // Unix timestamp in milliseconds
                                                 // Used for temporal ordering and stale data detection

// Update the measurement in real-time
// This updates the existing measurement if deviceId matches, or adds a new measurement
// Thread-safe operation that can be called from multiple threads
processor.updateMeasurement(update);

// Run incremental state estimation (faster than full estimation)
// Uses the previous state estimate as initial guess, reducing iterations needed
// Ideal for real-time applications where state changes are small
// result: Same StateEstimationResult structure as estimate(), but typically with fewer iterations
auto result = estimator.estimateIncremental();

// Convenience methods for accessing estimated state
// Get voltage magnitude at a specific bus (by bus ID)
// busId: Bus ID (1-based, IEEE format)
// Returns: Voltage magnitude in p.u., or 0.0 if bus not found
Real voltage = estimator.getVoltageMagnitude(busId);

// Get voltage angle at a specific bus (by bus ID)
// busId: Bus ID (1-based, IEEE format)
// Returns: Voltage angle in radians, or 0.0 if bus not found
Real angle = estimator.getVoltageAngle(busId);
```

### ModelLoader

Load network models from various formats.

```cpp
#include <sle/interface/ModelLoader.h>

// Load network model from IEEE Common Format file
// network.dat: Text file containing bus and branch data in IEEE standard format
//              Returns: unique_ptr<NetworkModel> containing all buses, branches, and transformers
//              Includes: bus types, impedances, tap ratios, phase shifts, ratings
//              Throws: exception if file not found or format invalid
auto network = sle::interface::ModelLoader::loadFromIEEE("network.dat");

// Load network model from JSON format file
// network.json: JSON file with structured network data
//               Returns: unique_ptr<NetworkModel> (same structure as IEEE format)
//               More human-readable but larger file size than IEEE format
//               Supports additional metadata and nested structures
auto network = sle::interface::ModelLoader::loadFromJSON("network.json");

// Auto-detect file format and load
// Automatically determines format based on file extension and content
// Supports: .dat (IEEE), .json (JSON), .txt (IEEE)
// Returns: unique_ptr<NetworkModel> or nullptr if format not recognized
auto network = sle::interface::ModelLoader::load("network.dat");
```

### MeasurementLoader

Load telemetry and measurement data.

```cpp
#include <sle/interface/MeasurementLoader.h>

// Load telemetry measurements from CSV file
// measurements.csv: CSV file with columns: Type,DeviceId,BusId,Value,StdDev
//                  For branch measurements: Type,DeviceId,FromBus,ToBus,Value,StdDev
// *network: NetworkModel reference used to validate bus/branch IDs
// Returns: shared_ptr<TelemetryData> containing all measurements with device metadata
//          Each measurement includes: type, value, standard deviation, device ID, location, timestamp
auto telemetry = sle::interface::MeasurementLoader::loadFromCSV(
    "measurements.csv", *network);

// Add virtual measurements (zero injection constraints)
// Virtual measurements enforce Kirchhoff's current law: sum of injections = 0 at buses
// Added automatically for buses with no load and no generation (zero injection buses)
// *telemetry: TelemetryData to modify (adds virtual measurements)
// *network: NetworkModel used to identify zero injection buses
// Virtual measurements have very high weight (low stdDev) to enforce constraints exactly
sle::interface::MeasurementLoader::addVirtualMeasurements(*telemetry, *network);

// Add pseudo measurements (load forecasts or historical patterns)
// Pseudo measurements improve observability when real measurements are insufficient
// forecasts: Vector of forecasted load values (one per bus, in p.u.)
//            Used when actual measurements are missing or unreliable
//            Typically have lower weight (higher stdDev) than real measurements
// *telemetry: TelemetryData to modify (adds pseudo measurements)
// *network: NetworkModel used to match forecasts to buses
// Pseudo measurements help restore observability but are less accurate than real measurements
std::vector<sle::Real> forecasts = {1.0, 1.2, 0.8, ...};  // Forecasted loads per bus (p.u.)
sle::interface::MeasurementLoader::addPseudoMeasurements(
    *telemetry, *network, forecasts);
```

## Observability Analysis

```cpp
#include <sle/observability/ObservabilityAnalyzer.h>

// Create observability analyzer instance
// Analyzes measurement redundancy and identifies unobservable buses
sle::observability::ObservabilityAnalyzer analyzer;

// Check if the entire system is fully observable
// observable: Boolean indicating if all buses can be estimated
//            true: System is observable, state estimation can proceed
//            false: System is unobservable, additional measurements needed
// *network: NetworkModel reference (topology and parameters)
// *telemetry: TelemetryData reference (available measurements)
// Uses numerical rank analysis of the measurement Jacobian matrix
bool observable = analyzer.isFullyObservable(*network, *telemetry);

// Get list of observable bus IDs
// observableBuses: Vector of BusId values for buses that can be estimated
//                 Buses with sufficient measurement redundancy
auto observableBuses = analyzer.getObservableBuses(*network, *telemetry);

// Get list of non-observable bus IDs
// nonObservableBuses: Vector of BusId values for buses that cannot be estimated
//                     Buses with insufficient measurements (need additional meters)
auto nonObservableBuses = analyzer.getNonObservableBuses(*network, *telemetry);

// Find minimum set of measurements needed for full observability
// placements: Vector of recommended measurement placements
//             Each placement specifies: measurement type, bus/branch location
//             Minimizes the number of additional measurements required
//             Used for optimal meter placement planning
auto placements = analyzer.findMinimumMeasurements(*network);
```

## Bad Data Detection

```cpp
#include <sle/baddata/BadDataDetector.h>

// Create bad data detector instance
// Implements chi-square test and largest normalized residual (LNR) methods
sle::baddata::BadDataDetector detector;

// Set threshold for normalized residual test
// 3.0: Standard threshold (3-sigma rule)
//      Measurements with normalized residual > 3.0 are flagged as bad
//      Typical range: 2.5 to 4.0 (lower = more sensitive, higher = less sensitive)
//      Normalized residual = residual / sqrt(variance)
detector.setNormalizedResidualThreshold(3.0);

// Detect bad data in measurements
// result: BadDataResult containing:
//         - hasBadData: Boolean indicating if any bad data was found
//         - badDeviceIds: Vector of device IDs with bad measurements
//         - badMeasurementIndices: Vector of measurement indices flagged as bad
//         - chiSquareStatistic: Overall chi-square test statistic
//         - normalizedResiduals: Vector of normalized residuals for each measurement
// *telemetry: TelemetryData with measurements to check
// *state: StateVector from state estimation (used to calculate residuals)
// *network: NetworkModel for measurement function evaluation
auto result = detector.detectBadData(*telemetry, *state, *network);

if (result.hasBadData) {
    // Output number of bad measurements found
    std::cout << "Bad measurements found: " << result.badDeviceIds.size() << "\n";
    
    // Remove bad measurements from telemetry data
    // Modifies *telemetry in-place, removing flagged measurements
    // Should be called before re-running state estimation
    detector.removeBadMeasurements(*telemetry, result);
}
```

## Data Consistency Checking

```cpp
#include <sle/baddata/DataConsistencyChecker.h>

// Create data consistency checker instance
// Validates measurements before state estimation (pre-processing check)
sle::baddata::DataConsistencyChecker checker;

// Check measurement data consistency
// consistency: ConsistencyResult containing:
//              - isConsistent: Boolean indicating if all checks passed
//              - inconsistencies: Vector of string descriptions of issues found
//                                 Examples: "Bus ID 5 not found", "Negative power flow", etc.
// *telemetry: TelemetryData to validate
// *network: NetworkModel used to validate bus/branch IDs and physical limits
// Checks include: bus/branch ID validity, measurement value ranges, unit consistency
auto consistency = checker.checkConsistency(*telemetry, *network);

if (!consistency.isConsistent) {
    // Print all consistency issues found
    // Issues may include: invalid bus IDs, out-of-range values, missing required data
    for (const auto& issue : consistency.inconsistencies) {
        std::cout << "Issue: " << issue << "\n";
    }
}
```

## Robust Estimation

```cpp
#include <sle/math/RobustEstimator.h>

// Create robust estimator instance
// Uses M-estimators to reduce the influence of bad data (outliers)
// More robust than standard WLS when bad data is present
sle::math::RobustEstimator robust;

// Configure robust estimation parameters
sle::math::RobustEstimatorConfig config;
config.weightFunction = sle::math::RobustWeightFunction::HUBER;  // Weight function type:
                                                                  // HUBER: Good balance between robustness and efficiency
                                                                  // BISQUARE: More robust, downweights outliers more aggressively
                                                                  // CAUCHY: Very robust but slower convergence
                                                                  // WELSCH: Similar to Huber but smoother transition
config.tuningConstant = 1.345;  // Tuning constant for the weight function
                                 // Controls the threshold where robust weighting begins
                                 // Typical values: 1.0-2.0 (lower = more robust, higher = closer to WLS)
                                 // 1.345 is standard for Huber estimator (95% efficiency)
robust.setConfig(config);

// Run robust state estimation
// result: StateEstimationResult (same structure as standard estimation)
//         Typically more accurate when bad data is present
// state: Initial StateVector (can be from previous estimation or flat start)
// network: NetworkModel reference
// telemetry: TelemetryData reference
auto result = robust.estimate(state, network, telemetry);
```

## Load Flow

```cpp
#include <sle/math/LoadFlow.h>

// Create load flow solver instance
// Solves power flow equations to find steady-state operating point
// Uses Newton-Raphson method (same as state estimation solver)
sle::math::LoadFlow loadflow;

// Solve power flow
// result: LoadFlowResult containing:
//         - converged: Boolean indicating convergence
//         - iterations: Number of iterations performed
//         - state: StateVector with voltage magnitudes and angles
//         - powerMismatch: Final power mismatch (should be near zero if converged)
// network: NetworkModel reference (must have loads, generation, and slack bus specified)
auto result = loadflow.solve(network);

// Use load flow solution as initial state for state estimation
// Provides better initial guess than flat start (all voltages = 1.0 p.u., angles = 0.0)
// Reduces number of iterations needed for state estimation convergence
// result.state: StateVector from load flow solution
estimator.setInitialState(*result.state);
```

## PMU Support

```cpp
#include <sle/io/PMUData.h>

// Parse C37.118 PMU data from binary file
// pmu_data.bin: Binary file containing PMU data frames in IEEE C37.118 format
//               Includes synchronized phasor measurements (voltage and current)
// frames: Vector of PMUFrame objects, each containing:
//         - timestamp: Synchronized timestamp (microseconds precision)
//         - voltagePhasors: Vector of voltage phasors (magnitude and angle)
//         - currentPhasors: Vector of current phasors (magnitude and angle)
//         - frequency: System frequency measurement
//         - rateOfChangeOfFrequency: ROCOF measurement
auto frames = sle::io::pmu::PMUParser::parseFromFile("pmu_data.bin");

// Convert PMU frame to measurement model
// frames[0]: First PMU frame (or any frame from the vector)
// busId: Bus identifier where the PMU is located
// Returns: MeasurementModel with type V_PHASOR or I_PHASOR
//          Includes: phasor magnitude, angle, and high-accuracy standard deviation
//          PMU measurements typically have very low stdDev (0.001-0.01 p.u.)
auto measurement = sle::io::pmu::PMUParser::convertToMeasurement(frames[0], busId);
```

## Multi-Area Estimation (3-Level Hierarchy)

Multi-area estimation supports a three-level hierarchy: **Region → Area → Zone** for maximum scalability and organization.

### Hierarchy Levels

1. **REGION** (highest): Large-scale interconnections
   - Examples: Eastern Interconnection, Western Interconnection, ERCOT
   - Contains multiple control areas (ISOs/RTOs)
   - Used for inter-regional coordination

2. **AREA** (middle): Control areas, ISOs, RTOs
   - Examples: PJM, NYISO, CAISO, ERCOT
   - Independent system operators or control areas
   - Contains multiple zones (optional)

3. **ZONE** (lowest): Subdivisions within an area
   - Examples: Transmission zones, load zones, operational subdivisions
   - Fine-grained organization within a control area
   - Used for parallel processing and detailed analysis

```cpp
#include <sle/multiarea/MultiAreaEstimator.h>

// Create multi-area estimator instance
// Performs hierarchical or distributed state estimation for large interconnected systems
// Supports 1-level (areas only), 2-level (zones+areas or areas+regions), or 3-level hierarchy
sle::multiarea::MultiAreaEstimator multiArea;

// ========================================================================
// STEP 1: Create Zones (Lowest Level - Optional)
// ========================================================================
// Zones are subdivisions within an area (e.g., transmission zones, load zones)
// Used for fine-grained parallelization and organization
sle::multiarea::Zone zone1;
zone1.name = "NorthZone";              // Zone identifier
zone1.areaName = "PJM";                // Parent area this zone belongs to
for (sle::BusId i = 1; i <= 3; ++i) {
    zone1.buses.insert(i);              // Buses in this zone
}
zone1.network = std::make_shared<sle::model::NetworkModel>(*network);
zone1.telemetry = telemetry;
multiArea.addZone(zone1);               // Add zone to estimator

// ========================================================================
// STEP 2: Create Areas (Middle Level)
// ========================================================================
// Areas are control areas, ISOs, or RTOs (e.g., PJM, NYISO, CAISO)
// Each area can contain multiple zones (optional)
sle::multiarea::Area area1;
area1.name = "PJM";                     // Area identifier
area1.regionName = "Eastern";          // Parent region (optional)
for (sle::BusId i = 1; i <= 7; ++i) {
    area1.buses.insert(i);              // Buses in this area
}
area1.network = std::make_shared<sle::model::NetworkModel>(*network);
area1.telemetry = telemetry;
area1.zones.push_back(zone1);           // Optional: add zones to area
multiArea.addArea(area1);               // Add area to estimator

// ========================================================================
// STEP 3: Create Region (Highest Level - Optional)
// ========================================================================
// Regions are large-scale interconnections (e.g., Eastern, Western, ERCOT)
// Each region contains multiple areas, which may contain zones
sle::multiarea::Region region1;
region1.name = "Eastern";              // Region identifier
for (sle::BusId i = 1; i <= 14; ++i) {
    region1.buses.insert(i);            // All buses in the region
}
region1.network = std::make_shared<sle::model::NetworkModel>(*network);
region1.telemetry = telemetry;
region1.areas.push_back(area1);         // Optional: add areas to region
multiArea.addRegion(region1);           // Add region to estimator

// ========================================================================
// STEP 4: Configure Tie-Line Measurements (Optional)
// ========================================================================
// Tie lines connect different hierarchy levels:
// - Zone-to-zone: Within an area
// - Area-to-area: Within a region
// - Region-to-region: Between interconnections
std::vector<sle::multiarea::TieLineMeasurement> tieLineMeasurements;

sle::multiarea::TieLineMeasurement tie1;
tie1.branchId = 1;                      // Branch ID
tie1.fromArea = 3;                      // From bus (in zone/area)
tie1.toArea = 4;                        // To bus (in zone/area)
tie1.type = sle::MeasurementType::P_FLOW;
tie1.value = 0.5;                       // Measured power flow (p.u.)
tie1.stdDev = 0.01;
tieLineMeasurements.push_back(tie1);

multiArea.setTieLineMeasurements(tieLineMeasurements);

// ========================================================================
// STEP 5: Run Hierarchical Estimation
// ========================================================================
// Hierarchical estimation:
// 1. Estimate zones independently (parallel) - fastest, most parallel
// 2. Coordinate zones within areas (exchange boundary conditions)
// 3. Estimate areas independently (parallel) - medium speed
// 4. Coordinate areas within regions (exchange boundary conditions)
// 5. Estimate regions independently (parallel) - slowest, least parallel
// 6. Coordinate regions (exchange inter-regional boundary conditions)
// 7. Repeat until convergence
auto result = multiArea.estimateHierarchical();

// result: MultiAreaResult containing:
//         - converged: Boolean indicating overall convergence
//         - totalIterations: Total iterations across all levels
//         - zoneResults: Map of zone name → StateEstimationResult (zone-level results)
//         - areaResults: Map of area name → StateEstimationResult (area-level results)
//         - regionResults: Map of region name → StateEstimationResult (region-level results)
//         - tieLineFlows: Vector of tie-line flow values
//         - message: Human-readable status message

// Access zone-level state
auto zoneState = multiArea.getZoneState("NorthZone");

// Access area-level state
auto areaState = multiArea.getAreaState("PJM");

// Access region-level state
auto regionState = multiArea.getRegionState("Eastern");

// ========================================================================
// Alternative: Distributed Estimation (Iterative Coordination)
// ========================================================================
// Distributed estimation iteratively coordinates between levels until convergence
// More accurate but slower than hierarchical estimation
auto distributedResult = multiArea.estimateDistributed(10);  // maxCoordinationIterations = 10
```

## Optimal Measurement Placement

```cpp
#include <sle/observability/OptimalPlacement.h>

// Create optimal placement analyzer instance
// Determines optimal locations for new measurements to maximize observability
sle::observability::OptimalPlacement placement;

// Find optimal measurement placement
// placements: Vector of recommended measurement placements, each containing:
//             - measurementType: Type of measurement to place (V_MAGNITUDE, P_INJECTION, etc.)
//             - busId or branchId: Location for the measurement
//             - priority: Ranking of importance (higher = more critical)
//             - observabilityGain: Improvement in observability from this placement
// network: NetworkModel reference (topology)
// existing: TelemetryData with existing measurements
// maxMeas: Maximum number of new measurements to place (budget constraint)
// budget: Optional cost budget for measurement placement (if costs are specified)
// Uses optimization algorithms (greedy, genetic algorithm, or integer programming)
auto placements = placement.findOptimalPlacement(network, existing, maxMeas, budget);

// Identify critical measurements
// critical: Vector of measurement indices that are critical for observability
//           Removing a critical measurement makes the system unobservable
//           Critical measurements should be protected/redundant
// network: NetworkModel reference
// measurements: TelemetryData to analyze
// Used for measurement redundancy analysis and protection planning
auto critical = placement.identifyCriticalMeasurements(network, measurements);
```


## Comparison Reports

```cpp
#include <sle/io/ComparisonReport.h>

// Compare measured values with estimated values
// comparisons: Vector of ComparisonEntry objects, each containing:
//              - deviceId: Device identifier
//              - measurementType: Type of measurement
//              - location: Bus or branch location
//              - measuredValue: Original measurement value
//              - estimatedValue: Value calculated from estimated state
//              - residual: Difference (measured - estimated)
//              - normalizedResidual: Residual normalized by standard deviation
//              - chiSquareContribution: Contribution to chi-square statistic
// *telemetry: TelemetryData with original measurements
// *result.state: StateVector from state estimation (estimated state)
// *network: NetworkModel for measurement function evaluation
// Used for validation, bad data detection, and measurement accuracy assessment
auto comparisons = sle::io::ComparisonReport::compare(
    *telemetry, *result.state, *network);

// Write comparison report to file
// comparison.txt: Output file path (text format)
// comparisons: ComparisonEntry vector from compare() function
// Creates human-readable report showing measured vs. estimated values and residuals
sle::io::ComparisonReport::writeReport("comparison.txt", comparisons);
```

## Output Formatting

```cpp
#include <sle/io/OutputFormatter.h>
#include <sle/interface/Results.h>

// Create results object from estimation result
// estimationResult: StateEstimationResult from estimator.estimate()
// results: Results object containing formatted output data:
//          - state: StateVector with voltage magnitudes and angles
//          - convergenceInfo: Convergence status, iterations, final norm
//          - busResults: Per-bus results (voltage, power injection, etc.)
//          - branchResults: Per-branch results (power flows, currents)
//          - statistics: Estimation statistics (chi-square, degrees of freedom, etc.)
sle::interface::Results results(estimationResult);

// Format results to JSON string
// json: JSON-formatted string with all estimation results
//       Human-readable, structured format suitable for web APIs and data exchange
//       Includes nested objects for buses, branches, and metadata
std::string json = sle::io::OutputFormatter::formatJSON(results);

// Format results to CSV string
// csv: CSV-formatted string with results in tabular format
//      Suitable for spreadsheet import and data analysis
//      Columns: BusId, VMag, VAngle, PInjection, QInjection, etc.
std::string csv = sle::io::OutputFormatter::formatCSV(results);

// Write results to file in specified format
// results.json: Output file path
// results: Results object to write
// "json": Format identifier ("json", "csv", "xml", "txt")
//         Determines output file format and structure
sle::io::OutputFormatter::writeToFile("results.json", results, "json");
```

