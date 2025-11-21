# State Estimation API Documentation

## Overview

The State Estimation library provides a comprehensive API for power system state estimation with CUDA acceleration and real-time update capabilities.

**Library Type:** The library is built as a **shared library (DLL on Windows, .so on Linux)** with all public APIs properly exported using the `SLE_API` macro. This ensures proper symbol visibility and allows dynamic linking.

**Latest Updates:**
- **Shared Library (DLL/.so)**: Project now builds as a DLL/shared library with proper symbol export
- **Computed Values Storage**: Voltage, power, and current values stored in Bus/Branch objects with clean getters
- **GPU Memory Optimization**: Memory pooling and cached device data for 5-20x GPU performance improvement
- **Adjacency Lists**: O(1) branch connectivity queries for 10-100x faster network operations
- **CPU Optimization**: OpenMP parallelization and SIMD vectorization
- Convenience methods for easier usage (`configureForRealTime()`, `loadFromFiles()`, etc.)
- Comparison reports for measured vs estimated values
- 3-level multi-area hierarchy (Region → Area → Zone)
- Enhanced documentation and examples

**Production Readiness:** See [FEATURES.md](FEATURES.md) for complete assessment (85-90% production-ready).

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
// result: RobustResult containing:
//         - converged: Boolean indicating convergence
//         - iterations: Number of IRLS iterations
//         - finalNorm: Final residual norm
//         - objectiveValue: Final objective function value
//         - weights: Vector of final robust weights for each measurement
//         - state: Unique pointer to estimated StateVector
//         - message: Status message
// state: Initial StateVector (can be from previous estimation or flat start)
// network: NetworkModel reference
// telemetry: TelemetryData reference
auto result = robust.estimate(state, network, telemetry);

// Compute values from robust estimation result
if (result.state) {
    bool useGPU = true;
    network->computeVoltEstimates(*result.state, useGPU);
    network->computePowerInjections(*result.state, useGPU);
    network->computePowerFlows(*result.state, useGPU);
    
    // Access computed values via Bus/Branch getters
    auto buses = network->getBuses();
    for (auto* bus : buses) {
        Real vPU = bus->getVPU();           // Voltage in p.u.
        Real vKV = bus->getVKV();           // Voltage in kV
        Real thetaDeg = bus->getThetaDeg(); // Angle in degrees
        Real pMW = bus->getPMW();           // Active power injection in MW
        Real qMVAR = bus->getQMVAR();       // Reactive power injection in MVAR
    }
    
    auto branches = network->getBranches();
    for (auto* branch : branches) {
        Real pMW = branch->getPMW();       // Active power flow in MW
        Real qMVAR = branch->getQMVAR();    // Reactive power flow in MVAR
        Real iAmps = branch->getIAmps();     // Current magnitude in Amperes
        Real iPU = branch->getIPU();         // Current magnitude in p.u.
    }
}

// Analyze robust weights to identify down-weighted measurements
// Measurements with weight < 1.0 were down-weighted (potential bad data)
const auto& measurements = telemetry.getMeasurements();
for (size_t i = 0; i < measurements.size() && i < result.weights.size(); ++i) {
    if (result.weights[i] < 0.99) {
        std::cout << "Measurement " << i << " (" << measurements[i]->getDeviceId()
                  << ") was down-weighted: weight = " << result.weights[i] << "\n";
    }
}
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

## Power and Current Measurements

The library supports comprehensive power and current measurement types for state estimation:

### Power Measurements

**Power Injections** (at buses):
- `P_INJECTION`: Active power injection (MW or p.u.)
- `Q_INJECTION`: Reactive power injection (MVAR or p.u.)
- Used to measure net power at buses (generation - load)

**Power Flows** (on branches):
- `P_FLOW`: Active power flow (MW or p.u.)
- `Q_FLOW`: Reactive power flow (MVAR or p.u.)
- Used to measure power flow on transmission lines and transformers

```cpp
// Power measurements are automatically loaded from CSV files
// Format in measurements.csv:
// P_INJECTION,METER_001,2,-1,-1,40.0,0.1
// P_FLOW,FLOW_001,1,2,0.5,0.05
// Q_FLOW,FLOW_002,1,2,0.3,0.05

// Power measurements are used in the measurement function h(x):
// - P_INJECTION: h(x) = Σ(P_flow_in) - Σ(P_flow_out) + P_gen - P_load
// - Q_INJECTION: h(x) = Σ(Q_flow_in) - Σ(Q_flow_out) + Q_gen - Q_load
// - P_FLOW: h(x) = V_from² * G / tap² - V_from * V_to * (G*cos(θ_diff) + B*sin(θ_diff)) / tap
// - Q_FLOW: h(x) = -V_from² * B / tap² - V_from * V_to * (G*sin(θ_diff) - B*cos(θ_diff)) / tap
```

### Current Measurements

**Current Magnitude** (`I_MAGNITUDE`):
- Measured current magnitude on branches (Amperes or p.u.)
- Typically from current transformers (CTs) on transmission lines
- Provides additional redundancy for power flow estimation

**Current Phasor** (`I_PHASOR`):
- Synchronized current phasor from PMUs (magnitude and angle)
- High accuracy, synchronized measurements
- Provides both magnitude and phase angle information

```cpp
// Current magnitude measurements in CSV format:
// I_MAGNITUDE,CT_001,1,2,0.25,0.02
// Format: Type,DeviceId,BusId,FromBus,ToBus,Value,StdDev
//         - FromBus/ToBus: Branch endpoints for current measurement
//         - Value: Current magnitude in p.u. or Amperes
//         - StdDev: Measurement uncertainty (typically 0.01-0.05 p.u.)

// Current phasor from PMU:
#include <sle/io/PMUData.h>
auto frames = sle::io::pmu::PMUParser::parseFromFile("pmu_data.bin");
auto measurement = sle::io::pmu::PMUParser::convertToMeasurement(frames[0], busId);
// measurement.type will be I_PHASOR for current phasor measurements

// Current measurements are used in the measurement function h(x):
// - I_MAGNITUDE: h(x) = |I| = |(P + jQ) / (V * e^(jθ))|
// - I_PHASOR: h(x) = I = (P + jQ) / (V * e^(jθ))  (complex current)
```

### Measurement Types Summary

| Type | Location | Units | Typical StdDev | Use Case |
|------|----------|-------|----------------|----------|
| `P_INJECTION` | Bus | MW, p.u. | 0.01-0.05 | Net active power at bus |
| `Q_INJECTION` | Bus | MVAR, p.u. | 0.01-0.05 | Net reactive power at bus |
| `P_FLOW` | Branch | MW, p.u. | 0.01-0.05 | Active power flow on line |
| `Q_FLOW` | Branch | MVAR, p.u. | 0.01-0.05 | Reactive power flow on line |
| `I_MAGNITUDE` | Branch | A, p.u. | 0.01-0.05 | Current magnitude on line |
| `I_PHASOR` | Bus/Branch | p.u. (complex) | 0.0001-0.001 | PMU current phasor |
| `V_MAGNITUDE` | Bus | p.u., kV | 0.001-0.01 | Bus voltage magnitude |
| `V_PHASOR` | Bus | p.u. (complex) | 0.0001-0.001 | PMU voltage phasor |

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

## Extracting Computed Values

After state estimation (standard WLS or robust estimation), computed values (voltage, power, current) are stored directly in Bus and Branch objects for easy access. This provides a clean, object-oriented API that works seamlessly with area/zone/region hierarchies.

**Note**: The same value extraction methods work for both standard WLS and robust estimation results. Simply pass the `StateVector` from either estimation method to the compute functions.

### Computing and Storing Values

```cpp
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>

// After state estimation, compute and store all values
// All computations are GPU-accelerated when useGPU=true
// Values are stored in Bus and Branch objects for easy access

// Step 1: Compute voltage estimates (stores in Bus objects)
// Computes: vPU, vKV, thetaRad, thetaDeg for all buses
// useGPU: Enable GPU acceleration (default: false)
network->computeVoltEstimates(*result.state, useGPU);

// Step 2: Compute power injections (stores in Bus objects)
// Computes: P, Q, MW, MVAR injections for all buses
network->computePowerInjections(*result.state, useGPU);

// Step 3: Compute power flows (stores in Branch objects)
// Computes: P, Q, MW, MVAR, I (amps and p.u.) for all branches
network->computePowerFlows(*result.state, useGPU);
```

### Extracting Values from Buses

```cpp
// Get all buses from network
auto buses = network->getBuses();

for (auto* bus : buses) {
    if (bus) {
        // Voltage estimates (computed by computeVoltEstimates)
        Real vPU = bus->getVPU();           // Voltage in per-unit
        Real vKV = bus->getVKV();           // Voltage in kV
        Real thetaRad = bus->getThetaRad();  // Angle in radians
        Real thetaDeg = bus->getThetaDeg();  // Angle in degrees
        
        // Power injections (computed by computePowerInjections)
        Real pInj = bus->getPInjection();        // P injection in p.u.
        Real qInj = bus->getQInjection();        // Q injection in p.u.
        Real pMW = bus->getPInjectionMW();       // P injection in MW
        Real qMVAR = bus->getQInjectionMVAR();  // Q injection in MVAR
        
        // Example: Check for voltage violations
        if (vPU < 0.95 || vPU > 1.05) {
            std::cout << "Bus " << bus->getId() << " voltage violation: " 
                      << vKV << " kV\n";
        }
    }
}
```

### Extracting Values from Branches

```cpp
// Get all branches from network
auto branches = network->getBranches();

for (auto* branch : branches) {
    if (branch) {
        // Power flows (computed by computePowerFlows)
        Real pFlow = branch->getPFlow();     // P flow in p.u.
        Real qFlow = branch->getQFlow();     // Q flow in p.u.
        Real pMW = branch->getPMW();         // P flow in MW
        Real qMVAR = branch->getQMVAR();     // Q flow in MVAR
        
        // Current (computed by computePowerFlows)
        Real iPU = branch->getIPU();         // Current in per-unit
        Real iAmps = branch->getIAmps();      // Current in Amperes
        
        // Example: Check for overload conditions
        Real mvaRating = branch->getRating();
        Real sFlow = std::sqrt(pMW * pMW + qMVAR * qMVAR);
        if (sFlow > mvaRating * 0.9) {  // 90% of rating
            std::cout << "Branch " << branch->getId() << " overload: " 
                      << sFlow << " MVA (rating: " << mvaRating << " MVA)\n";
        }
    }
}
```

### Complete Example: Standard WLS

```cpp
// Run standard WLS state estimation
auto result = estimator.estimate();

if (result.converged && result.state) {
    // Compute all values (GPU-accelerated)
    bool useGPU = true;
    network->computeVoltEstimates(*result.state, useGPU);
    network->computePowerInjections(*result.state, useGPU);
    network->computePowerFlows(*result.state, useGPU);
    
    // Extract and use values
    for (auto* bus : network->getBuses()) {
        Real v = bus->getVPU();
        Real pMW = bus->getPInjectionMW();
        // Use values for monitoring, control, etc.
    }
    
    for (auto* branch : network->getBranches()) {
        Real pMW = branch->getPMW();
        Real iAmps = branch->getIAmps();
        // Use values for overload detection, etc.
    }
}
```

### Complete Example: Robust Estimation

```cpp
#include <sle/math/RobustEstimator.h>

// Run robust state estimation
sle::math::RobustEstimator robustEstimator;
sle::math::RobustEstimatorConfig config;
config.weightFunction = sle::math::RobustWeightFunction::HUBER;
config.tuningConstant = 1.345;
config.tolerance = 1e-6;
config.maxIterations = 50;
config.useGPU = true;
robustEstimator.setConfig(config);

// Use WLS result as initial state (optional but recommended)
auto wlsResult = estimator.estimate();
auto robustState = std::make_unique<sle::model::StateVector>(*wlsResult.state);
auto robustResult = robustEstimator.estimate(*robustState, *network, *telemetry);

if (robustResult.converged && robustResult.state) {
    // Compute all values from robust estimation (GPU-accelerated)
    bool useGPU = true;
    network->computeVoltEstimates(*robustResult.state, useGPU);
    network->computePowerInjections(*robustResult.state, useGPU);
    network->computePowerFlows(*robustResult.state, useGPU);
    
    // Extract and use values (same API as standard WLS)
    for (auto* bus : network->getBuses()) {
        Real v = bus->getVPU();           // Voltage from robust estimation
        Real pMW = bus->getPInjectionMW(); // Power injection from robust estimation
        // Use values for monitoring, control, etc.
    }
    
    for (auto* branch : network->getBranches()) {
        Real pMW = branch->getPMW();      // Power flow from robust estimation
        Real iAmps = branch->getIAmps();   // Current from robust estimation
        // Use values for overload detection, etc.
    }
    
    // Analyze robust weights to identify down-weighted measurements
    const auto& measurements = telemetry->getMeasurements();
    for (size_t i = 0; i < measurements.size() && i < robustResult.weights.size(); ++i) {
        if (robustResult.weights[i] < 0.99) {
            std::cout << "Measurement " << measurements[i]->getDeviceId()
                      << " was down-weighted: weight = " << robustResult.weights[i] << "\n";
        }
    }
}
```

### Benefits

- **Clean API**: Simple getters, no manual calculations
- **GPU Acceleration**: All computations can use GPU (5-20x speedup)
- **Hierarchy Support**: Works seamlessly with area/zone/region (each has its own NetworkModel)
- **Efficient**: Values computed once and stored
- **Type-Safe**: All values properly typed and accessible
- **Real-Time Ready**: Fast access for real-time monitoring and control

