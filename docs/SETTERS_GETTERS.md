# Setters and Getters Reference

Complete reference for all setters and getters in the State Estimation library.

## Table of Contents

1. [StateEstimator](#stateestimator)
2. [NetworkModel](#networkmodel)
3. [Bus](#bus)
4. [Branch](#branch)
5. [StateVector](#statevector)
6. [TelemetryData](#telemetrydata)
7. [MeasurementModel](#measurementmodel)
8. [DeviceModel](#devicemodel)
9. [TelemetryProcessor](#telemetryprocessor)
10. [RobustEstimator](#robustestimator)
11. [MultiAreaEstimator](#multiareaestimator)
12. [Solver](#solver)
13. [LoadFlow](#loadflow)

---

## StateEstimator

**Header:** `include/sle/interface/StateEstimator.h`

Main class for state estimation operations.

### Setters

#### `void setNetwork(std::shared_ptr<model::NetworkModel> network)`
- **Description:** Set the power system network model (topology, buses, branches)
- **Parameters:** `network` - Shared pointer to NetworkModel
- **Usage:** Called once at initialization or when network topology changes

#### `void setTelemetryData(std::shared_ptr<model::TelemetryData> telemetry)`
- **Description:** Set the telemetry measurements data
- **Parameters:** `telemetry` - Shared pointer to TelemetryData
- **Usage:** Called once at initialization or when measurements are reloaded

#### `void setSolverConfig(const math::SolverConfig& config)`
- **Description:** Configure the Newton-Raphson solver parameters
- **Parameters:** `config` - SolverConfig with tolerance, maxIterations, useGPU
- **Usage:** Configure before running estimation
- **Example:**
  ```cpp
  sle::math::SolverConfig config;
  config.tolerance = 1e-6;
  config.maxIterations = 50;
  config.useGPU = true;
  estimator.setSolverConfig(config);
  ```

#### `void updateNetworkModel(std::shared_ptr<model::NetworkModel> network)`
- **Description:** Update network model in real-time (marks as updated)
- **Parameters:** `network` - New network model
- **Usage:** For real-time topology updates

#### `void updateTelemetryData(std::shared_ptr<model::TelemetryData> telemetry)`
- **Description:** Update telemetry data in real-time (marks as updated)
- **Parameters:** `telemetry` - New telemetry data
- **Usage:** For real-time measurement updates

#### `void markModelUpdated()`
- **Description:** Mark network model as updated (called automatically)
- **Usage:** Internal use, or manual marking after external updates

#### `void markTelemetryUpdated()`
- **Description:** Mark telemetry data as updated (called automatically)
- **Usage:** Internal use, or manual marking after external updates

### Getters

#### `std::shared_ptr<model::NetworkModel> getNetwork() const`
- **Description:** Get the current network model
- **Returns:** Shared pointer to NetworkModel
- **Usage:** Access network topology, buses, branches

#### `std::shared_ptr<model::TelemetryData> getTelemetryData() const`
- **Description:** Get the current telemetry data
- **Returns:** Shared pointer to TelemetryData
- **Usage:** Access measurements, query measurement data

#### `TelemetryProcessor& getTelemetryProcessor()`
- **Description:** Get reference to telemetry processor for real-time updates
- **Returns:** Reference to TelemetryProcessor
- **Usage:** 
  ```cpp
  auto& processor = estimator.getTelemetryProcessor();
  processor.startRealTimeProcessing();
  processor.updateMeasurement(update);
  ```

#### `const math::SolverConfig& getSolverConfig() const`
- **Description:** Get current solver configuration
- **Returns:** Const reference to SolverConfig
- **Usage:** Check current solver settings

#### `std::shared_ptr<model::StateVector> getCurrentState() const`
- **Description:** Get the current state estimate (voltage magnitudes and angles)
- **Returns:** Shared pointer to StateVector
- **Usage:** Access estimated state after estimation
- **Example:**
  ```cpp
  auto state = estimator.getCurrentState();
  Real v = state->getVoltageMagnitude(0);  // Voltage at bus 0
  Real theta = state->getVoltageAngle(0); // Angle at bus 0
  ```

#### `bool isModelUpdated() const`
- **Description:** Check if network model has been updated
- **Returns:** True if model was updated since last estimation
- **Usage:** Determine if re-estimation is needed

#### `bool isTelemetryUpdated() const`
- **Description:** Check if telemetry data has been updated
- **Returns:** True if telemetry was updated since last estimation
- **Usage:** Determine if re-estimation is needed

#### `void configureForRealTime(Real tolerance = 1e-5, int maxIterations = 15, bool useGPU = true)`
- **Description:** Convenience method to configure estimator for real-time operation
- **Parameters:**
  - `tolerance` - Convergence tolerance (default: 1e-5, relaxed for speed)
  - `maxIterations` - Maximum iterations (default: 15, fewer for speed)
  - `useGPU` - Enable GPU acceleration (default: true)
- **Usage:** Quick setup for real-time applications
- **Example:**
  ```cpp
  estimator.configureForRealTime();  // Use defaults
  estimator.configureForRealTime(1e-4, 10, true);  // Custom settings
  ```

#### `void configureForOffline(Real tolerance = 1e-8, int maxIterations = 50, bool useGPU = true)`
- **Description:** Convenience method to configure estimator for offline analysis
- **Parameters:**
  - `tolerance` - Convergence tolerance (default: 1e-8, tight for accuracy)
  - `maxIterations` - Maximum iterations (default: 50, more for accuracy)
  - `useGPU` - Enable GPU acceleration (default: true)
- **Usage:** Quick setup for offline/analysis applications
- **Example:**
  ```cpp
  estimator.configureForOffline();  // Use defaults
  estimator.configureForOffline(1e-9, 100, true);  // Maximum accuracy
  ```

#### `bool loadFromFiles(const std::string& networkFile, const std::string& telemetryFile)`
- **Description:** Load network and telemetry from files in one call
- **Parameters:**
  - `networkFile` - Path to network file (IEEE or JSON format)
  - `telemetryFile` - Path to telemetry file (CSV format)
- **Returns:** True if successful, false on error
- **Usage:** Simplified setup for quick testing
- **Example:**
  ```cpp
  if (estimator.loadFromFiles("network.dat", "measurements.csv")) {
      // Ready to estimate
  }
  ```

#### `Real getVoltageMagnitude(BusId busId) const`
- **Description:** Get voltage magnitude at a specific bus (convenience wrapper)
- **Parameters:** `busId` - Bus ID (1-based, IEEE format)
- **Returns:** Voltage magnitude in p.u., or 0.0 if bus not found
- **Usage:** Quick access to estimated voltage without accessing state vector
- **Example:**
  ```cpp
  Real v = estimator.getVoltageMagnitude(5);  // Voltage at bus 5
  ```

#### `Real getVoltageAngle(BusId busId) const`
- **Description:** Get voltage angle at a specific bus (convenience wrapper)
- **Parameters:** `busId` - Bus ID (1-based, IEEE format)
- **Returns:** Voltage angle in radians, or 0.0 if bus not found
- **Usage:** Quick access to estimated angle without accessing state vector
- **Example:**
  ```cpp
  Real theta = estimator.getVoltageAngle(5);  // Angle at bus 5
  ```

#### `bool isReady() const`
- **Description:** Check if estimator is ready to run (has network and telemetry)
- **Returns:** True if both network and telemetry are set
- **Usage:** Validate before running estimation
- **Example:**
  ```cpp
  if (estimator.isReady()) {
      auto result = estimator.estimate();
  }
  ```

---

## NetworkModel

**Header:** `include/sle/model/NetworkModel.h`

Power system network topology (buses, branches, transformers).

### Setters

#### `void setBaseMVA(Real baseMVA)`
- **Description:** Set base MVA for per-unit calculations
- **Parameters:** `baseMVA` - Base MVA (typically 100.0)
- **Usage:** Set before building admittance matrix

#### `void setReferenceBus(BusId busId)`
- **Description:** Set the reference (slack) bus
- **Parameters:** `busId` - Bus ID of reference bus
- **Usage:** Set once at initialization (angle = 0.0 at reference bus)

### Getters

#### `Bus* getBus(BusId id)` / `const Bus* getBus(BusId id) const`
- **Description:** Get bus by ID
- **Parameters:** `id` - Bus ID
- **Returns:** Pointer to Bus (nullptr if not found)
- **Usage:** Access bus data, modify bus parameters

#### `Branch* getBranch(BranchId id)` / `const Branch* getBranch(BranchId id) const`
- **Description:** Get branch by ID
- **Parameters:** `id` - Branch ID
- **Returns:** Pointer to Branch (nullptr if not found)
- **Usage:** Access branch data, modify branch parameters

#### `std::vector<Bus*> getBuses()` / `std::vector<const Bus*> getBuses() const`
- **Description:** Get all buses
- **Returns:** Vector of bus pointers
- **Usage:** Iterate over all buses

#### `std::vector<Branch*> getBranches()` / `std::vector<const Branch*> getBranches() const`
- **Description:** Get all branches
- **Returns:** Vector of branch pointers
- **Usage:** Iterate over all branches

#### `std::vector<Branch*> getBranchesFromBus(BusId busId)`
- **Description:** Get branches connected from a bus
- **Parameters:** `busId` - Bus ID
- **Returns:** Vector of branch pointers
- **Usage:** Find outgoing branches from a bus

#### `std::vector<Branch*> getBranchesToBus(BusId busId)`
- **Description:** Get branches connected to a bus
- **Parameters:** `busId` - Bus ID
- **Returns:** Vector of branch pointers
- **Usage:** Find incoming branches to a bus

#### `size_t getBusCount() const`
- **Description:** Get number of buses
- **Returns:** Number of buses
- **Usage:** Size of state vector (2 * busCount - 1)

#### `size_t getBranchCount() const`
- **Description:** Get number of branches
- **Returns:** Number of branches
- **Usage:** Iteration bounds, statistics

#### `Real getBaseMVA() const`
- **Description:** Get base MVA
- **Returns:** Base MVA value
- **Usage:** Per-unit conversions

#### `BusId getReferenceBus() const`
- **Description:** Get reference bus ID
- **Returns:** Bus ID of reference bus
- **Usage:** Identify slack bus

---

## Bus

**Header:** `include/sle/model/Bus.h`

Power system bus (node) data.

### Setters

#### `void setType(BusType type)`
- **Description:** Set bus type (PQ, PV, Slack, Isolated)
- **Parameters:** `type` - BusType enum
- **Usage:** Configure bus type for load flow

#### `void setBaseKV(Real baseKV)`
- **Description:** Set base voltage in kV
- **Parameters:** `baseKV` - Base voltage (e.g., 230.0, 500.0)
- **Usage:** Set before loading network

#### `void setVoltage(Real magnitude, Real angle = 0.0)`
- **Description:** Set voltage magnitude and angle
- **Parameters:** 
  - `magnitude` - Voltage magnitude (p.u.)
  - `angle` - Voltage angle (radians, default 0.0)
- **Usage:** Set initial voltage or update after estimation

#### `void setLoad(Real pLoad, Real qLoad)`
- **Description:** Set active and reactive load
- **Parameters:**
  - `pLoad` - Active load (p.u.)
  - `qLoad` - Reactive load (p.u.)
- **Usage:** Set load at bus

#### `void setGeneration(Real pGen, Real qGen)`
- **Description:** Set active and reactive generation
- **Parameters:**
  - `pGen` - Active generation (p.u.)
  - `qGen` - Reactive generation (p.u.)
- **Usage:** Set generation at bus

#### `void setShunt(Real gShunt, Real bShunt)`
- **Description:** Set shunt conductance and susceptance
- **Parameters:**
  - `gShunt` - Shunt conductance (p.u.)
  - `bShunt` - Shunt susceptance (p.u.)
- **Usage:** Set shunt elements (capacitors, reactors)

#### `void setVoltageLimits(Real vMin, Real vMax)`
- **Description:** Set voltage magnitude limits
- **Parameters:**
  - `vMin` - Minimum voltage (p.u., typically 0.9)
  - `vMax` - Maximum voltage (p.u., typically 1.1)
- **Usage:** Set operational limits

### Getters

#### `BusId getId() const`
- **Description:** Get bus ID
- **Returns:** Bus ID
- **Usage:** Identify bus

#### `const std::string& getName() const`
- **Description:** Get bus name
- **Returns:** Bus name string
- **Usage:** Display, logging

#### `BusType getType() const`
- **Description:** Get bus type
- **Returns:** BusType enum
- **Usage:** Check bus type

#### `Real getBaseKV() const`
- **Description:** Get base voltage
- **Returns:** Base voltage in kV
- **Usage:** Per-unit conversions

#### `Real getVoltageMagnitude() const`
- **Description:** Get voltage magnitude
- **Returns:** Voltage magnitude (p.u.)
- **Usage:** Access estimated voltage

#### `Real getVoltageAngle() const`
- **Description:** Get voltage angle
- **Returns:** Voltage angle (radians)
- **Usage:** Access estimated angle

#### `Real getPLoad() const`
- **Description:** Get active load
- **Returns:** Active load (p.u.)
- **Usage:** Access load data

#### `Real getQLoad() const`
- **Description:** Get reactive load
- **Returns:** Reactive load (p.u.)
- **Usage:** Access load data

#### `Real getPGeneration() const`
- **Description:** Get active generation
- **Returns:** Active generation (p.u.)
- **Usage:** Access generation data

#### `Real getQGeneration() const`
- **Description:** Get reactive generation
- **Returns:** Reactive generation (p.u.)
- **Usage:** Access generation data

#### `Real getGShunt() const`
- **Description:** Get shunt conductance
- **Returns:** Shunt conductance (p.u.)
- **Usage:** Access shunt data

#### `Real getBShunt() const`
- **Description:** Get shunt susceptance
- **Returns:** Shunt susceptance (p.u.)
- **Usage:** Access shunt data

#### `Real getVMin() const`
- **Description:** Get minimum voltage limit
- **Returns:** Minimum voltage (p.u.)
- **Usage:** Check voltage limits

#### `Real getVMax() const`
- **Description:** Get maximum voltage limit
- **Returns:** Maximum voltage (p.u.)
- **Usage:** Check voltage limits

#### `bool isZeroInjection() const`
- **Description:** Check if bus has zero injection (no load, no generation)
- **Returns:** True if zero injection
- **Usage:** Virtual measurement generation

---

## Branch

**Header:** `include/sle/model/Branch.h`

Power system branch (transmission line or transformer).

### Setters

#### `void setImpedance(Real r, Real x)`
- **Description:** Set resistance and reactance
- **Parameters:**
  - `r` - Resistance (p.u.)
  - `x` - Reactance (p.u.)
- **Usage:** Set branch impedance

#### `void setCharging(Real b)`
- **Description:** Set charging susceptance
- **Parameters:** `b` - Charging susceptance (p.u.)
- **Usage:** Set line charging (for long lines)

#### `void setRating(Real mvaRating)`
- **Description:** Set MVA rating
- **Parameters:** `mvaRating` - MVA rating
- **Usage:** Set thermal limit

#### `void setTapRatio(Real tap)`
- **Description:** Set transformer tap ratio
- **Parameters:** `tap` - Tap ratio (1.0 for lines, ≠1.0 for transformers)
- **Usage:** Configure transformer (tap > 1.0 increases voltage)
- **Example:**
  ```cpp
  branch->setTapRatio(1.05);  // 5% boost transformer
  ```

#### `void setPhaseShift(Real shift)`
- **Description:** Set phase shift angle
- **Parameters:** `shift` - Phase shift (radians)
- **Usage:** Configure phase-shifting transformer
- **Example:**
  ```cpp
  branch->setPhaseShift(0.1);  // 0.1 rad (~5.7 degrees)
  ```

### Getters

#### `BranchId getId() const`
- **Description:** Get branch ID
- **Returns:** Branch ID
- **Usage:** Identify branch

#### `BusId getFromBus() const`
- **Description:** Get "from" bus ID
- **Returns:** Bus ID
- **Usage:** Identify branch endpoints

#### `BusId getToBus() const`
- **Description:** Get "to" bus ID
- **Returns:** Bus ID
- **Usage:** Identify branch endpoints

#### `Real getR() const`
- **Description:** Get resistance
- **Returns:** Resistance (p.u.)
- **Usage:** Access impedance data

#### `Real getX() const`
- **Description:** Get reactance
- **Returns:** Reactance (p.u.)
- **Usage:** Access impedance data

#### `Real getB() const`
- **Description:** Get charging susceptance
- **Returns:** Charging susceptance (p.u.)
- **Usage:** Access charging data

#### `Real getRating() const`
- **Description:** Get MVA rating
- **Returns:** MVA rating
- **Usage:** Check thermal limits

#### `Real getTapRatio() const`
- **Description:** Get tap ratio
- **Returns:** Tap ratio (1.0 for lines)
- **Usage:** Check if transformer

#### `Real getPhaseShift() const`
- **Description:** Get phase shift
- **Returns:** Phase shift (radians)
- **Usage:** Check if phase-shifting transformer

#### `bool isTransformer() const`
- **Description:** Check if branch is a transformer
- **Returns:** True if tap ≠ 1.0 or phase shift ≠ 0.0
- **Usage:** Identify transformers vs. lines

#### `Complex getAdmittance() const`
- **Description:** Get branch admittance (Y = 1/(R+jX))
- **Returns:** Complex admittance
- **Usage:** Admittance matrix construction

---

## StateVector

**Header:** `include/sle/model/StateVector.h`

State vector containing voltage magnitudes and angles.

### Setters

#### `void setVoltageMagnitude(Index busIdx, Real v)`
- **Description:** Set voltage magnitude at bus index
- **Parameters:**
  - `busIdx` - Bus index (0-based)
  - `v` - Voltage magnitude (p.u.)
- **Usage:** Update state after estimation

#### `void setVoltageAngle(Index busIdx, Real angle)`
- **Description:** Set voltage angle at bus index
- **Parameters:**
  - `busIdx` - Bus index (0-based)
  - `angle` - Voltage angle (radians)
- **Usage:** Update state after estimation

### Getters

#### `Real getVoltageMagnitude(Index busIdx) const`
- **Description:** Get voltage magnitude at bus index
- **Parameters:** `busIdx` - Bus index (0-based)
- **Returns:** Voltage magnitude (p.u.)
- **Usage:** Access estimated voltage

#### `Real getVoltageAngle(Index busIdx) const`
- **Description:** Get voltage angle at bus index
- **Parameters:** `busIdx` - Bus index (0-based)
- **Returns:** Voltage angle (radians)
- **Usage:** Access estimated angle

#### `const std::vector<Real>& getStateVector() const`
- **Description:** Get full state vector [θ₁, ..., θₙ, V₁, ..., Vₙ]ᵀ
- **Returns:** Const reference to state vector
- **Usage:** Access full state for calculations

#### `const std::vector<Real>& getAngles() const`
- **Description:** Get voltage angles [θ₁, ..., θₙ]
- **Returns:** Const reference to angles vector
- **Usage:** Access angles only

#### `const std::vector<Real>& getMagnitudes() const`
- **Description:** Get voltage magnitudes [V₁, ..., Vₙ]
- **Returns:** Const reference to magnitudes vector
- **Usage:** Access magnitudes only

#### `size_t size() const`
- **Description:** Get number of buses
- **Returns:** Number of buses
- **Usage:** Iteration bounds

---

## TelemetryData

**Header:** `include/sle/model/TelemetryData.h`

Container for telemetry measurements.

### Getters

#### `size_t getMeasurementCount() const`
- **Description:** Get number of measurements
- **Returns:** Number of measurements
- **Usage:** Statistics, iteration bounds

#### `const std::vector<std::unique_ptr<MeasurementModel>>& getMeasurements() const`
- **Description:** Get all measurements
- **Returns:** Const reference to measurements vector
- **Usage:** Iterate over all measurements

#### `std::vector<const MeasurementModel*> getMeasurementsByType(MeasurementType type) const`
- **Description:** Get measurements by type
- **Parameters:** `type` - MeasurementType enum
- **Returns:** Vector of measurement pointers
- **Usage:** Filter measurements by type

#### `std::vector<const MeasurementModel*> getMeasurementsByBus(BusId busId) const`
- **Description:** Get measurements at a bus
- **Parameters:** `busId` - Bus ID
- **Returns:** Vector of measurement pointers
- **Usage:** Find measurements at specific bus

#### `std::vector<const MeasurementModel*> getMeasurementsByBranch(BusId fromBus, BusId toBus) const`
- **Description:** Get measurements on a branch
- **Parameters:**
  - `fromBus` - From bus ID
  - `toBus` - To bus ID
- **Returns:** Vector of measurement pointers
- **Usage:** Find measurements on specific branch

---

## MeasurementModel

**Header:** `include/sle/model/MeasurementModel.h`

Individual measurement data.

### Getters

#### `DeviceId getDeviceId() const`
- **Description:** Get device ID
- **Returns:** Device ID
- **Usage:** Identify measurement device

#### `MeasurementType getType() const`
- **Description:** Get measurement type
- **Returns:** MeasurementType enum
- **Usage:** Determine measurement function

#### `Real getValue() const`
- **Description:** Get measurement value
- **Returns:** Measurement value (p.u. or engineering units)
- **Usage:** Access measurement data

#### `Real getStdDev() const`
- **Description:** Get standard deviation
- **Returns:** Standard deviation
- **Usage:** Calculate weight (weight = 1 / stdDev²)

#### `BusId getBusId() const`
- **Description:** Get bus ID (for bus measurements)
- **Returns:** Bus ID
- **Usage:** Identify measurement location

---

## DeviceModel

**Header:** `include/sle/model/DeviceModel.h`

Measurement device model.

### Setters

#### `void setLocation(BusId busId)`
- **Description:** Set bus location
- **Parameters:** `busId` - Bus ID
- **Usage:** Set device location

#### `void setBranchLocation(BusId fromBus, BusId toBus)`
- **Description:** Set branch location
- **Parameters:**
  - `fromBus` - From bus ID
  - `toBus` - To bus ID
- **Usage:** Set device location for branch measurements

#### `void setAccuracy(Real stdDev)`
- **Description:** Set measurement accuracy (standard deviation)
- **Parameters:** `stdDev` - Standard deviation
- **Usage:** Configure device accuracy

#### `void setEnabled(bool enabled)`
- **Description:** Enable/disable device
- **Parameters:** `enabled` - True to enable
- **Usage:** Temporarily disable bad devices

#### `void setTimestamp(int64_t timestamp)`
- **Description:** Set measurement timestamp
- **Parameters:** `timestamp` - Unix timestamp (milliseconds)
- **Usage:** Track measurement time

### Getters

#### `DeviceId getId() const`
- **Description:** Get device ID
- **Returns:** Device ID
- **Usage:** Identify device

#### `const std::string& getName() const`
- **Description:** Get device name
- **Returns:** Device name
- **Usage:** Display, logging

#### `DeviceType getType() const`
- **Description:** Get device type
- **Returns:** DeviceType enum
- **Usage:** Identify device type

#### `BusId getLocation() const`
- **Description:** Get bus location
- **Returns:** Bus ID
- **Usage:** Find device location

#### `BusId getFromBus() const`
- **Description:** Get from bus (for branch devices)
- **Returns:** Bus ID
- **Usage:** Identify branch endpoints

#### `BusId getToBus() const`
- **Description:** Get to bus (for branch devices)
- **Returns:** Bus ID
- **Usage:** Identify branch endpoints

#### `Real getStdDev() const`
- **Description:** Get standard deviation
- **Returns:** Standard deviation
- **Usage:** Calculate weight

#### `Real getVariance() const`
- **Description:** Get variance (stdDev²)
- **Returns:** Variance
- **Usage:** Weight calculation

#### `Real getWeight() const`
- **Description:** Get weight (1 / variance)
- **Returns:** Weight
- **Usage:** WLS weight matrix

#### `bool isEnabled() const`
- **Description:** Check if device is enabled
- **Returns:** True if enabled
- **Usage:** Check device status

#### `int64_t getTimestamp() const`
- **Description:** Get timestamp
- **Returns:** Unix timestamp (milliseconds)
- **Usage:** Check measurement age

---

## TelemetryProcessor

**Header:** `include/sle/interface/TelemetryProcessor.h`

Real-time telemetry update processor.

### Setters

#### `void setTelemetryData(model::TelemetryData* telemetry)`
- **Description:** Set telemetry data container
- **Parameters:** `telemetry` - Pointer to TelemetryData
- **Usage:** Initialize processor

### Getters

#### `bool hasPendingUpdates() const`
- **Description:** Check if updates are pending
- **Returns:** True if queue has updates
- **Usage:** Check update status

#### `int64_t getLatestTimestamp() const`
- **Description:** Get latest measurement timestamp
- **Returns:** Unix timestamp (milliseconds)
- **Usage:** Check data freshness

---

## RobustEstimator

**Header:** `include/sle/math/RobustEstimator.h`

Robust M-estimator for bad data handling.

### Setters

#### `void setConfig(const RobustEstimatorConfig& config)`
- **Description:** Set robust estimator configuration
- **Parameters:** `config` - RobustEstimatorConfig
- **Usage:** Configure M-estimator type, tuning constant, GPU usage
- **Example:**
  ```cpp
  sle::math::RobustEstimatorConfig config;
  config.weightFunction = sle::math::RobustWeightFunction::HUBER;
  config.tuningConstant = 1.345;
  config.useGPU = true;
  robustEstimator.setConfig(config);
  ```

### Getters

#### `const RobustEstimatorConfig& getConfig() const`
- **Description:** Get current configuration
- **Returns:** Const reference to RobustEstimatorConfig
- **Usage:** Check current settings

---

## MultiAreaEstimator

**Header:** `include/sle/multiarea/MultiAreaEstimator.h`

Multi-area hierarchical state estimation.

### Getters

#### `std::shared_ptr<model::StateVector> getZoneState(const std::string& zoneName)`
- **Description:** Get state estimate for a zone
- **Parameters:** `zoneName` - Zone name
- **Returns:** Shared pointer to StateVector
- **Usage:** Access zone-level state

#### `std::shared_ptr<model::StateVector> getAreaState(const std::string& areaName)`
- **Description:** Get state estimate for an area
- **Parameters:** `areaName` - Area name
- **Returns:** Shared pointer to StateVector
- **Usage:** Access area-level state

#### `std::shared_ptr<model::StateVector> getRegionState(const std::string& regionName)`
- **Description:** Get state estimate for a region
- **Parameters:** `regionName` - Region name
- **Returns:** Shared pointer to StateVector
- **Usage:** Access region-level state

---

## Solver

**Header:** `include/sle/math/Solver.h`

Newton-Raphson solver for WLS state estimation.

### Configuration (SolverConfig)

#### `Real tolerance`
- **Description:** Convergence tolerance
- **Default:** 1e-6
- **Range:** 1e-4 to 1e-8
- **Usage:** Set via `setSolverConfig()`

#### `Index maxIterations`
- **Description:** Maximum iterations
- **Default:** 50
- **Range:** 20-100
- **Usage:** Set via `setSolverConfig()`

#### `bool useGPU`
- **Description:** Enable GPU acceleration
- **Default:** true
- **Usage:** Set via `setSolverConfig()`

---

## LoadFlow

**Header:** `include/sle/math/LoadFlow.h`

Power flow solver.

### Configuration (LoadFlowConfig)

#### `Real tolerance`
- **Description:** Power mismatch tolerance
- **Default:** 1e-6
- **Usage:** Set via `setConfig()`

#### `Index maxIterations`
- **Description:** Maximum Newton-Raphson iterations
- **Default:** 100
- **Usage:** Set via `setConfig()`

---

## Quick Reference

### Most Common Operations

```cpp
// Quick setup
estimator.loadFromFiles("network.dat", "measurements.csv");
estimator.configureForRealTime();  // or configureForOffline()

// Check if ready
if (estimator.isReady()) {
    auto result = estimator.estimate();
}

// Get estimated voltage at bus (convenience methods)
Real v = estimator.getVoltageMagnitude(busId);
Real theta = estimator.getVoltageAngle(busId);

// Or access state vector directly
auto state = estimator.getCurrentState();
Real v = state->getVoltageMagnitude(busIndex);
Real theta = state->getVoltageAngle(busIndex);

// Get bus data
auto network = estimator.getNetwork();
auto* bus = network->getBus(busId);
Real baseKV = bus->getBaseKV();

// Get branch data
auto* branch = network->getBranch(branchId);
Real r = branch->getR();
Real x = branch->getX();
bool isTransformer = branch->isTransformer();

// Get measurements
auto telemetry = estimator.getTelemetryData();
size_t nMeas = telemetry->getMeasurementCount();
auto busMeas = telemetry->getMeasurementsByBus(busId);

// Real-time updates
auto& processor = estimator.getTelemetryProcessor();
processor.updateMeasurement(update);

// Compare measured vs estimated
#include <sle/io/ComparisonReport.h>
auto comparisons = sle::io::ComparisonReport::compare(
    *telemetry, *result.state, *network);
sle::io::ComparisonReport::writeReport("report.txt", comparisons);
```

---

## Notes

- All getters are **const** methods (do not modify object state)
- Setters may trigger internal updates (e.g., admittance matrix rebuild)
- Memory: Use shared pointers for network and telemetry data
- Indexing: Bus indices are 0-based, Bus IDs are 1-based (IEEE format)

