# Usage Guide

Complete guide to using the State Estimation library, from basic concepts to advanced real-time operations.

## What is State Estimation?

State estimation calculates the **voltage magnitude and angle** at every bus in a power system using:
- **Measurements** from sensors (voltmeters, power meters, current transformers)
- **Network topology** (how buses and branches are connected)
- **Mathematical optimization** (Weighted Least Squares)

**Why it matters**: Real power systems have limited sensors. State estimation fills in the gaps, giving you a complete picture of system voltages and power flows.

## Quick Start

The simplest way to get started:

```cpp
#include <sle/interface/StateEstimator.h>

sle::interface::StateEstimator estimator;

// Load network and measurements from files
if (estimator.loadFromFiles("network.dat", "measurements.csv")) {
    // Configure for your use case
    estimator.configureForRealTime();  // Fast, for real-time monitoring
    // OR
    estimator.configureForOffline();   // Accurate, for analysis/planning
    
    // Run estimation
    auto result = estimator.estimate();
    
    if (result.converged) {
        // Get estimated voltage at bus 5
        Real voltage = estimator.getVoltageMagnitude(5);
        Real angle = estimator.getVoltageAngle(5);
    }
}
```

**What happens**: The estimator loads your network topology and measurements, solves the optimization problem on GPU, and returns estimated voltages and angles for all buses.

## Configuration: Real-Time vs Offline

Choose based on your needs:

### Real-Time Mode (Production Monitoring)
- **Use when**: Running continuously in SCADA/EMS systems
- **Speed**: Fast (~300-500 ms per cycle)
- **Accuracy**: Good enough for monitoring (tolerance 1e-5)
- **Best for**: Live system monitoring, alarms, real-time displays

```cpp
estimator.configureForRealTime();  // tolerance=1e-5, maxIter=15
```

### Offline Mode (Analysis/Planning)
- **Use when**: Running one-time studies or analysis
- **Speed**: Slower (~1-2 seconds)
- **Accuracy**: High precision (tolerance 1e-8)
- **Best for**: Planning studies, accuracy validation, research

```cpp
estimator.configureForOffline();   // tolerance=1e-8, maxIter=50
```

**Why the difference?** Real-time mode trades some accuracy for speed. Offline mode prioritizes accuracy over speed.

## Real-Time Operation

### Understanding Measurement Updates

In real-time systems, measurements arrive continuously from SCADA or PMU systems. You update them without reloading everything:

```cpp
// Get telemetry data handle
auto telemetry = estimator.getTelemetryData();
telemetry->setNetworkModel(network.get());

// Create an update structure
sle::model::TelemetryUpdate update;
update.deviceId = "METER_1";                    // Which sensor
update.type = sle::MeasurementType::P_INJECTION; // What it measures
update.value = 1.5;                              // New value (MW)
update.stdDev = 0.01;                            // Uncertainty (1%)
update.busId = 1;                                // Where it's located
update.timestamp = getCurrentTimestamp();        // When measured

// Update the measurement
telemetry->updateMeasurement(update);

// Re-estimate (faster method for measurement-only changes)
auto result = estimator.estimateIncremental();  // ~300-500 ms
```

**What happens**: The measurement value is updated in memory, and incremental estimation reuses the previous solution as a starting point, converging faster.

### Understanding Topology Changes

**Topology** = how the network is connected (which branches are in service).

When a **circuit breaker opens or closes**, the network topology changes. This requires:
1. Rebuilding the Jacobian matrix structure (slower)
2. Full re-estimation (not incremental)

**Automatic Detection**: The system automatically detects topology changes when you update circuit breaker status:

```cpp
// Create a circuit breaker for a branch
auto* cb = network->addCircuitBreaker("CB_1", branchId, fromBus, toBus, "Breaker 1");

// Open the breaker (topology change automatically detected!)
cb->setStatus(false);  // false = open, true = closed

// The system automatically:
//   1. Updates branch status
//   2. Sets topologyChanged_ flag
//   3. Marks model as updated

// Check if topology changed
if (estimator.isTopologyChanged()) {
    // Use full estimation (rebuilds Jacobian structure)
    result = estimator.estimate();  // ~500-700 ms
} else {
    // Use incremental estimation (reuses structure)
    result = estimator.estimateIncremental();  // ~300-500 ms
}
```

**Why automatic detection matters**: You don't need to manually track topology changes or call `markModelUpdated()`. Just update the circuit breaker status, and the system handles the rest.

### Complete Real-Time Loop

Here's how a production real-time loop works:

```cpp
auto telemetry = estimator.getTelemetryData();
telemetry->setNetworkModel(network.get());

while (running) {
    // 1. Receive new measurements from SCADA/PMU
    sle::model::TelemetryUpdate update;
    update.deviceId = receiveDeviceId();
    update.value = receiveValue();
    update.stdDev = receiveUncertainty();
    update.busId = receiveBusId();
    telemetry->updateMeasurement(update);
    
    // 2. Handle circuit breaker status changes (if any)
    if (breakerStatusChanged) {
        auto* cb = network->getCircuitBreakerByBranch(branchId);
        if (cb) {
            cb->setStatus(newStatus);  // Automatically detects topology change
        }
    }
    
    // 3. Run estimation (automatically chooses best method)
    sle::interface::StateEstimationResult result;
    if (estimator.isTopologyChanged()) {
        result = estimator.estimate();  // Full estimation (topology changed)
    } else {
        result = estimator.estimateIncremental();  // Incremental (measurement-only)
    }
    
    // 4. Process results (alarms, displays, etc.)
    if (result.converged) {
        processResults(result);
    } else {
        handleConvergenceFailure(result);
    }
}
```

**Key points**:
- Measurement updates are fast (incremental estimation)
- Topology changes are automatically detected
- The system chooses the right estimation method
- Results are available immediately after convergence

## API Reference

### StateEstimator - Main Class

**Estimation Methods:**
- `estimate()` - Full estimation (rebuilds everything, slower but always accurate)
- `estimateIncremental()` - Incremental estimation (reuses previous solution, faster for small changes)
- `detectBadData()` - Find bad measurements using statistical tests

**Status Checking:**
- `isTopologyChanged()` - Returns true if circuit breaker status changed (requires full estimation)
- `isModelUpdated()` - Returns true if network model changed
- `isTelemetryUpdated()` - Returns true if measurements changed

**Result Access:**
- `getVoltageMagnitude(busId)` - Get estimated voltage magnitude at bus (p.u.)
- `getVoltageAngle(busId)` - Get estimated voltage angle at bus (radians)
- `getCurrentState()` - Get full state vector (all voltages and angles)

### NetworkModel - Network Topology

**Circuit Breakers** (for automatic topology detection):
- `addCircuitBreaker(id, branchId, fromBus, toBus, name)` - Create circuit breaker
- `getCircuitBreakerByBranch(branchId)` - Find breaker for a branch
- `getCircuitBreakers()` - Get all circuit breakers

**Buses** (network nodes):
- `addBus(id, name)` - Add a bus
- `getBus(id)` - Get bus by ID
- `getBusByName(name)` - Get bus by name (fast lookup)

**Branches** (transmission lines/transformers):
- `addBranch(id, fromBus, toBus)` - Add a branch
- `getBranch(id)` - Get branch by ID
- `getBranchByBuses(fromBus, toBus)` - Find branch connecting two buses (fast lookup)

### TelemetryData - Measurements

**Updating Measurements:**
- `updateMeasurement(update)` - Update one measurement
- `updateMeasurements(updates)` - Update multiple measurements at once (faster)
- `addMeasurement(update)` - Add a new measurement

**Querying Measurements:**
- `getDevicesByBus(busId)` - Get all measurement devices at a bus (fast lookup)
- `getDevicesByBranch(fromBus, toBus)` - Get all devices on a branch (fast lookup)
- `getMeasurementCount()` - Get total number of measurements

## Observability & Bad Data

### What is Observability?

**Observability** = Can we estimate all bus voltages with the available measurements?

If the system is **unobservable**, some buses can't be estimated. You need more measurements.

```cpp
#include <sle/observability/ObservabilityAnalyzer.h>

sle::observability::ObservabilityAnalyzer analyzer;

// Check if system is fully observable
bool observable = analyzer.isFullyObservable(*network, *telemetry);

if (!observable) {
    // Find which buses can't be estimated
    auto nonObservableBuses = analyzer.getNonObservableBuses(*network, *telemetry);
    
    // Find minimum measurements needed
    auto placements = analyzer.findMinimumMeasurements(*network);
}
```

### What is Bad Data Detection?

**Bad data** = measurements with large errors (faulty sensors, communication errors, etc.).

Bad data detection finds these problematic measurements using statistical tests:

```cpp
#include <sle/baddata/BadDataDetector.h>

sle::baddata::BadDataDetector detector;

// Set threshold (3.0 = 3 standard deviations, standard choice)
detector.setNormalizedResidualThreshold(3.0);

// Detect bad data after estimation
auto result = detector.detectBadData(*telemetry, *state, *network);

if (result.hasBadData) {
    for (const auto& bad : result.badMeasurements) {
        std::cout << "Bad measurement: " << bad.deviceId 
                  << " (residual: " << bad.normalizedResidual << ")\n";
    }
}
```

**How it works**: Compares measurement residuals (difference between measured and estimated values) to expected statistical distribution. Measurements with residuals > 3 standard deviations are flagged as bad.

## File Formats

**Network Models** (topology and parameters):
- **IEEE Common Format**: `.dat` or `.raw` files (standard power system format)
- **JSON**: `.json` files (human-readable, easier to parse)

**Measurements** (sensor data):
- **CSV**: `.csv` files (simple text format, see [IO_FORMATS.md](IO_FORMATS.md))
- **SCADA**: `.dat` files (SCADA system format)
- **PMU**: `.bin` files (C37.118 binary format for phasor measurement units)

**Devices** (sensor definitions):
- **CSV**: `.csv` files with device info (multimeters, voltmeters, CT/PT ratios)

## Examples

See `examples/` directory for complete working examples:

- **`offlinesetup.cpp`** - Offline analysis with high accuracy
- **`realtimesetup.cpp`** - Production real-time setup with automatic topology detection
- **`hybridsetup.cpp`** - WLS + robust estimation for systems with bad data
- **`advancedsetup.cpp`** - Advanced features (PMU, multi-area, transformers)

## Common Patterns

### Pattern 1: One-Time Analysis

```cpp
estimator.configureForOffline();
auto result = estimator.estimate();
// Process results...
```

### Pattern 2: Real-Time Monitoring

```cpp
estimator.configureForRealTime();
while (running) {
    updateMeasurements();
    auto result = estimator.estimateIncremental();
    processResults(result);
}
```

### Pattern 3: Real-Time with Topology Changes

```cpp
estimator.configureForRealTime();
while (running) {
    updateMeasurements();
    if (breakerChanged) {
        cb->setStatus(newStatus);  // Auto-detects topology change
    }
    auto result = estimator.isTopologyChanged() ? 
        estimator.estimate() : estimator.estimateIncremental();
    processResults(result);
}
```

## Best Practices

1. **Choose the right mode**: Use `configureForRealTime()` for production, `configureForOffline()` for analysis
2. **Check readiness**: Always call `isReady()` before estimation
3. **Use incremental for measurements**: Use `estimateIncremental()` when only measurements changed
4. **Use full for topology**: Use `estimate()` when topology changed (auto-detected via `isTopologyChanged()`)
5. **Update circuit breakers properly**: Use `CircuitBreaker::setStatus()` for automatic topology detection
6. **Monitor convergence**: Check `result.converged` and handle failures
7. **Detect bad data periodically**: Run bad data detection every N cycles
8. **Handle errors**: Always check return values and handle exceptions

## Troubleshooting

**Problem**: Estimation doesn't converge
- **Solution**: Check observability, verify measurements are reasonable, try increasing `maxIterations`

**Problem**: Results seem inaccurate
- **Solution**: Use `configureForOffline()` for higher accuracy, check for bad data

**Problem**: Real-time loop is too slow
- **Solution**: Use `estimateIncremental()` for measurement-only updates, ensure GPU is enabled

**Problem**: Topology changes not detected
- **Solution**: Make sure you're using `CircuitBreaker::setStatus()` and checking `isTopologyChanged()`

For detailed format specifications, see [IO_FORMATS.md](IO_FORMATS.md) and [MODEL_FORMAT.md](MODEL_FORMAT.md).
