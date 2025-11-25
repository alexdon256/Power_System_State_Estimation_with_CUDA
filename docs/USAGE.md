# Usage Guide

Complete guide to using the State Estimation library, from quick start to advanced real-time operations.

## Quick Start

### Simplest Usage

```cpp
#include <sle/interface/StateEstimator.h>

sle::interface::StateEstimator estimator;
if (estimator.loadFromFiles("network.dat", "measurements.csv")) {
    estimator.configureForRealTime();  // Fast, for real-time
    // or
    estimator.configureForOffline();   // Accurate, for analysis
    
    auto result = estimator.estimate();
    if (result.converged) {
        Real v = estimator.getVoltageMagnitude(5);
        Real theta = estimator.getVoltageAngle(5);
    }
}
```

### Manual Setup

```cpp
#include <sle/interface/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>

// Load network and measurements
auto network = sle::interface::ModelLoader::loadFromIEEE("network.dat");
auto telemetry = sle::interface::MeasurementLoader::loadTelemetry("measurements.csv", *network);

// Optionally load devices (multimeters, voltmeters)
sle::interface::MeasurementLoader::loadDevices("devices.csv", *telemetry, *network);

// Create and configure estimator
sle::interface::StateEstimator estimator;
estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));
estimator.setTelemetryData(telemetry);
estimator.configureForRealTime();

if (estimator.isReady()) {
    auto result = estimator.estimate();
}
```

## Configuration

### Convenience Methods (Recommended)

```cpp
// Real-time: fast, relaxed tolerance
estimator.configureForRealTime();  // tolerance=1e-5, maxIter=15, GPU=true

// Offline: accurate, tight tolerance
estimator.configureForOffline();   // tolerance=1e-8, maxIter=50, GPU=true

// Custom settings
estimator.configureForRealTime(1e-4, 10, true);
estimator.configureForOffline(1e-9, 100, true);
```

### Manual Configuration

```cpp
sle::math::SolverConfig config;
config.tolerance = 1e-6;
config.maxIterations = 50;
config.useGPU = true;
estimator.setSolverConfig(config);
```

## Real-Time Operation

### Basic Real-Time Updates

```cpp
auto telemetry = estimator.getTelemetryData();
telemetry->setNetworkModel(network.get());

// Update measurements on the fly
sle::model::TelemetryUpdate update;
update.deviceId = "METER_1";
update.type = sle::MeasurementType::P_INJECTION;  // Required: device may have multiple measurements
update.value = 1.5;
update.stdDev = 0.01;  // Standard deviation
update.busId = 1;
update.timestamp = getCurrentTimestamp();
telemetry->updateMeasurement(update);

// Run incremental estimation (faster)
auto result = estimator.estimateIncremental();
```

### Dynamic Network Updates

```cpp
// Update bus properties
network->getBus(busId)->setLoad(newPLoad, newQLoad);
network->invalidateAdmittanceMatrix();
estimator.markModelUpdated();

// Update branch properties
network->getBranch(branchId)->setImpedance(newR, newX);
network->invalidateAdmittanceMatrix();
estimator.markModelUpdated();

// Add/remove components
Bus* newBus = network->addBus(100, "New Bus");
network->removeBranch(branchId);
```

### Zero-Copy Topology Reuse

For systems where topology changes less frequently than measurements:

1. **Static Topology**: Jacobian structure built once and persisted on GPU
2. **Dynamic Analogs**: Only measurement values transferred per cycle
3. **Zero-Copy**: Skip re-uploading bus/branch data

This reduces PCIe bandwidth by 90%+ and eliminates symbolic factorization overhead.

## File Formats

### Network Models
- IEEE Common Format: `network.dat` or `network.raw`
- JSON format: `network.json`

### Measurements
- CSV: `measurements.csv` (see [IO_FORMATS.md](IO_FORMATS.md))
- SCADA: `scada.dat`
- PMU (C37.118): `pmu.bin`

### Devices
- CSV: `devices.csv` (multimeters, voltmeters with CT/PT ratios)

## Output and Results

### Accessing Results

```cpp
auto result = estimator.estimate();

// Check convergence
if (result.converged) {
    std::cout << "Converged in " << result.iterations << " iterations\n";
    std::cout << "Final norm: " << result.finalNorm << "\n";
}

// Quick access to estimated values
Real v = estimator.getVoltageMagnitude(busId);
Real theta = estimator.getVoltageAngle(busId);
Real pFlow = estimator.getPowerFlow(branchId);
```

### Comparison Reports

```cpp
#include <sle/io/ComparisonReport.h>

// Compare measured vs estimated values
auto comparisons = sle::io::ComparisonReport::compare(
    *telemetry, *result.state, *network);

// Analyze bad data
int badCount = 0;
for (const auto& comp : comparisons) {
    if (comp.isBad) badCount++;  // Normalized residual > 3.0
}

// Write report
sle::io::ComparisonReport::writeReport("comparison_report.txt", comparisons);
```

## Performance

- **GPU Acceleration**: Enabled by default (5-100x speedup)
- **CPU Fallback**: Automatic if GPU unavailable
- **Incremental Estimation**: Faster for small state changes
- **Memory Pooling**: Reuses GPU buffers for 100-500x faster allocations

## Examples

See `examples/` directory:
- `offlinesetup.cpp` - Offline analysis
- `realtimesetup.cpp` - Real-time production setup
- `hybridsetup.cpp` - Hybrid WLS + robust estimation
- `advancedsetup.cpp` - Advanced features
- `compare_measured_estimated.cpp` - Measured vs estimated comparison
- `observability_example.cpp` - Observability analysis

## Best Practices

1. Use `configureForRealTime()` for real-time, `configureForOffline()` for analysis
2. Check `isReady()` before estimation
3. Use `estimateIncremental()` for frequent updates
4. Process update queue before estimation
5. Monitor convergence in real-time
6. Use comparison reports to validate accuracy

For complete API reference, see [API.md](API.md).

