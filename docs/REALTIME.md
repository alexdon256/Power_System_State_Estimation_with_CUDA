# Real-Time State Estimation Guide

## Overview

The State Estimation system supports real-time operation where network models and measurements can be updated on the fly without requiring full system reload.

## Real-Time Architecture

The system is optimized for high-frequency updates using a zero-copy topology reuse architecture.

### Zero-Copy Topology Reuse

For systems where the topology (switches, breakers) changes less frequently than analog measurements (voltages, flows):

1.  **Static Topology**: The Jacobian structure and network graph are built once and persisted on the GPU.
2.  **Dynamic Analogs**: Only measurement values ($z$) and state vector ($x$) are transferred per cycle.
3.  **Zero-Copy**: Skip re-uploading bus/branch data and re-analyzing matrix structure.

This reduces PCIe bandwidth usage by over 90% and eliminates symbolic factorization overhead.

### Asynchronous Pipeline

The estimation pipeline runs asynchronously on CUDA streams:
1.  Host queues data upload (non-blocking).
2.  GPU computes measurement functions $h(x)$ and Jacobian $H(x)$.
3.  GPU solves linear system $G \Delta x = H^T W r$.
4.  Host synchronizes only when results are needed.

### TelemetryProcessor

The `TelemetryProcessor` handles asynchronous measurement updates:

- Update queue
- Background processing thread
- Automatic timestamp tracking
- Batch update support

### StateEstimator Updates

The `StateEstimator` supports:

- Incremental estimation (faster convergence using previous state)
- Model update detection
- Telemetry update detection

## Usage Example

```cpp
#include <sle/interface/StateEstimator.h>
#include <sle/interface/TelemetryProcessor.h>

// Initialize estimator
sle::interface::StateEstimator estimator;
estimator.setNetwork(network);
estimator.setTelemetryData(telemetry);

// Start real-time processing
auto& processor = estimator.getTelemetryProcessor();
processor.startRealTimeProcessing();

// In your real-time loop
while (running) {
    // Receive telemetry updates from SCADA/PMU
    sle::interface::TelemetryUpdate update;
    update.deviceId = receiveDeviceId();
    update.type = receiveMeasurementType();
    update.value = receiveValue();
    update.stdDev = receiveUncertainty();
    update.busId = receiveBusId();
    update.timestamp = getCurrentTimestamp();
    
    // Queue update
    processor.updateMeasurement(update);
    
    // Periodically run estimation
    if (shouldEstimate()) {
        auto result = estimator.estimateIncremental();
        processResult(result);
    }
}

// Stop processing
processor.stopRealTimeProcessing();
```

## Network Model Updates

```cpp
// Update network model in real-time
auto updatedNetwork = loadUpdatedNetwork();
estimator.updateNetworkModel(updatedNetwork);

// Update specific bus
network->getBus(busId)->setLoad(newPLoad, newQLoad);
network->invalidateAdmittanceMatrix();  // Mark for rebuild
estimator.markModelUpdated();

// Update branch
network->getBranch(branchId)->setImpedance(newR, newX);
network->invalidateAdmittanceMatrix();
estimator.markModelUpdated();
```

## Performance Considerations

1. **Incremental Estimation**: Use `estimateIncremental()` for faster updates when state changes are small
2. **Update Frequency**: Balance between update rate and estimation accuracy
3. **GPU Acceleration**: CUDA acceleration is essential for real-time performance
4. **Batch Updates**: Group multiple updates together when possible

## Convenience Methods

The `StateEstimator` provides convenience methods for easier real-time setup:

```cpp
// Quick configuration for real-time operation
estimator.configureForRealTime();  // Uses defaults: tolerance=1e-5, maxIter=15, GPU=true

// Quick setup from files
if (estimator.loadFromFiles("network.dat", "measurements.csv")) {
    // Ready to estimate
}

// Check if ready
if (estimator.isReady()) {
    auto result = estimator.estimate();
}

// Quick access to estimated voltages
Real v = estimator.getVoltageMagnitude(busId);
Real theta = estimator.getVoltageAngle(busId);
```

## Comparison Reports

Compare measured vs estimated values to validate estimation and detect bad data:

```cpp
#include <sle/io/ComparisonReport.h>

// After estimation, compare measured vs estimated
auto comparisons = sle::io::ComparisonReport::compare(
    *telemetry, *result.state, *network);

// Analyze results
int badCount = 0;
for (const auto& comp : comparisons) {
    if (comp.isBad) badCount++;  // Normalized residual > 3.0
}

// Write report to file
sle::io::ComparisonReport::writeReport("comparison_report.txt", comparisons);
```

## Best Practices

1. Always check `isModelUpdated()` and `isTelemetryUpdated()` before estimation
2. Use incremental estimation for frequent updates
3. Process update queue before estimation
4. Monitor estimation convergence in real-time
5. Handle bad data detection in real-time loop
6. Use comparison reports to validate estimation accuracy
7. Use convenience methods (`configureForRealTime()`, `loadFromFiles()`) for faster setup

