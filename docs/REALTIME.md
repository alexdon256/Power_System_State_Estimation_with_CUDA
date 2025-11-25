# Real-Time and Dynamic Updates

Complete guide to real-time operation and dynamic network updates.

## Overview

The State Estimation system supports real-time operation where network models and measurements can be updated on the fly without requiring full system reload. All Bus and Branch properties can be updated dynamically using their comprehensive setter methods.

## Real-Time Architecture

### Zero-Copy Topology Reuse

For systems where topology (switches, breakers) changes less frequently than analog measurements (voltages, flows):

1. **Static Topology**: Jacobian structure and network graph built once and persisted on GPU
2. **Dynamic Analogs**: Only measurement values ($z$) and state vector ($x$) transferred per cycle
3. **Zero-Copy**: Skip re-uploading bus/branch data and re-analyzing matrix structure

This reduces PCIe bandwidth by 90%+ and eliminates symbolic factorization overhead.

### Asynchronous Pipeline

The estimation pipeline runs asynchronously on CUDA streams:
1. Host queues data upload (non-blocking)
2. GPU computes measurement functions $h(x)$ and Jacobian $H(x)$
3. GPU solves linear system $G \Delta x = H^T W r$
4. Host synchronizes only when results are needed

## Real-Time Telemetry Updates

### Basic Usage

```cpp
#include <sle/interface/StateEstimator.h>
#include <sle/model/TelemetryData.h>

// Initialize estimator
sle::interface::StateEstimator estimator;
estimator.setNetwork(network);
estimator.setTelemetryData(telemetry);

// Configure telemetry for real-time updates
auto telemetry = estimator.getTelemetryData();
telemetry->setNetworkModel(network.get());

// Update measurements on the fly
sle::model::TelemetryUpdate update;
update.deviceId = "VM-001";
update.type = sle::MeasurementType::V_MAGNITUDE;
update.value = 1.05;
update.stdDev = 0.01;
update.busId = 5;
update.timestamp = getCurrentTimestamp();
telemetry->updateMeasurement(update);

// Bus immediately sees updated value
const Bus* bus = network->getBus(5);
Real currentVoltage = bus->getCurrentVoltageMeasurement(*telemetry);

// Run incremental estimation (faster)
auto result = estimator.estimateIncremental();
```

**Important**: All query methods query telemetry each time they're called, so they **always return the latest values**. Updates are immediately visible without caching.

### Real-Time Loop Example

```cpp
auto telemetry = estimator.getTelemetryData();
telemetry->setNetworkModel(network.get());

while (running) {
    // Receive telemetry updates from SCADA/PMU
    sle::model::TelemetryUpdate update;
    update.deviceId = receiveDeviceId();
    update.type = receiveMeasurementType();
    update.value = receiveValue();
    update.stdDev = receiveUncertainty();
    update.busId = receiveBusId();
    update.timestamp = getCurrentTimestamp();
    
    // Update measurement
    telemetry->updateMeasurement(update);
    
    // Periodically run estimation
    if (shouldEstimate()) {
        auto result = estimator.estimateIncremental();
        processResult(result);
    }
}
```

## Dynamic Network Updates

### Adding Components

```cpp
// Add a new bus
Bus* newBus = network->addBus(100, "New Bus");
newBus->setBaseKV(230.0);
newBus->setType(BusType::PQ);
newBus->setLoad(50.0, 20.0);  // MW, MVAR

// Add a new branch
Branch* newBranch = network->addBranch(200, 1, 100);
newBranch->setImpedance(0.01, 0.05);  // R, X in p.u.
newBranch->setRating(100.0);  // MVA rating
```

### Removing Components

```cpp
// Remove a bus (automatically invalidates caches)
network->removeBus(100);

// Remove a branch (automatically invalidates caches)
network->removeBranch(200);
```

### Updating Bus Properties

```cpp
Bus* bus = network->getBus(1);
if (bus) {
    bus->setType(BusType::PQ);
    bus->setBaseKV(230.0);
    bus->setVoltage(1.05, 0.0);  // magnitude (p.u.), angle (radians)
    bus->setLoad(50.0, 20.0);  // P_load (MW), Q_load (MVAR)
    bus->setGeneration(100.0, 30.0);  // P_gen (MW), Q_gen (MVAR)
    bus->setShunt(0.01, 0.05);  // G_shunt (p.u.), B_shunt (p.u.)
    bus->setVoltageLimits(0.95, 1.05);  // V_min (p.u.), V_max (p.u.)
}
```

### Updating Branch Properties

```cpp
Branch* branch = network->getBranch(1);
if (branch) {
    branch->setImpedance(0.01, 0.05);  // R (p.u.), X (p.u.)
    branch->setCharging(0.001);  // B (p.u.)
    branch->setRating(100.0);  // MVA rating
    branch->setTapRatio(1.05);  // Tap ratio
    branch->setPhaseShift(0.0);  // Phase shift (radians)
}
```

### Searching Buses by Name

```cpp
// O(1) average-case lookup
Bus* bus = network->getBusByName("Main Substation");
if (bus) {
    bus->setVoltage(1.05, 0.0);
}
```

## Measurement Management

### Adding Measurements

```cpp
// Via TelemetryData directly
auto measurement = std::make_unique<MeasurementModel>(
    MeasurementType::V_MAGNITUDE, 1.05, 0.01, "PMU_001");
measurement->setLocation(1);
telemetry->addMeasurement(std::move(measurement));

// Via TelemetryData (real-time updates)
sle::model::TelemetryUpdate update;
update.deviceId = "PMU_001";
update.type = MeasurementType::V_MAGNITUDE;
update.value = 1.05;
update.stdDev = 0.01;
update.busId = 1;
telemetry->addMeasurement(update);

// Batch updates
std::vector<sle::model::TelemetryUpdate> updates = {...};
telemetry->updateMeasurements(updates);
```

### Updating Measurements

```cpp
// Update existing measurement by device ID
sle::model::TelemetryUpdate update;
update.deviceId = "PMU_001";  // Must match existing device ID
update.value = 1.06;  // New value
update.stdDev = 0.01;
telemetry->updateMeasurement(update);
```

### Removing Measurements

```cpp
// Remove all measurements from a device
size_t removed = telemetry->removeAllMeasurementsFromDevice("PMU_001");

// Remove specific measurement by device ID and type
bool removed = telemetry->removeMeasurement("PMU_001", MeasurementType::V_MAGNITUDE);
```

## Cache Invalidation

All dynamic updates automatically invalidate caches:

- **Network changes** (`addBus`, `removeBus`, `addBranch`, `removeBranch`):
  - Invalidates adjacency lists
  - Invalidates GPU device data
  - Invalidates cached power injection/flow vectors

- **Measurement changes**:
  - Jacobian matrix structure needs rebuilding
  - Measurement vector needs updating
  - State estimator handles this automatically

## Performance Considerations

1. **Incremental Estimation**: Use `estimateIncremental()` for faster updates when state changes are small
2. **Update Frequency**: Balance between update rate and estimation accuracy
3. **GPU Acceleration**: CUDA acceleration is essential for real-time performance
4. **Batch Updates**: Group multiple updates together when possible
5. **Topology Reuse**: Use `reuseStructure=true` flag when topology hasn't changed

## Best Practices

1. Always check `isModelUpdated()` and `isTelemetryUpdated()` before estimation
2. Use incremental estimation for frequent updates
3. Process update queue before estimation
4. Monitor estimation convergence in real-time
5. Handle bad data detection in real-time loop
6. Use convenience methods (`configureForRealTime()`, `loadFromFiles()`) for faster setup
7. Invalidate admittance matrix after network changes: `network->invalidateAdmittanceMatrix()`
8. Mark model updated: `estimator.markModelUpdated()` after network changes

## Complete Example

```cpp
#include <sle/interface/StateEstimator.h>

StateEstimator estimator;
estimator.setNetwork(network);
estimator.setTelemetryData(telemetry);
estimator.configureForRealTime();

auto telemetry = estimator.getTelemetryData();
telemetry->setNetworkModel(network.get());

while (running) {
    // 1. Add new bus dynamically
    Bus* newBus = network->addBus(999, "Dynamic Bus");
    newBus->setType(BusType::PQ);
    newBus->setBaseKV(230.0);
    newBus->setLoad(10.0, 5.0);
    
    // 2. Add branch to new bus
    Branch* newBranch = network->addBranch(888, 1, 999);
    newBranch->setImpedance(0.02, 0.1);
    newBranch->setRating(50.0);
    
    // 3. Update existing bus
    Bus* bus = network->getBus(1);
    if (bus) {
        bus->setLoad(60.0, 25.0);
        bus->setGeneration(120.0, 40.0);
    }
    
    // 4. Update measurement
    sle::model::TelemetryUpdate update;
    update.deviceId = "PMU_999";
    update.type = MeasurementType::V_MAGNITUDE;
    update.value = 1.02;
    update.busId = 999;
    telemetry->updateMeasurement(update);
    
    // 5. Process updates and re-estimate
    network->invalidateAdmittanceMatrix();
    estimator.markModelUpdated();
    auto result = estimator.estimateIncremental();
}
```
