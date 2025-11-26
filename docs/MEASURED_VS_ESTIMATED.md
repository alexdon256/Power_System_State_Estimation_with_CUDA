# Measured vs Estimated Values

Compare real measured values (from devices) with estimated values (from state estimation).

## Overview

After state estimation:
1. Get **measured values** from devices associated with buses/branches
2. Get **estimated values** from state estimation results
3. Compare to assess estimation quality and detect discrepancies

## Quick Example

```cpp
#include <sle/StateEstimator.h>
#include <sle/model/Bus.h>
#include <sle/model/Branch.h>

// Run estimation
auto result = estimator.estimate();

// Compare voltage at bus
const Bus* bus = network->getBus(busId);
Real measured = bus->getCurrentVoltageMeasurement(*telemetry);
Real estimated = bus->getVPU();
Real difference = std::abs(measured - estimated);

// Compare power flow on branch
const Branch* branch = network->getBranch(branchId);
Real measuredP = branch->getCurrentPowerFlow(*telemetry, pFlow, qFlow);
Real estimatedP = branch->getPMW();
```

## Using Comparison Reports

```cpp
#include <sle/io/ComparisonReport.h>

// Generate comparison report
auto comparisons = sle::io::ComparisonReport::compare(
    *telemetry, *result.state, *network);

// Write report
sle::io::ComparisonReport::writeReport("comparison.txt", comparisons);

// Analyze bad data
for (const auto& comp : comparisons) {
    if (comp.isBad) {  // Normalized residual > 3.0
        std::cout << "Bad measurement: " << comp.deviceId << "\n";
    }
}
```

## Device-Level Analysis

When devices are loaded (via `loadDevices()`), you can:
- Track CT/PT ratios
- Monitor device status
- Validate accuracy based on device specifications
- Analyze device-level discrepancies

## Example Files

See `examples/compare_measured_estimated.cpp` for complete example.

**Input files:**
- `examples/ieee14/network.dat` - Network topology
- `examples/ieee14/measurements.csv` - Measurement data
- `examples/ieee14/devices.csv` - Device definitions (optional)

**Run:**
```bash
./compare_measured_estimated network.dat measurements.csv devices.csv
```
