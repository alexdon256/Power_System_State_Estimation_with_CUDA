# Measurement Devices

This document describes how to model physical measurement devices (multimeters, voltmeters) and associate them with measurements. Devices can be loaded from CSV files and measurements are automatically linked to their devices.

## Loading Devices from Files

### CSV Format

Devices can be loaded from CSV files using `MeasurementLoader::loadDevices()`:

```cpp
#include <sle/interface/MeasurementLoader.h>

// Load devices from CSV file
sle::interface::MeasurementLoader::loadDevices("devices.csv", telemetry, network);
```

**CSV Format:**
```
deviceType,deviceId,name,location,ctRatio,ptRatio,accuracy
MULTIMETER,MM-001,Branch 1 Multimeter,2:3,100.0,1000.0,0.01
VOLTMETER,VM-005,Bus 5 Voltmeter,5,,1000.0,0.005
```

**Fields:**
- `deviceType`: `MULTIMETER` or `MM` for multimeters, `VOLTMETER` or `VM` for voltmeters
- `deviceId`: Unique device identifier (must match measurement deviceId)
- `name`: Human-readable device name
- `location`: 
  - For multimeters: `fromBus:toBus` (e.g., `2:3`) or branchId
  - For voltmeters: busId
- `ctRatio`: Current Transformer ratio (multimeters only, default: 1.0)
- `ptRatio`: Potential Transformer ratio (default: 1.0)
- `accuracy`: Device accuracy as standard deviation multiplier (default: 0.01)

### Complete Example: Loading Devices and Measurements

```cpp
#include <sle/interface/MeasurementLoader.h>
#include <sle/model/TelemetryData.h>

// Load measurements first
auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
    "measurements.csv", network
);

// Then load devices (measurements will be automatically linked by location matching)
// Measurements are linked to devices based on:
// - For voltmeters: matching busId
// - For multimeters: matching fromBus/toBus
sle::interface::MeasurementLoader::loadDevices(
    "devices.csv", *telemetry, network
);

// Devices are now linked to measurements via location matching
// Note: If measurements were loaded with deviceId in CSV but devices weren't loaded yet,
// they will be linked when devices are loaded based on location matching
```

## Overview

Measurement devices represent physical equipment that produces measurements:
- **Multimeters**: Measure power flow on branches using Current Transformers (CT) and Potential Transformers (PT)
- **Voltmeters**: Measure voltage magnitude at buses using Potential Transformers (PT)

## Device Model

### Base Class: `MeasurementDevice`

All measurement devices inherit from `MeasurementDevice` and provide:
- Device identification (ID, name)
- Device status (OPERATIONAL, CALIBRATING, MAINTENANCE, FAILED, OFFLINE)
- Device accuracy (standard deviation multiplier)
- Association with measurements produced by the device

### Multimeter

A multimeter sits on a branch and measures power flow using CT and PT:

```cpp
#include <sle/model/MeasurementDevice.h>
#include <sle/model/TelemetryData.h>

// Create a multimeter on branch 1 (from bus 2 to bus 3)
// CT ratio: 100:1, PT ratio: 1000:1
auto multimeter = std::make_unique<Multimeter>(
    "MM-001",           // Device ID
    1,                  // Branch ID
    2,                  // From bus
    3,                  // To bus
    100.0,              // CT ratio (100:1)
    1000.0,             // PT ratio (1000:1)
    "Branch 1 Multimeter" // Name
);

// Add to telemetry data
telemetryData.addDevice(std::move(multimeter));

// Create measurements from this multimeter
auto pFlow = std::make_unique<MeasurementModel>(
    MeasurementType::P_FLOW,
    50.0,               // Measured value (already scaled by CT/PT)
    0.5                 // Standard deviation
);
pFlow->setBranchLocation(2, 3);
// Pass device ID to addMeasurement for automatic linking
telemetryData.addMeasurement(std::move(pFlow), "MM-001");

auto qFlow = std::make_unique<MeasurementModel>(
    MeasurementType::Q_FLOW,
    10.0,
    0.5
);
qFlow->setBranchLocation(2, 3);
// Same device ID - both measurements from same multimeter
telemetryData.addMeasurement(std::move(qFlow), "MM-001");
```

### Voltmeter

A voltmeter measures voltage at a bus:

```cpp
// Create a voltmeter on bus 5
// PT ratio: 1000:1
auto voltmeter = std::make_unique<Voltmeter>(
    "VM-005",           // Device ID
    5,                  // Bus ID
    1000.0,             // PT ratio (1000:1)
    "Bus 5 Voltmeter"   // Name
);

telemetryData.addDevice(std::move(voltmeter));

// Create voltage measurement from this voltmeter
auto vMag = std::make_unique<MeasurementModel>(
    MeasurementType::V_MAGNITUDE,
    1.05,               // Measured value (already scaled by PT)
    0.01                // Standard deviation
);
vMag->setLocation(5);
// Pass device ID to addMeasurement for automatic linking
telemetryData.addMeasurement(std::move(vMag), "VM-005");
```

## Querying Devices

### Get Device by ID

```cpp
// Get a specific device
const MeasurementDevice* device = telemetryData.getDevice("MM-001");
if (device) {
    std::cout << "Device: " << device->getName() << std::endl;
    std::cout << "Status: " << (device->isOperational() ? "Operational" : "Not Operational") << std::endl;
    std::cout << "Accuracy: " << device->getAccuracy() << std::endl;
    
    // Get all measurements from this device
    const auto& measurements = device->getMeasurements();
    std::cout << "Produces " << measurements.size() << " measurements" << std::endl;
}
```

### Get Devices by Location

```cpp
// Get all voltmeters on a bus
auto voltmeters = telemetryData.getDevicesByBus(5);
for (const auto* device : voltmeters) {
    const Voltmeter* vm = dynamic_cast<const Voltmeter*>(device);
    if (vm) {
        std::cout << "Voltmeter " << vm->getId() 
                  << " PT ratio: " << vm->getPTRatio() << std::endl;
    }
}

// Get all multimeters on a branch
auto multimeters = telemetryData.getDevicesByBranch(2, 3);
for (const auto* device : multimeters) {
    const Multimeter* mm = dynamic_cast<const Multimeter*>(device);
    if (mm) {
        std::cout << "Multimeter " << mm->getId()
                  << " CT ratio: " << mm->getCTRatio()
                  << " PT ratio: " << mm->getPTRatio() << std::endl;
    }
}
```

## Transformer Ratio Handling

### Applying Transformer Ratios

When a device measures a value, it applies transformer ratios:

```cpp
Multimeter mm("MM-001", 1, 2, 3, 100.0, 1000.0);

// Raw measurement value (before transformer scaling)
Real rawValue = 0.05;  // 0.05 pu

// Apply transformer ratios to get measured value
Real measuredValue = mm.applyTransformerRatio(rawValue);
// measuredValue = 0.05 * 100 * 1000 = 5000.0

// Reverse operation (convert measured value back to raw)
Real backToRaw = mm.reverseTransformerRatio(measuredValue);
// backToRaw = 5000.0 / (100 * 1000) = 0.05
```

### Device Accuracy

Device accuracy affects measurement standard deviation:

```cpp
// Set device accuracy (1% = 0.01)
multimeter->setAccuracy(0.01);

// When creating measurements, use device accuracy to compute stdDev
Real baseStdDev = 0.5;
Real adjustedStdDev = baseStdDev * (1.0 + multimeter->getAccuracy());
// adjustedStdDev = 0.5 * 1.01 = 0.505
```

## Device Status Management

Track device operational status:

```cpp
#include <sle/model/MeasurementDevice.h>

// Set device status
multimeter->setStatus(DeviceStatus::MAINTENANCE);

// Check if device is operational
if (!multimeter->isOperational()) {
    // Skip measurements from this device or mark them as suspect
    for (auto* measurement : multimeter->getMeasurements()) {
        measurement->setStatus(MeasurementStatus::SUSPECT);
    }
}

// Device statuses:
// - OPERATIONAL: Device working normally
// - CALIBRATING: Device being calibrated
// - MAINTENANCE: Device under maintenance
// - FAILED: Device has failed
// - OFFLINE: Device offline/disabled
```

## Complete Example

```cpp
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementDevice.h>
#include <sle/model/MeasurementModel.h>

// Create telemetry data container
TelemetryData telemetry;

// 1. Add multimeter on branch 1 (bus 2 -> bus 3)
auto multimeter = std::make_unique<Multimeter>(
    "MM-BR1", 1, 2, 3, 100.0, 1000.0, "Branch 1 Multimeter"
);
telemetry.addDevice(std::move(multimeter));

// 2. Add measurements from multimeter
auto pFlow = std::make_unique<MeasurementModel>(
    MeasurementType::P_FLOW, 50.0, 0.5
);
pFlow->setBranchLocation(2, 3);
// Pass device ID to addMeasurement for automatic linking
telemetry.addMeasurement(std::move(pFlow), "MM-BR1");

auto qFlow = std::make_unique<MeasurementModel>(
    MeasurementType::Q_FLOW, 10.0, 0.5
);
qFlow->setBranchLocation(2, 3);
// Same device ID - both measurements from same multimeter
telemetry.addMeasurement(std::move(qFlow), "MM-BR1");

// 3. Add voltmeter on bus 5
auto voltmeter = std::make_unique<Voltmeter>(
    "VM-BUS5", 5, 1000.0, "Bus 5 Voltmeter"
);
telemetry.addDevice(std::move(voltmeter));

// 4. Add voltage measurement from voltmeter
auto vMag = std::make_unique<MeasurementModel>(
    MeasurementType::V_MAGNITUDE, 1.05, 0.01
);
vMag->setLocation(5);
// Pass device ID to addMeasurement for automatic linking
telemetry.addMeasurement(std::move(vMag), "VM-BUS5");

// 5. Query devices
const auto& deviceMap = telemetry.getDevices();
auto deviceIt = deviceMap.find("MM-BR1");
if (deviceIt != deviceMap.end() && deviceIt->second) {
    const Multimeter* mm = dynamic_cast<const Multimeter*>(deviceIt->second.get());
    if (mm) {
        std::cout << "Multimeter From Bus: " << mm->getFromBus() << std::endl;
        std::cout << "Multimeter To Bus: " << mm->getToBus() << std::endl;
        std::cout << "Produces " << mm->getMeasurements().size() 
                  << " measurements" << std::endl;
    }
}
```

## Benefits

1. **Device-Level Tracking**: Know which physical device produced each measurement
2. **Transformer Modeling**: Properly account for CT/PT ratios
3. **Device Status**: Track device health and maintenance
4. **Bad Data Detection**: Use device characteristics for better bad data detection
5. **Calibration**: Model device calibration and accuracy
6. **Multiple Measurements**: One device can produce multiple measurement types (P, Q, I, V)

## Integration with State Estimation

The device model integrates seamlessly with existing state estimation:

- Measurements are still stored in `TelemetryData` as before
- Devices provide metadata and transformer ratios
- State estimation algorithms work unchanged
- Bad data detection can use device characteristics
- Device status affects measurement validity

## Querying Telemetry from Buses and Branches

Buses and branches can now query telemetry from their associated devices:

### Bus Telemetry Queries

```cpp
// Get all devices (voltmeters) associated with a bus
const Bus* bus = network.getBus(5);
auto devices = bus->getAssociatedDevices(telemetry);
for (const auto* device : devices) {
    const Voltmeter* vm = dynamic_cast<const Voltmeter*>(device);
    if (vm) {
        std::cout << "Voltmeter: " << vm->getId() << std::endl;
    }
}

// Get all measurements from devices on this bus
auto measurements = bus->getMeasurementsFromDevices(telemetry);
for (const auto* m : measurements) {
    std::cout << "Measurement type: " << static_cast<int>(m->getType())
              << " value: " << m->getValue() << std::endl;
}

// Get specific measurement type (e.g., voltage magnitude)
auto voltageMeasurements = bus->getMeasurementsFromDevices(
    telemetry, MeasurementType::V_MAGNITUDE
);
```

### Branch Telemetry Queries

```cpp
// Get all devices (multimeters) associated with a branch
const Branch* branch = network.getBranch(1);
auto devices = branch->getAssociatedDevices(telemetry);
for (const auto* device : devices) {
    const Multimeter* mm = dynamic_cast<const Multimeter*>(device);
    if (mm) {
        std::cout << "Multimeter: " << mm->getId() << std::endl;
        std::cout << "CT ratio: " << mm->getCTRatio() << std::endl;
        std::cout << "PT ratio: " << mm->getPTRatio() << std::endl;
    }
}

// Get all measurements from devices on this branch
auto measurements = branch->getMeasurementsFromDevices(telemetry);
for (const auto* m : measurements) {
    std::cout << "Measurement: " << m->getValue() << std::endl;
}

// Get power flow measurements from multimeters
auto pFlowMeasurements = branch->getMeasurementsFromDevices(
    telemetry, MeasurementType::P_FLOW
);
auto qFlowMeasurements = branch->getMeasurementsFromDevices(
    telemetry, MeasurementType::Q_FLOW
);
```

### Complete Example: Bus Querying Its Devices

```cpp
#include <sle/model/NetworkModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementDevice.h>

// Setup network and telemetry
NetworkModel network;
TelemetryData telemetry;

// Add bus
Bus* bus5 = network.addBus(5, "Bus 5");

// Add voltmeter to bus 5
auto voltmeter = std::make_unique<Voltmeter>("VM-BUS5", 5, 1000.0);
telemetry.addDevice(std::move(voltmeter));

// Add voltage measurement
auto vMag = std::make_unique<MeasurementModel>(
    MeasurementType::V_MAGNITUDE, 1.05, 0.01
);
vMag->setLocation(5);
// Pass device ID to addMeasurement for automatic linking
telemetry.addMeasurement(std::move(vMag), "VM-BUS5");

// Query telemetry from bus
const Bus* bus = network.getBus(5);
auto devices = bus->getAssociatedDevices(telemetry);
std::cout << "Bus 5 has " << devices.size() << " devices" << std::endl;

auto measurements = bus->getMeasurementsFromDevices(telemetry);
std::cout << "Bus 5 has " << measurements.size() << " measurements" << std::endl;

// Get voltage measurement specifically
auto vMeasurements = bus->getMeasurementsFromDevices(
    telemetry, MeasurementType::V_MAGNITUDE
);
if (!vMeasurements.empty()) {
    std::cout << "Voltage: " << vMeasurements[0]->getValue() << " p.u." << std::endl;
}
```

### Real-Time Updates

When telemetry is updated via `TelemetryData`, buses and branches automatically see the updated values:

```cpp
#include <sle/model/TelemetryData.h>

// Update a measurement
sle::model::TelemetryUpdate update;
update.deviceId = "VM-001";
update.type = sle::MeasurementType::V_MAGNITUDE;  // Required: device may have multiple measurements
update.value = 1.05;  // New voltage value
update.stdDev = 0.01;
update.busId = 5;
update.timestamp = getCurrentTimestamp();

// Process the update
telemetry->setNetworkModel(&network);
telemetry->updateMeasurement(update);

// Bus immediately sees the updated value
const Bus* bus = network.getBus(5);
Real currentVoltage = bus->getCurrentVoltageMeasurement(telemetry);
// currentVoltage is now 1.05 (the updated value)
```

**Important**: All query methods (`getMeasurementsFromDevices()`, `getCurrentVoltageMeasurement()`, etc.) query telemetry each time they're called, so they **always return the latest values**. There's no caching - updates are immediately visible.

### Convenience Methods for Current Values

Buses and branches provide convenience methods to get current measurement values directly:

```cpp
// Bus: Get current voltage measurement
Real voltage = bus->getCurrentVoltageMeasurement(telemetry);
if (!std::isnan(voltage)) {
    std::cout << "Current voltage: " << voltage << " p.u." << std::endl;
}

// Bus: Get current power injections
Real pInj, qInj;
if (bus->getCurrentPowerInjections(telemetry, pInj, qInj)) {
    std::cout << "P injection: " << pInj << " p.u." << std::endl;
    std::cout << "Q injection: " << qInj << " p.u." << std::endl;
}

// Branch: Get current power flow
Real pFlow, qFlow;
if (branch->getCurrentPowerFlow(telemetry, pFlow, qFlow)) {
    std::cout << "P flow: " << pFlow << " p.u." << std::endl;
    std::cout << "Q flow: " << qFlow << " p.u." << std::endl;
}

// Branch: Get current current measurement
Real current = branch->getCurrentCurrentMeasurement(telemetry);
if (!std::isnan(current)) {
    std::cout << "Current: " << current << " p.u." << std::endl;
}
```

### Benefits of Device-Based Queries

1. **Clear Ownership**: Buses/branches know which devices provide their measurements
2. **Device Metadata**: Access device characteristics (CT/PT ratios, accuracy)
3. **Device Status**: Check if devices are operational before using measurements
4. **Multiple Devices**: Handle cases where multiple devices measure the same quantity
5. **Better Bad Data Detection**: Use device characteristics for validation
6. **Real-Time Updates**: All queries return the latest values - updates are immediately visible
7. **Convenience Methods**: Direct access to current measurement values without manual filtering

