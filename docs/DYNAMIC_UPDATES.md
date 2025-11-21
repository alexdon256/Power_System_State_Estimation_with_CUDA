# Dynamic Updates - Adding/Removing Components On The Fly

## Overview

The codebase supports **dynamic updates** for network components and measurements, allowing real-time modifications without full system reload. All Bus and Branch properties can be updated on the fly using their comprehensive setter methods. However, some functionality (measurement removal/update) is incomplete.

## ✅ Currently Supported

### Network Model Updates

#### Adding Components
```cpp
// Add a new bus
Bus* newBus = network->addBus(100, "New Bus");
newBus->setBaseKV(230.0);
newBus->setType(BusType::PQ);
newBus->setLoad(50.0, 20.0);  // MW, MVAR

// Add a new branch
Branch* newBranch = network->addBranch(200, 1, 100);  // from bus 1 to bus 100
newBranch->setImpedance(0.01, 0.05);  // R, X in p.u.
newBranch->setRating(100.0);  // MVA rating
```

#### Removing Components
```cpp
// Remove a bus (automatically invalidates caches)
network->removeBus(100);

// Remove a branch (automatically invalidates caches)
network->removeBranch(200);
```

#### Updating Components - Complete Setter Examples

**Bus Updates - All Available Setters:**
```cpp
Bus* bus = network->getBus(1);
if (bus) {
    // Bus type (Slack, PV, PQ)
    bus->setType(BusType::PQ);
    
    // Base voltage (kV)
    bus->setBaseKV(230.0);
    
    // Voltage magnitude and angle (initial guess)
    bus->setVoltage(1.05, 0.0);  // magnitude (p.u.), angle (radians)
    
    // Load (MW, MVAR)
    bus->setLoad(50.0, 20.0);  // P_load (MW), Q_load (MVAR)
    
    // Generation (MW, MVAR)
    bus->setGeneration(100.0, 30.0);  // P_gen (MW), Q_gen (MVAR)
    
    // Shunt admittance (conductance, susceptance in p.u.)
    bus->setShunt(0.01, 0.05);  // G_shunt (p.u.), B_shunt (p.u.)
    
    // Voltage limits (p.u.)
    bus->setVoltageLimits(0.95, 1.05);  // V_min (p.u.), V_max (p.u.)
    
    // All changes automatically invalidate caches
}
```

**Branch Updates - All Available Setters:**
```cpp
Branch* branch = network->getBranch(1);
if (branch) {
    // Series impedance (resistance, reactance in p.u.)
    branch->setImpedance(0.01, 0.05);  // R (p.u.), X (p.u.)
    
    // Charging susceptance (p.u.) - for transmission line capacitance
    branch->setCharging(0.001);  // B (p.u.)
    
    // MVA rating (thermal limit)
    branch->setRating(100.0);  // MVA rating
    
    // Transformer tap ratio (1.0 for lines, variable for transformers)
    branch->setTapRatio(1.05);  // Tap ratio (e.g., 1.05 = 5% boost)
    
    // Phase shift angle (radians) - for phase-shifting transformers
    branch->setPhaseShift(0.0);  // Phase shift (radians)
    
    // All changes automatically invalidate caches
}
```

**Note:** `updateBus()` and `updateBranch()` methods are now fully functional with copy assignment operators implemented. You can also use direct modification via setters.

#### Searching Buses by Name
```cpp
// Search for a bus by name (O(1) average-case lookup)
Bus* bus = network->getBusByName("Main Substation");
if (bus) {
    std::cout << "Found bus ID: " << bus->getId() << std::endl;
    bus->setVoltage(1.05, 0.0);
}

// Const version for read-only access
const Bus* constBus = network->getBusByName("Generator Bus");
if (constBus) {
    std::cout << "Voltage: " << constBus->getVoltageMagnitude() << std::endl;
}
```

### Measurement Updates

#### Adding Measurements
```cpp
// Via TelemetryData directly
auto measurement = std::make_unique<MeasurementModel>(
    MeasurementType::V_MAGNITUDE, 1.05, 0.01, "PMU_001");
measurement->setLocation(1);  // Bus ID
telemetry->addMeasurement(std::move(measurement));

// Via TelemetryProcessor (thread-safe, queued)
TelemetryUpdate update;
update.deviceId = "PMU_001";
update.type = MeasurementType::V_MAGNITUDE;
update.value = 1.05;
update.stdDev = 0.01;
update.busId = 1;
update.timestamp = getCurrentTimestamp();

processor.addMeasurement(update);
// Or batch updates
std::vector<TelemetryUpdate> updates = {...};
processor.updateMeasurements(updates);
```

#### Real-Time Processing
```cpp
// Start background processing thread
processor.startRealTimeProcessing();

// Updates are queued and processed asynchronously
processor.updateMeasurement(update);

// Check for pending updates
if (processor.hasPendingUpdates()) {
    processor.processUpdateQueue();
}

// Stop processing
processor.stopRealTimeProcessing();
```

## ✅ All Functionality Implemented

All previously missing functionality has been implemented:

### 1. ✅ Remove Measurements
**Status:** Fully implemented

```cpp
// TelemetryProcessor (thread-safe, O(1) average-case lookup)
processor.removeMeasurement("PMU_001");

// TelemetryData (direct access, O(1) average-case lookup)
bool removed = telemetry->removeMeasurement("PMU_001");
```

**Implementation:**
- ✅ `TelemetryData::removeMeasurement()` method implemented
- ✅ Device ID lookup index (`std::unordered_map<std::string, size_t>`) for O(1) removal
- ✅ `TelemetryProcessor::removeMeasurement()` fully functional
- ✅ `BadDataDetector::removeBadMeasurements()` fully implemented (removes by device ID, handles indices)

### 2. ✅ Update Existing Measurements
**Status:** Fully implemented

`TelemetryProcessor::updateMeasurement()` and `TelemetryData::updateMeasurement()` now properly update existing measurements by device ID, or create new ones if not found.

**Implementation:**
- ✅ Device ID lookup in `TelemetryData` (`std::unordered_map<std::string, size_t>`)
- ✅ `TelemetryData::updateMeasurement()` method
- ✅ Logic in `TelemetryProcessor::applyUpdate()` to find and update existing measurements

### 3. ✅ Copy Assignment for Bus/Branch
**Status:** Fully implemented

The `updateBus()` and `updateBranch()` methods now work properly with copy assignment operators:

```cpp
// This now works:
Bus newBusData = ...;
network->updateBus(1, newBusData);  // ✅ Works with Bus::operator=()

// Or use direct modification:
Bus* bus = network->getBus(1);
*bus = newBusData;  // ✅ Also works with Bus::operator=()
```

**Note:** Both approaches now work. The copy assignment operators copy all properties except the identifying fields (id/name for Bus, id/fromBus/toBus for Branch).

## 🔄 Cache Invalidation

All dynamic updates automatically invalidate caches:

- **Network changes** (`addBus`, `removeBus`, `addBranch`, `removeBranch`):
  - Invalidates adjacency lists
  - Invalidates GPU device data
  - Invalidates cached power injection/flow vectors
  - Next computation will rebuild everything

- **Measurement changes**:
  - Jacobian matrix structure needs rebuilding
  - Measurement vector needs updating
  - State estimator handles this automatically

## 📝 Example: Complete Dynamic Update Workflow

```cpp
#include <sle/interface/StateEstimator.h>
#include <sle/interface/TelemetryProcessor.h>

// Initialize
StateEstimator estimator;
estimator.setNetwork(network);
estimator.setTelemetryData(telemetry);

// Real-time loop
while (running) {
    // 1. Add new bus dynamically with all properties
    Bus* newBus = network->addBus(999, "Dynamic Bus");
    newBus->setType(BusType::PQ);
    newBus->setBaseKV(230.0);
    newBus->setVoltage(1.0, 0.0);  // Initial guess: 1.0 p.u., 0 radians
    newBus->setLoad(10.0, 5.0);    // 10 MW, 5 MVAR load
    newBus->setGeneration(0.0, 0.0);  // No generation
    newBus->setShunt(0.0, 0.0);    // No shunt admittance
    newBus->setVoltageLimits(0.95, 1.05);  // Voltage limits
    
    // 2. Add branch to new bus with all properties
    Branch* newBranch = network->addBranch(888, 1, 999);
    newBranch->setImpedance(0.02, 0.1);  // R=0.02, X=0.1 p.u.
    newBranch->setCharging(0.001);        // Charging susceptance
    newBranch->setRating(50.0);           // 50 MVA rating
    newBranch->setTapRatio(1.0);          // No transformer tap
    newBranch->setPhaseShift(0.0);         // No phase shift
    
    // 3. Update existing bus - comprehensive example
    Bus* bus = network->getBus(1);
    if (bus) {
        // Change bus type
        bus->setType(BusType::PV);  // Change from PQ to PV
        
        // Update load
        bus->setLoad(60.0, 25.0);  // New load: 60 MW, 25 MVAR
        
        // Update generation
        bus->setGeneration(120.0, 40.0);  // New generation: 120 MW, 40 MVAR
        
        // Update shunt admittance (e.g., capacitor bank added)
        bus->setShunt(0.0, 0.02);  // Add 0.02 p.u. capacitive susceptance
        
        // Update voltage limits
        bus->setVoltageLimits(0.90, 1.10);  // Wider limits
        
        // Update base voltage (if transformer tap changed)
        bus->setBaseKV(240.0);  // New base voltage
    }
    
    // 4. Update existing branch - comprehensive example
    Branch* branch = network->getBranch(1);
    if (branch) {
        // Update impedance (e.g., line reconductoring)
        branch->setImpedance(0.015, 0.08);  // Lower resistance, lower reactance
        
        // Update charging (e.g., line length change)
        branch->setCharging(0.002);  // New charging susceptance
        
        // Update rating (e.g., thermal upgrade)
        branch->setRating(150.0);  // New MVA rating
        
        // Update transformer tap (if it's a transformer)
        branch->setTapRatio(1.03);  // 3% boost
        
        // Update phase shift (for phase-shifting transformer)
        branch->setPhaseShift(0.05);  // 0.05 radians phase shift
    }
    
    // 5. Add new measurement
    TelemetryUpdate update;
    update.deviceId = "PMU_999";
    update.type = MeasurementType::V_MAGNITUDE;
    update.value = 1.02;
    update.stdDev = 0.01;
    update.busId = 999;
    update.timestamp = getCurrentTimestamp();
    processor.addMeasurement(update);
    
    // 6. Process updates and re-estimate
    processor.processUpdateQueue();
    auto result = estimator.estimateIncremental();
    
    // 7. Remove component if needed
    network->removeBranch(888);
    network->removeBus(999);
}
```

## ✅ Completed Enhancements

1. **✅ Added `TelemetryData::removeMeasurement()`**
   - Implemented device ID lookup index (`std::unordered_map<std::string, size_t>`) for O(1) average-case lookup
   - Added `removeMeasurement(const std::string& deviceId)` method
   - Updated `TelemetryProcessor::removeMeasurement()` to use the new method

2. **✅ Added `TelemetryData::updateMeasurement()`**
   - Implemented `findMeasurementByDeviceId()` for O(1) lookup
   - Added `updateMeasurement()` to update value, stdDev, and timestamp
   - Updated `TelemetryProcessor::applyUpdate()` to check for existing measurements before creating new ones

3. **✅ Added copy assignment operators**
   - Implemented `Bus::operator=(const Bus&)` (copies all properties except id/name)
   - Implemented `Branch::operator=(const Branch&)` (copies all properties except id/fromBus/toBus)
   - `updateBus()` and `updateBranch()` now work properly

4. **✅ Added bus search by name**
   - Implemented `getBusByName(const std::string& name)` (const and non-const versions)
   - Added `busNameMap_` for O(1) name-based lookup
   - Name index is automatically maintained on add/remove/clear operations

## Summary

✅ **Fully Supported:**
- Add/remove buses and branches
- Add/remove/update measurements (via TelemetryData or TelemetryProcessor)
- Real-time asynchronous measurement processing
- Automatic cache invalidation
- **All Bus setters:** `setType()`, `setBaseKV()`, `setVoltage()`, `setLoad()`, `setGeneration()`, `setShunt()`, `setVoltageLimits()`
- **All Branch setters:** `setImpedance()`, `setCharging()`, `setRating()`, `setTapRatio()`, `setPhaseShift()`
- **Bad data removal:** `BadDataDetector::removeBadMeasurements()` fully implemented

## Dead Code: DeviceModel

**Status:** `DeviceModel` class is **unused** and can be safely removed.

**Analysis:**
- `DeviceModel` is included in `MeasurementModel.h` but never used
- `MeasurementModel` duplicates all `DeviceModel` functionality independently
- `DeviceModel` is included in `MeasurementLoader.h` but never used in implementation
- `DeviceType` enum (used only by `DeviceModel`) is also unused

**Recommendation:** Remove `DeviceModel` to reduce code complexity:
- Delete `include/sle/model/DeviceModel.h`
- Delete `src/model/DeviceModel.cpp`
- Remove `#include <sle/model/DeviceModel.h>` from `MeasurementModel.h` and `MeasurementLoader.h`
- Remove `src/model/DeviceModel.cpp` from `CMakeLists.txt`
- Consider removing `DeviceType` enum from `Types.h` if not needed for future use

**Why it exists:** Likely intended as a base class for `MeasurementModel`, but `MeasurementModel` was implemented independently with duplicate functionality.

