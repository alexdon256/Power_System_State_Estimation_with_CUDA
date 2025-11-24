# I/O Formats

This document describes the file formats supported for loading network models, measurements, and devices.

## Measurement Files

### CSV Format

**File:** `measurements.csv`

**Format:**
```
type,deviceId,busId,fromBus,toBus,value,stdDev
V_MAGNITUDE,VM-001,5,,,1.05,0.01
P_FLOW,MM-001,0,2,3,50.0,0.5
```

**Fields:**
- `type`: Measurement type (`V_MAGNITUDE`, `P_FLOW`, `Q_FLOW`, `P_INJECTION`, `Q_INJECTION`, `I_MAGNITUDE`, `BREAKER_STATUS`)
- `deviceId`: Device identifier (links to measurement device, must match `devices.csv`)
- `busId`: Bus ID (for bus measurements) or `0` placeholder (for branch measurements)
- `fromBus`: From bus ID (for branch measurements only, leave empty for bus measurements)
- `toBus`: To bus ID (for branch measurements only, leave empty for bus measurements)
- `value`: Measurement value
- `stdDev`: Standard deviation

**Branch Measurement Format:**
```
P_FLOW,MM-001,0,2,3,50.0,0.5
I_MAGNITUDE,CT-001,0,1,2,0.25,0.02
BREAKER_STATUS,CB-001,0,1,2,1.0,0.001
```
Format: `type,deviceId,0,fromBus,toBus,value,stdDev`
- `busId`: Must be `0` (placeholder for branch measurements)
- `fromBus`: From bus ID (required, must be numeric)
- `toBus`: To bus ID (required, must be numeric)
- **Detection**: Loader detects branch measurements when `fromBus` field is numeric

**Breaker Status Values:**
- `1.0` = Closed (breaker allows power flow)
- `0.0` = Open (breaker blocks power flow)
- Values > 0.5 are interpreted as closed, ≤ 0.5 as open

**Bus Measurement Format:**
```
V_MAGNITUDE,VM-001,5,,,1.05,0.01
P_INJECTION,METER-001,2,,,40.0,0.1
```
Format: `type,deviceId,busId,,,value,stdDev`
- `busId`: Bus ID where measurement is located (required, must be numeric)
- `fromBus`: Leave empty (empty field between commas)
- `toBus`: Leave empty (empty field between commas)
- **Detection**: Loader detects bus measurements when `fromBus` field is empty or non-numeric

## Device Files

### CSV Format

**File:** `devices.csv`

**Format:**
```
deviceType,deviceId,name,location,ctRatio,ptRatio,accuracy
MULTIMETER,MM-001,Branch 1 Multimeter,2:3,100.0,1000.0,0.01
VOLTMETER,VM-005,Bus 5 Voltmeter,5,,1000.0,0.005
```

**Fields:**
- `deviceType`: `MULTIMETER` or `MM` for multimeters, `VOLTMETER` or `VM` for voltmeters
- `deviceId`: Unique device identifier (must match measurement `deviceId`)
- `name`: Human-readable device name
- `location`: 
  - For multimeters: `fromBus:toBus` (e.g., `2:3`) or branchId
  - For voltmeters: busId
- `ctRatio`: Current Transformer ratio (multimeters only, default: 1.0)
- `ptRatio`: Potential Transformer ratio (default: 1.0)
- `accuracy`: Device accuracy as standard deviation multiplier (default: 0.01)

**Example:**
```csv
deviceType,deviceId,name,location,ctRatio,ptRatio,accuracy
MULTIMETER,MM-001,Branch 1-2 Multimeter,1:2,100.0,1000.0,0.01
MULTIMETER,MM-002,Branch 2-3 Multimeter,2:3,100.0,1000.0,0.01
VOLTMETER,VM-001,Bus 1 Voltmeter,1,,1000.0,0.005
VOLTMETER,VM-002,Bus 2 Voltmeter,2,,1000.0,0.005
```

## Network Model Files

### IEEE Common Format

**File:** `network.dat` or `network.raw`

See [MODEL_FORMAT.md](MODEL_FORMAT.md) for detailed IEEE format specification.

### JSON Format

**File:** `network.json`

```json
{
  "buses": [
    {
      "id": 1,
      "name": "Bus 1",
      "type": "SLACK",
      "baseKV": 230.0,
      "voltage": {"magnitude": 1.0, "angle": 0.0},
      "load": {"p": 0.0, "q": 0.0},
      "generation": {"p": 0.0, "q": 0.0}
    }
  ],
  "branches": [
    {
      "id": 1,
      "fromBus": 1,
      "toBus": 2,
      "impedance": {"r": 0.01, "x": 0.05},
      "charging": 0.001,
      "rating": 100.0
    }
  ],
  "baseMVA": 100.0
}
```

## Usage Examples

### Loading Complete Setup

```cpp
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>

// Load network
auto network = sle::interface::ModelLoader::loadFromIEEE("network.dat");

// Load measurements
auto telemetry = sle::interface::MeasurementLoader::loadTelemetry(
    "measurements.csv", *network
);

// Load devices
sle::interface::MeasurementLoader::loadDevices(
    "devices.csv", *telemetry, *network
);
```

### Command Line Usage

```bash
# Basic usage
./SLE_main network.dat measurements.csv

# With devices
./SLE_main network.dat measurements.csv devices.csv
```

