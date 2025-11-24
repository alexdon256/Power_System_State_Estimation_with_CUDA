# IEEE 14-Bus Test Case Example Files

This directory contains example input files for the IEEE 14-bus power system test case.

## Files

- **`network.dat`** - Network topology in IEEE Common Format
  - 14 buses
  - 20 branches (transmission lines and transformers)
  - Base MVA: 100 MVA
  - Base voltage: 230 kV

- **`measurements.csv`** - Measurement data in CSV format
  - Format: `type,deviceId,busId,fromBus,toBus,value,stdDev`
  - Bus measurements: `type,deviceId,busId,,,value,stdDev` (empty fromBus/toBus)
  - Branch measurements: `type,deviceId,0,fromBus,toBus,value,stdDev` (busId=0 placeholder)
  - Contains:
    - 3 voltage magnitude measurements (VT_001, VT_002, VT_003)
    - 4 power injection measurements (METER_001-004)
    - 8 power flow measurements (FLOW_001-008)
    - 2 current magnitude measurements (CT_001-002)
    - 5 breaker status measurements (CB_001-005): 4 closed (1.0), 1 open (0.0)
    - 3 breaker status measurements (CB_001-003)

- **`devices.csv`** - Measurement device definitions in CSV format
  - Format: `deviceType,deviceId,name,location,ctRatio,ptRatio,accuracy`
  - Voltmeters: `VOLTMETER,deviceId,name,busId,,ptRatio,accuracy`
  - Multimeters: `MULTIMETER,deviceId,name,fromBus:toBus,ctRatio,ptRatio,accuracy`
  - Links measurements to physical devices with CT/PT ratios and accuracy

## Usage

```bash
# Basic usage (without devices)
./compare_measured_estimated examples/ieee14/network.dat examples/ieee14/measurements.csv

# With device file
./compare_measured_estimated examples/ieee14/network.dat examples/ieee14/measurements.csv examples/ieee14/devices.csv
```

## Format Details

### Measurement CSV Format

**Bus Measurements:**
```csv
type,deviceId,busId,,,value,stdDev
V_MAGNITUDE,VT_001,1,,,1.060,0.005
P_INJECTION,METER_001,2,,,40.0,0.1
```

**Branch Measurements:**
```csv
type,deviceId,0,fromBus,toBus,value,stdDev
P_FLOW,FLOW_001,0,1,2,0.5,0.05
I_MAGNITUDE,CT_001,0,1,2,0.25,0.02
BREAKER_STATUS,CB_001,0,1,2,1.0,0.001
```

**Breaker Status Values:**
- `1.0` = Closed (breaker allows power flow)
- `0.0` = Open (breaker blocks power flow)

### Device CSV Format

**Voltmeters:**
```csv
deviceType,deviceId,name,location,ctRatio,ptRatio,accuracy
VOLTMETER,VT_001,Bus 1 Voltmeter,1,,1000.0,0.005
```

**Multimeters:**
```csv
deviceType,deviceId,name,location,ctRatio,ptRatio,accuracy
MULTIMETER,FLOW_001,Branch 1-2 P Flow Meter,1:2,100.0,1000.0,0.01
```

## Notes

- Device IDs in `devices.csv` must match `deviceId` values in `measurements.csv`
- Branch locations use format `fromBus:toBus` (e.g., `1:2`)
- Empty fields in CSV should be left empty (not `-1` or `0` unless specified)
- CT/PT ratios are transformer ratios (e.g., 100:1 CT = 100.0)
- Accuracy is standard deviation multiplier (e.g., 0.01 = 1% accuracy)

