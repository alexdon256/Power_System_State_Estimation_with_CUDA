# Model Format Specifications

## IEEE Common Format

The IEEE Common Format is the standard format for power system data used in power flow and state estimation studies.

### Bus Data Format

```
BUS DATA FOLLOWS
  1   'BUS1'  138.00  1  0.0  0.0  0.0  0.0  0.0  0.0  1.000  0.000
  2   'BUS2'  138.00  1  0.0  0.0  0.0  0.0  0.0  0.0  1.000  0.000
```

**Format:** `BusId Name BaseKV Type PLoad QLoad PGen QGen GShunt BShunt VMag VAngle`

#### Component Descriptions:

- **BusId** (integer): Unique bus identifier. Must be positive and unique within the network. Used as the primary key for bus references in branches and measurements.

- **Name** (string, optional): Human-readable bus name or label. Typically enclosed in single quotes. Used for display and reporting purposes. Can be empty or omitted.

- **BaseKV** (real): Base voltage in kilovolts (kV) for this bus. Used for per-unit calculations. Common values: 138, 230, 345, 500, 765 kV. All per-unit values are calculated relative to this base voltage and the system base MVA.

- **Type** (integer): Bus type classification:
  - `1` = **PQ Bus** (Load Bus): Active and reactive power are specified. Voltage magnitude and angle are unknown and will be calculated. Most common bus type for load buses.
  - `2` = **PV Bus** (Generator Bus): Active power and voltage magnitude are specified. Reactive power and voltage angle are unknown. Used for generator buses with voltage control.
  - `3` = **Slack Bus** (Reference Bus): Voltage magnitude and angle are specified. Active and reactive power are unknown and will be calculated. One slack bus is required per network to provide the reference angle (typically 0.0 radians). The slack bus balances power in the system.
  - `4` = **Isolated Bus**: Bus with no connections. Rarely used in practice.

- **PLoad** (real, p.u.): Active power load in per-unit. Positive values indicate power consumption (load). Negative values indicate power injection. Default: 0.0 p.u.

- **QLoad** (real, p.u.): Reactive power load in per-unit. Positive values indicate inductive load (lagging). Negative values indicate capacitive load (leading). Default: 0.0 p.u.

- **PGen** (real, p.u.): Active power generation in per-unit. Positive values indicate power generation. Typically specified for PV and Slack buses. Default: 0.0 p.u.

- **QGen** (real, p.u.): Reactive power generation in per-unit. Positive values indicate reactive power injection. Typically calculated for PV buses. Default: 0.0 p.u.

- **GShunt** (real, p.u.): Shunt conductance in per-unit. Represents resistive shunt elements (e.g., line charging, capacitor banks). Positive values indicate power consumption. Default: 0.0 p.u.

- **BShunt** (real, p.u.): Shunt susceptance in per-unit. Represents reactive shunt elements:
  - Positive values: Capacitive (e.g., capacitor banks, line charging)
  - Negative values: Inductive (e.g., reactors)
  - Default: 0.0 p.u.

- **VMag** (real, p.u.): Initial voltage magnitude in per-unit. Typical range: 0.9 to 1.1 p.u. Used as initial guess for state estimation. For PV and Slack buses, this is the specified voltage setpoint. Default: 1.0 p.u.

- **VAngle** (real, radians): Initial voltage phase angle in radians. For the Slack bus, this is typically 0.0 (reference angle). For other buses, this is an initial guess. Default: 0.0 radians.

### Branch Data Format

```
BRANCH DATA FOLLOWS
  1   2   0.00281  0.0281  0.00712  400.0  1.0  0.0  1
```

**Format:** `FromBus ToBus R X B RateA Ratio Angle Status`

#### Component Descriptions:

- **FromBus** (integer): Source bus identifier. Must match an existing bus ID in the network. Defines the "from" end of the branch for power flow direction.

- **ToBus** (integer): Destination bus identifier. Must match an existing bus ID in the network. Defines the "to" end of the branch. Note: Branches are directional; power flow from FromBus to ToBus.

- **R** (real, p.u.): Series resistance in per-unit. Represents the resistive losses in the transmission line or transformer winding. Always positive. Typical range: 0.0001 to 0.1 p.u. for transmission lines. Higher values indicate longer lines or higher resistance.

- **X** (real, p.u.): Series reactance in per-unit. Represents the inductive reactance of the transmission line or transformer. Always positive. Typical range: 0.001 to 1.0 p.u. The X/R ratio determines the power flow characteristics. Higher X/R ratios are typical for transmission lines.

- **B** (real, p.u.): Total charging susceptance in per-unit. Represents the capacitive charging of the transmission line (line charging). Always positive for transmission lines. Typical range: 0.001 to 0.1 p.u. For transformers, this is typically 0.0. Higher values indicate longer lines with significant capacitance.

- **RateA** (real, MVA): Normal continuous rating in MVA. The maximum apparent power that the branch can carry continuously under normal operating conditions. Used for thermal limit checking. Typical values: 100-2000 MVA for transmission lines, depending on voltage level and conductor type.

- **Ratio** (real, dimensionless): **Transformer tap ratio** (also called turns ratio). 
  - For **transmission lines**: Always 1.0 (no transformation)
  - For **transformers**: The ratio of primary to secondary voltage (e.g., 1.05 means 5% boost, 0.95 means 5% buck)
  - Typical range: 0.9 to 1.1 for tap-changing transformers
  - Default: 1.0 (line or fixed-ratio transformer)
  - **Note**: The tap ratio is applied at the "from" bus side. A ratio > 1.0 increases voltage at the "to" bus relative to the "from" bus.

- **Angle** (real, radians): **Phase shift angle** in radians. 
  - For **transmission lines**: Always 0.0 (no phase shift)
  - For **phase-shifting transformers**: The phase shift angle introduced by the transformer
  - Typical range: -0.2 to +0.2 radians (-11.5° to +11.5°)
  - Positive values: Phase leads at "to" bus relative to "from" bus
  - Default: 0.0 (no phase shift)
  - **Note**: Used to model phase-shifting transformers that control power flow direction.

- **Status** (integer): Branch status flag:
  - `0` = **Out of Service**: Branch is disconnected/open. Not included in network calculations.
  - `1` = **In Service**: Branch is connected/closed. Included in all network calculations.
  - Default: 1 (in service)

**Transformer Identification:**
A branch is considered a transformer if:
- `|Ratio - 1.0| > 1e-6` (tap ratio differs from 1.0), OR
- `|Angle| > 1e-6` (non-zero phase shift)

Otherwise, it is treated as a transmission line.

## JSON Format

The JSON format provides a structured, human-readable representation of the power system network model.

```json
{
  "network": {
    "baseMVA": 100.0,
    "buses": [
      {
        "id": 1,
        "name": "BUS1",
        "baseKV": 138.0,
        "type": "PV",
        "load": {"p": 0.0, "q": 0.0},
        "generation": {"p": 0.0, "q": 0.0},
        "shunt": {"g": 0.0, "b": 0.0},
        "voltage": {"magnitude": 1.0, "angle": 0.0},
        "limits": {"vMin": 0.95, "vMax": 1.05}
      }
    ],
    "branches": [
      {
        "id": 1,
        "fromBus": 1,
        "toBus": 2,
        "impedance": {"r": 0.00281, "x": 0.0281},
        "charging": 0.00712,
        "rating": 400.0,
        "tapRatio": 1.0,
        "phaseShift": 0.0,
        "status": 1
      }
    ]
  }
}
```

#### JSON Component Descriptions:

**Network Level:**
- **baseMVA** (real): System base MVA for per-unit calculations. Common values: 100, 1000 MVA. All power values in the network are normalized to this base.

**Bus Object:**
- **id** (integer): Bus identifier (same as IEEE format BusId)
- **name** (string): Bus name (same as IEEE format Name)
- **baseKV** (real): Base voltage in kV (same as IEEE format BaseKV)
- **type** (string): Bus type as string: `"PQ"`, `"PV"`, `"Slack"`, or `"Isolated"` (same as IEEE format Type)
- **load** (object): Load specification:
  - **p** (real, p.u.): Active power load (same as IEEE format PLoad)
  - **q** (real, p.u.): Reactive power load (same as IEEE format QLoad)
- **generation** (object): Generation specification:
  - **p** (real, p.u.): Active power generation (same as IEEE format PGen)
  - **q** (real, p.u.): Reactive power generation (same as IEEE format QGen)
- **shunt** (object): Shunt admittance:
  - **g** (real, p.u.): Shunt conductance (same as IEEE format GShunt)
  - **b** (real, p.u.): Shunt susceptance (same as IEEE format BShunt)
- **voltage** (object): Initial voltage state:
  - **magnitude** (real, p.u.): Voltage magnitude (same as IEEE format VMag)
  - **angle** (real, radians): Voltage angle (same as IEEE format VAngle)
- **limits** (object, optional): Voltage limits:
  - **vMin** (real, p.u.): Minimum allowed voltage (typically 0.95 p.u.)
  - **vMax** (real, p.u.): Maximum allowed voltage (typically 1.05 p.u.)

**Branch Object:**
- **id** (integer): Branch identifier (optional, for reference)
- **fromBus** (integer): Source bus ID (same as IEEE format FromBus)
- **toBus** (integer): Destination bus ID (same as IEEE format ToBus)
- **impedance** (object): Series impedance:
  - **r** (real, p.u.): Series resistance (same as IEEE format R)
  - **x** (real, p.u.): Series reactance (same as IEEE format X)
- **charging** (real, p.u.): Charging susceptance (same as IEEE format B)
- **rating** (real, MVA): Normal continuous rating (same as IEEE format RateA)
- **tapRatio** (real, dimensionless): **Transformer tap ratio** (same as IEEE format Ratio). Default: 1.0. Values ≠ 1.0 indicate a transformer.
- **phaseShift** (real, radians): Phase shift angle (same as IEEE format Angle). Default: 0.0. Non-zero values indicate a phase-shifting transformer.
- **status** (integer): Branch status (same as IEEE format Status). 0 = out of service, 1 = in service. Default: 1.

## Measurement Format (CSV)

The CSV measurement format is used for loading telemetry data for state estimation.

```csv
Type,DeviceId,BusId,Value,StdDev
P_INJECTION,METER_001,1,1.5,0.01
Q_INJECTION,METER_001,1,0.8,0.01
V_MAGNITUDE,VT_001,1,1.0,0.005
P_FLOW,FLOW_001,1,2,0.5,0.01
Q_FLOW,FLOW_002,2,3,0.3,0.01
```

**Format:** `Type,DeviceId,BusId,Value,StdDev` (for bus measurements)
**Format:** `Type,DeviceId,FromBus,ToBus,Value,StdDev` (for branch/flow measurements)

#### CSV Component Descriptions:

- **Type** (string): Measurement type identifier:
  - `P_INJECTION`: Active power injection at a bus (MW, p.u.)
  - `Q_INJECTION`: Reactive power injection at a bus (MVAR, p.u.)
  - `V_MAGNITUDE`: Voltage magnitude at a bus (p.u. or kV)
  - `I_MAGNITUDE`: Current magnitude (p.u. or A)
  - `P_FLOW`: Active power flow on a branch (MW, p.u.)
  - `Q_FLOW`: Reactive power flow on a branch (MVAR, p.u.)
  - `V_PHASOR`: Voltage phasor from PMU (magnitude and angle)
  - `I_PHASOR`: Current phasor from PMU (magnitude and angle)
  - `VIRTUAL`: Virtual measurement (zero injection constraint)
  - `PSEUDO`: Pseudo measurement (load forecast)

- **DeviceId** (string): Unique identifier for the measuring device. Used to track device status, accuracy, and for bad data detection. Examples: "METER_001", "VT_001", "PMU_001".

- **BusId** (integer): Bus identifier where the measurement is taken. For bus measurements (injection, voltage), this is the bus ID. For branch measurements (flow), this field may be omitted or used for the "from" bus.

- **FromBus** (integer, optional): Source bus for branch flow measurements. Required for P_FLOW and Q_FLOW measurements.

- **ToBus** (integer, optional): Destination bus for branch flow measurements. Required for P_FLOW and Q_FLOW measurements.

- **Value** (real): Measured value in the appropriate units:
  - Power: MW or p.u. (active), MVAR or p.u. (reactive)
  - Voltage: p.u. or kV
  - Current: p.u. or Amperes
  - Angle: radians

- **StdDev** (real): Standard deviation of measurement error. Used to calculate measurement weights in weighted least squares (WLS) estimation. Weight = 1 / (StdDev²). Smaller StdDev indicates higher accuracy and higher weight. Typical values:
  - Voltage magnitude: 0.001-0.01 p.u.
  - Power injection: 0.01-0.05 p.u.
  - Power flow: 0.01-0.05 p.u.
  - PMU phasors: 0.0001-0.001 p.u.

## SCADA Format

The SCADA format represents real-time telemetry data from Supervisory Control and Data Acquisition systems.

```csv
DeviceId,Type,BusId,Value,Timestamp
METER_001,P,1,1.5,1234567890
METER_001,Q,1,0.8,1234567890
VT_001,V,1,1.0,1234567890
```

**Format:** `DeviceId,Type,BusId,Value,Timestamp`

#### SCADA Component Descriptions:

- **DeviceId** (string): SCADA device identifier (same as CSV format DeviceId)

- **Type** (string): Abbreviated measurement type:
  - `P`: Active power (same as P_INJECTION or P_FLOW)
  - `Q`: Reactive power (same as Q_INJECTION or Q_FLOW)
  - `V`: Voltage magnitude (same as V_MAGNITUDE)
  - `I`: Current magnitude (same as I_MAGNITUDE)

- **BusId** (integer): Bus identifier (same as CSV format BusId)

- **Value** (real): Measured value in engineering units (same as CSV format Value)

- **Timestamp** (integer): Unix timestamp in milliseconds or seconds. Used for temporal ordering and real-time processing. Enables detection of stale measurements and time-synchronized state estimation.

