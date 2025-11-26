# API Reference

Complete reference for setters and getters.

## StateEstimator

**Estimation:**
- `estimate()` → `StateEstimationResult` - Full estimation
- `estimateIncremental()` → `StateEstimationResult` - Incremental estimation
- `detectBadData()` → `BadDataResult` - Bad data detection

**Configuration:**
- `setNetwork(network)` - Set network model
- `setTelemetryData(telemetry)` - Set telemetry data
- `setSolverConfig(config)` - Configure solver
- `configureForRealTime(tolerance, maxIter)` - Quick real-time config
- `configureForOffline(tolerance, maxIter)` - Quick offline config

**Status:**
- `isTopologyChanged()` → `bool` - Topology change detected
- `isModelUpdated()` → `bool` - Network updated
- `isTelemetryUpdated()` → `bool` - Measurements updated
- `markModelUpdated()` - Mark network updated
- `markTelemetryUpdated()` - Mark measurements updated

**Access:**
- `getNetwork()` → `NetworkModel*` - Get network
- `getTelemetryData()` → `TelemetryData*` - Get telemetry
- `getCurrentState()` → `StateVector*` - Get state estimate
- `getVoltageMagnitude(busId)` → `Real` - Get voltage magnitude
- `getVoltageAngle(busId)` → `Real` - Get voltage angle

## NetworkModel

**Circuit Breakers:**
- `addCircuitBreaker(id, branchId, fromBus, toBus, name)` → `CircuitBreaker*`
- `getCircuitBreaker(id)` → `CircuitBreaker*`
- `getCircuitBreakerByBranch(branchId)` → `CircuitBreaker*`
- `getCircuitBreakers()` → `vector<CircuitBreaker*>`

**Buses:**
- `addBus(id, name)` → `Bus*`
- `getBus(id)` → `Bus*`
- `getBusByName(name)` → `Bus*` (O(1) lookup)
- `getBuses()` → `vector<Bus*>`
- `getBusCount()` → `size_t`

**Branches:**
- `addBranch(id, fromBus, toBus)` → `Branch*`
- `getBranch(id)` → `Branch*`
- `getBranchByBuses(fromBus, toBus)` → `Branch*` (O(1) lookup)
- `getBranches()` → `vector<Branch*>`
- `getBranchCount()` → `size_t`
- `updateBranch(id, branchData)` - Update branch properties

**Configuration:**
- `setBaseMVA(baseMVA)` - Set base MVA
- `setReferenceBus(busId)` - Set reference bus
- `getBaseMVA()` → `Real`
- `getReferenceBus()` → `BusId`

## Bus

**Properties:**
- `setType(type)` - Set bus type (PQ, PV, Slack)
- `setBaseKV(kv)` - Set base voltage
- `setVoltage(magnitude, angle)` - Set voltage
- `setLoad(pMW, qMVAR)` - Set load
- `setGeneration(pMW, qMVAR)` - Set generation
- `setShunt(g, b)` - Set shunt admittance
- `setVoltageLimits(min, max)` - Set voltage limits

**Getters:**
- `getType()` → `BusType`
- `getBaseKV()` → `Real`
- `getVPU()` → `Real` - Voltage in p.u.
- `getVKV()` → `Real` - Voltage in kV
- `getAngle()` → `Real` - Angle in radians
- `getPMW()` → `Real` - Active power injection (MW)
- `getQMVAR()` → `Real` - Reactive power injection (MVAR)
- `getAssociatedDevices()` → `vector<MeasurementDevice*>` - O(1) access

## Branch

**Properties:**
- `setImpedance(r, x)` - Set impedance (p.u.)
- `setCharging(b)` - Set charging susceptance (p.u.)
- `setRating(mva)` - Set MVA rating
- `setTapRatio(ratio)` - Set tap ratio
- `setPhaseShift(radians)` - Set phase shift
- `setStatus(closed)` - Set status (true=closed, false=open)

**Getters:**
- `getR()` → `Real` - Resistance (p.u.)
- `getX()` → `Real` - Reactance (p.u.)
- `getB()` → `Real` - Charging susceptance (p.u.)
- `getRating()` → `Real` - MVA rating
- `getTapRatio()` → `Real` - Tap ratio
- `getPhaseShift()` → `Real` - Phase shift (radians)
- `getStatus()` → `bool` - Status (true=closed)
- `getPMW()` → `Real` - Active power flow (MW)
- `getQMVAR()` → `Real` - Reactive power flow (MVAR)
- `getAssociatedDevices()` → `vector<MeasurementDevice*>` - O(1) access

## TelemetryData

**Measurement Updates:**
- `updateMeasurement(update)` - Update single measurement
- `updateMeasurements(updates)` - Batch update
- `addMeasurement(update)` - Add new measurement
- `removeMeasurement(deviceId, type)` - Remove measurement

**Queries:**
- `getMeasurementCount()` → `size_t`
- `getMeasurements()` → Iterator over measurements
- `getDevicesByBus(busId)` → `vector<MeasurementDevice*>` (O(1))
- `getDevicesByBranch(fromBus, toBus)` → `vector<MeasurementDevice*>` (O(1))
- `getMeasurementVector()` → `vector<Real>` - Measurement values
- `getWeightMatrix()` → `SparseMatrix` - Weight matrix

**Configuration:**
- `setNetworkModel(network)` - Link to network for topology queries

## MeasurementModel

**Properties:**
- `setValue(value)` - Set measurement value
- `setStdDev(stdDev)` - Set standard deviation
- `setLocation(busId)` - Set bus location
- `setBranch(fromBus, toBus)` - Set branch location

**Getters:**
- `getType()` → `MeasurementType`
- `getValue()` → `Real`
- `getStdDev()` → `Real`
- `getWeight()` → `Real` - Computed weight (1/stdDev²)
- `getLocation()` → `BusId`
- `getGlobalIndex()` → `Index` - Global measurement index

## CircuitBreaker

**Properties:**
- `setStatus(closed)` - Set status (true=closed, false=open)

**Getters:**
- `getId()` → `string`
- `getBranchId()` → `BranchId`
- `getFromBus()` → `BusId`
- `getToBus()` → `BusId`
- `getStatus()` → `bool` - Status (true=closed)
- `isClosed()` → `bool`
- `isOpen()` → `bool`

## StateVector

**Access:**
- `getVoltageMagnitude(index)` → `Real`
- `getVoltageAngle(index)` → `Real`
- `setVoltageMagnitude(index, value)`
- `setVoltageAngle(index, value)`
- `size()` → `size_t`

**Initialization:**
- `initializeFromNetwork(network)` - Initialize from network
