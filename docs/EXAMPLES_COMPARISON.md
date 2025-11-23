# Examples Comparison: All Setup Types

This document compares all four setup examples to help you choose the right approach for your application.

## Quick Comparison Table

| Feature | Offline | Real-Time | Hybrid | Advanced |
|---------|---------|-----------|--------|----------|
| **Primary Focus** | High accuracy analysis | Production monitoring | Bad data handling | Advanced features demo |
| **Estimation Method** | Standard WLS | Standard WLS | WLS + Robust | WLS + Robust + Load Flow |
| **Accuracy** | High (1e-8 tolerance) | Medium (1e-5 tolerance) | Medium (1e-5 tolerance) | High (1e-8 tolerance) |
| **Real-Time Updates** | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Incremental Estimation** | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Robust Estimation** | ❌ No | ❌ No | ✅ Yes (periodic) | ✅ Yes |
| **Bad Data Detection** | ✅ Yes (once) | ✅ Yes (once) | ✅ Yes (periodic) | ❌ No |
| **System Monitoring** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Pre-Validation** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Observability Check** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **Report Generation** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Computed Values** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Load Flow** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Optimal Placement** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Transformer Config** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **PMU Support** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Multi-Area** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Performance** | Slower (high accuracy) | Fast (real-time) | Medium (robust overhead) | Slower (many features) |
| **Use Case** | Planning/analysis | Production SCADA/EMS | Systems with bad data | Feature demonstration |

## Detailed Feature Comparison

### 1. Estimation Strategy

#### Offline Setup
- **Method**: Standard WLS only
- **Tolerance**: 1e-8 (high accuracy)
- **Max Iterations**: 50
- **Mode**: Single estimation run
- **Focus**: Maximum accuracy for analysis

```cpp
estimator.configureForOffline(1e-8, 50, true);
auto result = estimator.estimate();
```

#### Real-Time Setup
- **Method**: Standard WLS only
- **Tolerance**: 1e-5 (medium accuracy)
- **Max Iterations**: 15
- **Mode**: Initial + incremental updates
- **Focus**: Fast real-time updates

```cpp
estimator.configureForRealTime(1e-5, 15, true);
auto result = estimator.estimate();  // Initial
auto incResult = estimator.estimateIncremental();  // Updates
```

#### Hybrid Setup
- **Method**: WLS (every cycle) + Robust (every 5 cycles)
- **Tolerance**: 1e-5 (WLS), 1e-6 (Robust)
- **Max Iterations**: 15 (WLS), 20 IRLS (Robust)
- **Mode**: Multi-layered strategy with pre-validation
- **Focus**: Automatic bad data handling with comprehensive validation
- **Pre-Validation**: Data consistency check, observability analysis, optimal placement

```cpp
// Pre-validation (before estimation)
DataConsistencyChecker consistencyChecker;
auto consistency = consistencyChecker.checkConsistency(*telemetry, *network);

ObservabilityAnalyzer analyzer;
bool observable = analyzer.isFullyObservable(*network, *telemetry);
if (!observable) {
    OptimalPlacement placement;
    auto placements = placement.findOptimalPlacement(*network, existing, 5);
}

// Real-time loop
estimator.configureForRealTime(1e-5, 15, true);
auto wlsResult = estimator.estimateIncremental();  // Every cycle
if (cycleCount % 5 == 0) {
    robustEstimator.estimate(...);  // Periodic
    // All values are automatically computed by estimate() (GPU-accelerated)
}
```

#### Advanced Setup
- **Method**: Standard WLS + Robust + Load Flow
- **Tolerance**: 1e-8 (WLS), 1e-6 (Robust)
- **Max Iterations**: 50 (WLS), 50 IRLS (Robust)
- **Mode**: Sequential estimation (WLS → Robust) with advanced features
- **Focus**: Feature demonstration with robust estimation and value extraction

```cpp
// Step 1: Standard WLS
estimator.configureForOffline(1e-8, 50, true);
auto wlsResult = estimator.estimate();

// Step 2: Robust estimation (uses WLS result as initial state)
auto robustState = std::make_unique<StateVector>(*wlsResult.state);
auto robustResult = robustEstimator.estimate(*robustState, *network, *telemetry);

// Step 3: All values are automatically computed by estimate() (GPU-accelerated)

// Also includes: load flow, optimal placement, multi-area, WLS vs Robust comparison
```

**Winner**: 
- **Accuracy**: Offline/Advanced
- **Speed**: Real-Time
- **Bad Data Handling**: Hybrid

---

### 2. Real-Time Capabilities

#### Offline Setup
- **Real-Time Updates**: ❌ No
- **Telemetry Processing**: ❌ No
- **Incremental Estimation**: ❌ No
- **Update Loop**: ❌ No
- **Use Case**: One-time analysis

#### Real-Time Setup
- **Real-Time Updates**: ✅ Yes (every cycle)
- **Telemetry Processing**: ✅ Yes (asynchronous)
- **Incremental Estimation**: ✅ Yes (faster convergence)
- **Update Loop**: ✅ Yes (10 cycles simulated)
- **Use Case**: Continuous operation

```cpp
estimator.getTelemetryProcessor().startRealTimeProcessing();
for (int i = 0; i < NUM_UPDATES; ++i) {
    estimator.getTelemetryProcessor().updateMeasurement(update);
    auto incResult = estimator.estimateIncremental();
}
```

#### Hybrid Setup
- **Real-Time Updates**: ✅ Yes (every cycle)
- **Telemetry Processing**: ✅ Yes (asynchronous)
- **Incremental Estimation**: ✅ Yes (faster convergence)
- **Update Loop**: ✅ Yes (20 cycles simulated)
- **Pre-Validation**: ✅ Yes (data consistency, observability, optimal placement)
- **Use Case**: Continuous operation with bad data handling and comprehensive validation

```cpp
estimator.getTelemetryProcessor().startRealTimeProcessing();
for (int i = 0; i < NUM_CYCLES; ++i) {
    estimator.getTelemetryProcessor().updateMeasurement(update);
    auto wlsResult = estimator.estimateIncremental();  // Every cycle
    if (cycleCount % 5 == 0) {
        robustEstimator.estimate(...);  // Periodic
    }
}
```

#### Advanced Setup
- **Real-Time Updates**: ❌ No
- **Telemetry Processing**: ❌ No
- **Incremental Estimation**: ❌ No
- **Update Loop**: ❌ No
- **Use Case**: Feature demonstration

**Winner**: Real-Time and Hybrid (both support real-time operation)

---

### 3. Bad Data Handling

#### Offline Setup
- **Detection**: ✅ Yes (chi-square, normalized residual)
- **Automatic Handling**: ❌ No
- **Timing**: Once at end
- **Method**: Statistical tests only

```cpp
badDataDetector.setNormalizedResidualThreshold(3.0);
auto badDataResult = badDataDetector.detectBadData(...);
```

#### Real-Time Setup
- **Detection**: ✅ Yes (chi-square, normalized residual)
- **Automatic Handling**: ❌ No
- **Timing**: Once at end
- **Method**: Statistical tests only

#### Hybrid Setup
- **Detection**: ✅ Yes (every 3 cycles)
- **Automatic Handling**: ✅ Yes (robust estimation every 5 cycles)
- **Timing**: Periodic during operation
- **Method**: Statistical tests + robust estimation

```cpp
// Periodic bad data detection
if (cycleCount % 3 == 0) {
    badDataDetector.detectBadData(...);
}

// Periodic robust estimation (handles bad data automatically)
if (cycleCount % 5 == 0) {
    robustEstimator.estimate(...);  // Down-weights outliers
}
```

#### Advanced Setup
- **Detection**: ❌ No (but shows robust weights analysis)
- **Automatic Handling**: ✅ Yes (robust estimation runs and down-weights outliers)
- **Timing**: After WLS estimation
- **Method**: Robust estimation (IRLS with M-estimators) + weight analysis
- **Features**: Shows which measurements were down-weighted, WLS vs Robust comparison

**Winner**: Hybrid (automatic handling + periodic detection)

---

### 4. System Monitoring

#### Offline Setup
- **Voltage Violations**: ✅ Yes (0.95 - 1.05 p.u.)
- **Branch Overloads**: ✅ Yes (90% of rating)
- **Real-Time Monitoring**: ❌ No
- **Violation Reporting**: ✅ Yes (in reports)

```cpp
if (vPU < 0.95 || vPU > 1.05) {
    std::cout << "⚠ Voltage violation at Bus " << bus->getId() << "\n";
}
```

#### Real-Time Setup
- **Voltage Violations**: ✅ Yes (0.95 - 1.05 p.u.)
- **Branch Overloads**: ✅ Yes (90% of rating)
- **Real-Time Monitoring**: ✅ Yes (during update loop)
- **Violation Reporting**: ✅ Yes (in reports)

```cpp
// Real-time monitoring during update loop
if (incResult.state) {
    // All values are automatically computed by estimateIncremental() (GPU-accelerated)
    // Check violations in real-time
}
```

#### Hybrid Setup
- **Voltage Violations**: ✅ Yes (displayed in computed values)
- **Branch Overloads**: ✅ Yes (displayed in computed values)
- **Real-Time Monitoring**: ✅ Yes (during update loop)
- **Violation Reporting**: ✅ Yes (console output)

#### Advanced Setup
- **Voltage Violations**: ❌ No
- **Branch Overloads**: ❌ No
- **Real-Time Monitoring**: ❌ No
- **Violation Reporting**: ❌ No

**Winner**: Offline and Real-Time (comprehensive monitoring)

---

### 5. Pre-Estimation Validation

#### Offline Setup
- **Observability Check**: ✅ Yes
- **Data Consistency**: ✅ Yes
- **Pre-Validation**: ✅ Yes (comprehensive)
- **Approach**: Validate thoroughly before estimation

```cpp
// Observability analysis
ObservabilityAnalyzer analyzer;
bool observable = analyzer.isFullyObservable(*network, *telemetry);

// Data consistency check
DataConsistencyChecker consistencyChecker;
auto consistency = consistencyChecker.checkConsistency(*telemetry, *network);
```

#### Real-Time Setup
- **Observability Check**: ❌ No
- **Data Consistency**: ✅ Yes
- **Pre-Validation**: ✅ Yes (basic)
- **Approach**: Quick validation then estimate

```cpp
// Data consistency check only
DataConsistencyChecker consistencyChecker;
auto consistency = consistencyChecker.checkConsistency(*telemetry, *network);
```

#### Hybrid Setup
- **Observability Check**: ✅ Yes (before estimation)
- **Data Consistency**: ✅ Yes (before estimation)
- **Optimal Placement**: ✅ Yes (if not observable)
- **Pre-Validation**: ✅ Yes (comprehensive)
- **Approach**: Validate thoroughly before estimation

#### Advanced Setup
- **Observability Check**: ❌ No
- **Data Consistency**: ❌ No
- **Pre-Validation**: ❌ No
- **Approach**: Direct to estimation

**Winner**: Offline and Hybrid (both have comprehensive validation)

---

### 6. Report Generation

#### Offline Setup
- **JSON Results**: ✅ Yes (`offline_results.json`)
- **Comparison Report**: ✅ Yes (`offline_comparison.txt`)
- **System Summary**: ✅ Yes (`offline_summary.txt`)
- **Output**: Comprehensive files

#### Real-Time Setup
- **JSON Results**: ✅ Yes (`realtime_results.json`)
- **Comparison Report**: ✅ Yes (`realtime_comparison.txt`)
- **System Summary**: ✅ Yes (`realtime_summary.txt`)
- **Output**: Comprehensive files

#### Hybrid Setup
- **JSON Results**: ❌ No
- **Comparison Report**: ❌ No
- **System Summary**: ❌ No
- **Output**: Console only (detailed computed values tables)

#### Advanced Setup
- **JSON Results**: ❌ No
- **Comparison Report**: ❌ No
- **System Summary**: ❌ No
- **Output**: Console only

**Winner**: Offline and Real-Time (comprehensive reporting)

---

### 7. Advanced Features

#### Offline Setup
- **Load Flow**: ❌ No
- **Optimal Placement**: ❌ No
- **Transformer Config**: ❌ No
- **PMU Support**: ❌ No
- **Multi-Area**: ❌ No

#### Real-Time Setup
- **Load Flow**: ❌ No
- **Optimal Placement**: ❌ No
- **Transformer Config**: ❌ No
- **PMU Support**: ❌ No
- **Multi-Area**: ❌ No

#### Hybrid Setup
- **Load Flow**: ❌ No
- **Optimal Placement**: ✅ Yes (if observability issues detected)
- **Transformer Config**: ❌ No
- **PMU Support**: ❌ No
- **Multi-Area**: ❌ No

#### Advanced Setup
- **Load Flow**: ✅ Yes
- **Optimal Placement**: ✅ Yes
- **Transformer Config**: ✅ Yes
- **PMU Support**: ✅ Yes
- **Multi-Area**: ✅ Yes (3-level hierarchy)

```cpp
// Load flow
LoadFlow loadflow;
auto lfResult = loadflow.solve(*network);

// Optimal placement
OptimalPlacement placement;
auto placements = placement.findOptimalPlacement(*network, existing, 5);

// Multi-area
MultiAreaEstimator multiArea;
auto multiResult = multiArea.estimateHierarchical();
```

**Winner**: Advanced (all advanced features)

---

### 8. Computed Values Extraction

#### Offline Setup
- **Voltage Estimates**: ✅ Yes
- **Power Injections**: ✅ Yes
- **Power Flows**: ✅ Yes
- **Current Values**: ✅ Yes
- **Access**: Via Bus/Branch getters

#### Real-Time Setup
- **Voltage Estimates**: ✅ Yes
- **Power Injections**: ✅ Yes
- **Power Flows**: ✅ Yes
- **Current Values**: ✅ Yes
- **Access**: Via Bus/Branch getters

#### Hybrid Setup
- **Voltage Estimates**: ✅ Yes (from robust estimation)
- **Power Injections**: ✅ Yes (from robust estimation)
- **Power Flows**: ✅ Yes (from robust estimation)
- **Current Values**: ✅ Yes (from robust estimation)
- **Access**: Via Bus/Branch getters
- **Display**: Detailed tables showing all computed values
- **Timing**: Computed during robust estimation cycles and at end

#### Advanced Setup
- **Voltage Estimates**: ✅ Yes (from robust estimation)
- **Power Injections**: ✅ Yes (from robust estimation)
- **Power Flows**: ✅ Yes (from robust estimation)
- **Current Values**: ✅ Yes (from robust estimation)
- **Access**: Via Bus/Branch getters
- **Display**: Detailed tables showing all computed values
- **Comparison**: WLS vs Robust results shown side-by-side

**Winner**: All four setups (all support computed values extraction)

---

## Performance Comparison

| Setup | Estimation Time | Update Time | Robust Overhead | Total Per Cycle |
|-------|----------------|-------------|-----------------|-----------------|
| **Offline** | 100-500 ms | N/A | None | N/A (single run) |
| **Real-Time** | 10-50 ms | 10-50 ms | None | 10-50 ms |
| **Hybrid** | 10-50 ms | 10-50 ms | 100-500 ms (every 5 cycles) | 10-50 ms + periodic |
| **Advanced** | 200-1000 ms | N/A | 100-500 ms | N/A (single run) |

**Note**: 
- Offline and Advanced are single-run (no update loop)
- Real-Time and Hybrid support continuous updates
- Hybrid has periodic robust estimation overhead

---

## When to Use Each Setup

### Use Offline Setup When:
- ✅ You need **high accuracy** (planning studies, analysis)
- ✅ You require **comprehensive validation** (observability, consistency)
- ✅ You need **detailed reports** (JSON, comparison, summary)
- ✅ You want **system monitoring** (violations, overloads)
- ✅ You're doing **one-time analysis** (not real-time)
- ✅ You need **complete value extraction**

**Example Use Cases:**
- Planning studies
- Historical data analysis
- Research and development
- System validation before deployment
- Offline validation and testing

### Use Real-Time Setup When:
- ✅ You need **production monitoring** (violations, overloads)
- ✅ You require **real-time updates** (SCADA/PMU integration)
- ✅ You need **comprehensive reporting** (JSON, comparison, summary)
- ✅ You want **fast performance** (no robust overhead)
- ✅ You're deploying to **production SCADA/EMS systems**
- ✅ You need **pre-validation** (data consistency)

**Example Use Cases:**
- Production SCADA systems
- Energy Management Systems (EMS)
- Real-time control center applications
- Systems requiring continuous monitoring
- Applications needing detailed reports for operators

### Use Hybrid Setup When:
- ✅ Your system has **occasional bad data** that needs automatic handling
- ✅ You need **robust estimation** to handle outliers automatically
- ✅ You require **real-time updates** (SCADA/PMU integration)
- ✅ Bad data detection alone **isn't sufficient** (need automatic correction)
- ✅ You can tolerate **slightly slower performance** for better accuracy
- ✅ You want a **multi-layered approach** (WLS + robust + detection)
- ✅ You need **comprehensive pre-validation** (data consistency, observability)
- ✅ You want **optimal measurement placement** recommendations
- ✅ You need **computed values extraction** (voltage, power, current)

**Example Use Cases:**
- Systems with unreliable measurement devices
- Real-time systems with occasional bad data
- Applications requiring automatic bad data handling
- Systems where bad data can't be easily removed
- Production systems with data quality issues

### Use Advanced Setup When:
- ✅ You want to **demonstrate all features** (robust, load flow, multi-area, etc.)
- ✅ You need **robust estimation** with detailed value extraction and comparison
- ✅ You want to **compare WLS vs Robust** estimation results side-by-side
- ✅ You need to **analyze robust weights** to identify down-weighted measurements
- ✅ You need **load flow** for initial state or validation
- ✅ You're doing **planning studies** (optimal measurement placement)
- ✅ You have **transformers** requiring accurate modeling
- ✅ You're integrating **PMU data**
- ✅ You're working with **large-scale systems** (multi-area)
- ✅ You need **high accuracy** (1e-8 tolerance) with robust handling

**Example Use Cases:**
- Feature demonstration and testing
- Comparing WLS and robust estimation methods
- Analyzing which measurements are down-weighted by robust estimator
- Planning studies for meter placement
- Systems with transformers requiring accurate modeling
- PMU integration for enhanced observability
- Large-scale systems requiring distributed estimation
- Research applications exploring all capabilities
- Understanding robust estimation behavior with bad data

---

## Recommendation Matrix

| Scenario | Recommended Setup |
|----------|------------------|
| Planning/analysis studies | **Offline Setup** |
| Production SCADA/EMS | **Real-Time Setup** |
| Systems with bad data | **Hybrid Setup** |
| Need monitoring/alerts | **Offline/Real-Time/Hybrid Setup** |
| Need comprehensive reports | **Offline/Real-Time Setup** |
| Need pre-validation + real-time | **Hybrid Setup** |
| Need optimal placement + real-time | **Hybrid Setup** |
| Research/feature exploration | **Advanced Setup** |
| High-frequency real-time updates | **Real-Time Setup** |
| Automatic bad data handling | **Hybrid Setup** |
| Need load flow | **Advanced Setup** |
| Need optimal placement | **Advanced Setup** |
| Need multi-area | **Advanced Setup** |
| Need PMU support | **Advanced Setup** |
| Need transformer modeling | **Advanced Setup** |
| Need robust estimation with value extraction | **Advanced Setup** |
| Want WLS vs Robust comparison | **Advanced Setup** |

---

## Summary

**Offline Setup**: High accuracy analysis with comprehensive validation and reporting  
**Real-Time Setup**: Production monitoring with real-time updates and comprehensive reporting  
**Hybrid Setup**: Real-time operation with automatic bad data handling, pre-validation, observability analysis, optimal placement, and computed values extraction  
**Advanced Setup**: Feature demonstration with all advanced capabilities including robust estimation with value extraction and WLS comparison

Choose based on your primary need:
- **Accuracy & Analysis**: Offline
- **Production & Monitoring**: Real-Time
- **Bad Data Handling**: Hybrid
- **Feature Exploration**: Advanced
