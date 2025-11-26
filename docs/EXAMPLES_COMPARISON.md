# Examples Comparison

Quick guide to choosing the right example for your needs.

## Quick Decision Guide

**Choose based on your use case:**

| Your Situation | Use This Example | Why |
|----------------|------------------|-----|
| Production SCADA/EMS system | `realtimesetup.cpp` | Automatic topology detection, fast updates |
| Planning study or analysis | `offlinesetup.cpp` | High accuracy, comprehensive validation |
| System with bad sensors | `hybridsetup.cpp` | Robust estimation handles bad data |
| Exploring features | `advancedsetup.cpp` | Shows all capabilities |

## Feature Comparison

| Feature | Offline | Real-Time | Hybrid | Advanced |
|---------|---------|-----------|--------|----------|
| **Estimation** | WLS | WLS | WLS + Robust | WLS + Robust + Load Flow |
| **Tolerance** | 1e-8 (high) | 1e-5 (medium) | 1e-5 (medium) | 1e-8 (high) |
| **Real-Time Updates** | ❌ | ✅ | ✅ | ❌ |
| **Incremental Estimation** | ❌ | ✅ | ✅ | ❌ |
| **Topology Detection** | ❌ | ✅ Auto | ❌ | ❌ |
| **Robust Estimation** | ❌ | ❌ | ✅ Periodic | ✅ |
| **Bad Data Detection** | ✅ Once | ✅ Once | ✅ Periodic | ❌ |
| **System Monitoring** | ✅ | ✅ | ✅ | ❌ |
| **Observability Check** | ✅ | ❌ | ✅ | ❌ |
| **Load Flow** | ❌ | ❌ | ❌ | ✅ |
| **PMU/Multi-Area** | ❌ | ❌ | ❌ | ✅ |
| **Speed** | Slow (~1-2s) | Fast (~300-700ms) | Medium (~1-1.5s) | Slow (~2-5s) |

## Detailed Examples

### 1. Offline Setup (`offlinesetup.cpp`)

**When to use**: Planning studies, accuracy validation, one-time analysis

**What it does**:
- Runs high-accuracy estimation (tolerance 1e-8)
- Validates data consistency and observability
- Detects bad data once
- Generates comprehensive reports
- Monitors system for violations

**Performance**: Slower (~1-2 seconds) but very accurate

**Code pattern**:
```cpp
estimator.configureForOffline(1e-8, 50);
auto result = estimator.estimate();
// Process results...
```

### 2. Real-Time Setup (`realtimesetup.cpp`)

**When to use**: Production SCADA/EMS systems, live monitoring

**What it does**:
- Runs fast estimation (tolerance 1e-5)
- Updates measurements in real-time
- **Automatically detects topology changes** via circuit breakers
- Uses incremental estimation for speed (~300-500 ms)
- Uses full estimation when topology changes (~500-700 ms)
- Monitors system continuously
- Detects bad data periodically

**Performance**: Fast (~300-500 ms for measurements, ~500-700 ms with topology change)

**Code pattern**:
```cpp
estimator.configureForRealTime(1e-5, 15);

while (running) {
    updateMeasurements();
    if (estimator.isTopologyChanged()) {
        result = estimator.estimate();  // Topology changed
    } else {
        result = estimator.estimateIncremental();  // Measurement-only
    }
}
```

**Key feature**: Automatic topology change detection - just update circuit breaker status, and the system handles the rest.

### 3. Hybrid Setup (`hybridsetup.cpp`)

**When to use**: Systems with unreliable sensors or frequent bad data

**What it does**:
- Combines WLS with robust estimation
- Runs robust estimation periodically (every N cycles)
- Handles bad data automatically via M-estimators
- Updates measurements in real-time
- Validates observability
- Finds optimal measurement placement

**Performance**: Medium (~1-1.5 seconds with robust estimation)

**Code pattern**:
```cpp
// Periodic robust estimation
if (cycle % 10 == 0) {
    auto robustResult = estimator.robustEstimate();
} else {
    auto result = estimator.estimateIncremental();
}
```

**Key feature**: Robust estimation automatically down-weights bad measurements, giving better results even with faulty sensors.

### 4. Advanced Setup (`advancedsetup.cpp`)

**When to use**: Exploring features, research, complex systems

**What it does**:
- Demonstrates all features
- PMU support (C37.118)
- Multi-area hierarchy (Region → Area → Zone)
- Transformer configuration
- Load flow analysis
- Optimal measurement placement

**Performance**: Slower (~2-5 seconds) due to many features

## Understanding the Differences

### Estimation Methods Explained

**Full Estimation** (`estimate()`):
- Rebuilds Jacobian matrix structure from scratch
- Slower (~500-700 ms) but always accurate
- **Use when**: Topology changed, first run, or after major changes

**Incremental Estimation** (`estimateIncremental()`):
- Reuses previous Jacobian structure
- Faster (~300-500 ms) for small changes
- **Use when**: Only measurements changed, topology unchanged

**Robust Estimation** (`robustEstimate()`):
- Iteratively reweights measurements to down-weight bad data
- Slower (~1-1.5 seconds) but handles bad sensors
- **Use when**: System has unreliable sensors or frequent bad data

### Topology Change Handling

**Real-Time Setup** (automatic):
```cpp
// Circuit breaker status change automatically detected
cb->setStatus(false);  // Opens breaker
// System automatically:
//   1. Updates branch status
//   2. Sets topologyChanged_ flag
//   3. Marks model as updated

if (estimator.isTopologyChanged()) {
    result = estimator.estimate();  // Full estimation required
}
```

**Other Setups** (manual):
```cpp
// Manual topology update
branch->setStatus(false);
network->updateBranch(branchId, *branch);
estimator.markModelUpdated();
result = estimator.estimate();
```

**Why automatic is better**: Less code, fewer bugs, handles edge cases automatically.

## Performance Comparison

| Operation | Offline | Real-Time | Hybrid | Advanced |
|-----------|---------|-----------|--------|----------|
| Initial estimation | ~1-2s | ~500ms | ~1s | ~2-5s |
| Measurement update | N/A | ~300-500ms | ~300-500ms | N/A |
| Topology change | N/A | ~500-700ms | N/A | N/A |
| Robust estimation | N/A | N/A | ~1-1.5s | ~1-1.5s |

## Choosing Summary

- **Need speed + automatic topology detection?** → `realtimesetup.cpp`
- **Need accuracy + validation?** → `offlinesetup.cpp`
- **Have bad sensors?** → `hybridsetup.cpp`
- **Exploring features?** → `advancedsetup.cpp`

For more details, see individual example files in `examples/` directory.
