# Easy Usage Guide

## Quick Start

The State Estimation system is designed to be easy to use with minimal setup:

### Basic Usage

```cpp
#include <sle/interface/StateEstimator.h>

// Simplest way: load from files in one call
sle::interface::StateEstimator estimator;
if (estimator.loadFromFiles("network.dat", "measurements.csv")) {
    // Configure for your use case
    estimator.configureForRealTime();  // Fast, for real-time
    // or
    estimator.configureForOffline();   // Accurate, for analysis
    
    // Run estimation
    auto result = estimator.estimate();
    
    // Check results
    if (result.converged) {
        std::cout << "Estimation converged in " << result.iterations << " iterations\n";
        
        // Quick access to estimated voltages
        Real v = estimator.getVoltageMagnitude(5);  // Voltage at bus 5
        Real theta = estimator.getVoltageAngle(5);  // Angle at bus 5
    }
}
```

### Alternative: Manual Setup

```cpp
#include <sle/interface/StateEstimator.h>
#include <sle/interface/ModelLoader.h>
#include <sle/interface/MeasurementLoader.h>

// Load network model
auto network = sle::interface::ModelLoader::load("network.dat");

// Load measurements
auto telemetry = sle::interface::MeasurementLoader::loadTelemetry("measurements.csv", *network);

// Create estimator (GPU-accelerated by default)
sle::interface::StateEstimator estimator;
estimator.setNetwork(std::make_shared<sle::model::NetworkModel>(*network));
estimator.setTelemetryData(telemetry);

// Configure for your use case
estimator.configureForRealTime();  // or configureForOffline()

// Check if ready
if (estimator.isReady()) {
    auto result = estimator.estimate();
}
```

### Command Line Usage

```bash
# Simple usage
./SLE_main network.dat measurements.csv

# Output: results.json with estimated state
```

## API Simplification

### Automatic GPU Acceleration

GPU acceleration is **enabled by default**. No configuration needed:

```cpp
// GPU acceleration is automatic
sle::interface::StateEstimator estimator;
auto result = estimator.estimate();  // Uses GPU automatically
```

### Disable GPU (if needed)

```cpp
sle::math::SolverConfig config;
config.useGPU = false;  // Use CPU fallback
estimator.setSolverConfig(config);
```

### Simple Configuration

**Convenience Methods (Recommended):**
```cpp
// For real-time operation (fast, relaxed tolerance)
estimator.configureForRealTime();  // tolerance=1e-5, maxIter=15, GPU=true

// For offline analysis (accurate, tight tolerance)
estimator.configureForOffline();   // tolerance=1e-8, maxIter=50, GPU=true

// Custom settings
estimator.configureForRealTime(1e-4, 10, true);   // Custom real-time
estimator.configureForOffline(1e-9, 100, true);    // Maximum accuracy
```

**Manual Configuration (Advanced):**
```cpp
sle::math::SolverConfig config;
config.tolerance = 1e-6;      // Convergence tolerance
config.maxIterations = 50;     // Maximum iterations
config.useGPU = true;          // Enable GPU acceleration
estimator.setSolverConfig(config);
```

## File Format Support

The system automatically detects file formats:

```cpp
// IEEE Common Format
auto network = sle::interface::ModelLoader::load("network.dat");

// JSON format
auto network = sle::interface::ModelLoader::load("network.json");

// CSV measurements
auto telemetry = sle::interface::MeasurementLoader::loadTelemetry("measurements.csv", *network);

// SCADA format
auto telemetry = sle::interface::MeasurementLoader::loadTelemetry("scada.dat", *network);

// PMU format (C37.118)
auto telemetry = sle::interface::MeasurementLoader::loadTelemetry("pmu.bin", *network);
```

## Real-time Operation

```cpp
// Start real-time processing
estimator.getTelemetryProcessor().startRealTimeProcessing();

// Update measurements on the fly
sle::interface::TelemetryUpdate update;
update.deviceId = "METER_1";
update.type = sle::MeasurementType::P_INJECTION;
update.value = 1.5;
update.busId = 1;
estimator.getTelemetryProcessor().updateMeasurement(update);

// Run incremental estimation
auto result = estimator.estimateIncremental();
```

## Output

Results are automatically formatted:

```cpp
// Save to JSON
sle::interface::Results results(result);
sle::io::OutputFormatter::writeToFile("results.json", results, "json");

// Save to CSV
sle::io::OutputFormatter::writeToFile("results.csv", results, "csv");
```

## Comparison Reports

Compare measured vs estimated values to validate estimation:

```cpp
#include <sle/io/ComparisonReport.h>

// After estimation
auto comparisons = sle::io::ComparisonReport::compare(
    *telemetry, *result.state, *network);

// Write report
sle::io::ComparisonReport::writeReport("comparison_report.txt", comparisons);

// Analyze results
int badCount = 0;
for (const auto& comp : comparisons) {
    if (comp.isBad) badCount++;  // Normalized residual > 3.0
}
std::cout << "Bad measurements: " << badCount << "\n";
```

## Examples

See the `examples/` directory (all with detailed comments):

- `basic_example.cpp` - Complete workflow with step-by-step explanations
- `realtime_example.cpp` - Real-time operation with comparison reports
- `observability_example.cpp` - Observability analysis and restoration
- `advanced_features_example.cpp` - All features including multi-area, transformers, PMU
- `hybrid_robust_example.cpp` - Hybrid approach combining WLS, robust estimation, and bad data detection

## Performance

The system automatically optimizes for your hardware:

- **GPU available**: Uses CUDA acceleration (5-100x speedup)
- **GPU not available**: Falls back to optimized CPU code
- **Small systems**: Automatically uses appropriate algorithms

No manual tuning required!

