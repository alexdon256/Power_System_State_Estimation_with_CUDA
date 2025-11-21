# Power System State Estimation with CUDA

A comprehensive C/C++ CUDA-accelerated power system State Estimation program that processes telemetry data to estimate bus voltage magnitudes and phase angles. The system supports **real-time operation** with on-the-fly model and measurement updates, and includes **highly optimized code** using template metaprogramming.

## Features

- **State Estimation**: Weighted Least Squares (WLS) with Newton-Raphson iterative solver
- **Power Estimation**: Active and reactive power injection and flow measurements
- **Current Estimation**: Current magnitude and phasor measurements (including PMU support)
- **Robust Estimation**: M-estimators (Huber, Bi-square, Cauchy, Welsch) for bad data handling
- **Load Flow Integration**: Newton-Raphson load flow solver
- **CUDA Acceleration**: GPU-accelerated parallel computation with optimized kernels
  - **10-50x speedup** for large systems (10,000+ buses)
  - Real-time capable: 100-500 ms per cycle for 10K buses
- **cuSOLVER Integration**: Full sparse linear system solving
- **PMU Support**: Complete C37.118 phasor measurement unit support
- **Template Metaprogramming**: Compile-time optimizations for maximum performance
- **Real-Time Operation**: Update network models and measurements on the fly without full reload
- **Telemetry Processing**: Support for SCADA and PMU data with asynchronous updates
- **Observability Analysis**: Advanced observability with optimal measurement placement
- **Bad Data Detection**: Chi-square test and largest normalized residual methods
- **Virtual & Pseudo Measurements**: Automatic generation for observability restoration
- **Multi-Area Support**: 3-level hierarchy (Region → Area → Zone) for large systems
  - Hierarchical and distributed estimation
  - Parallel processing across zones/areas/regions
  - Scalable to 50,000+ buses
- **Transformer Modeling**: Accurate tap ratio and phase shift support
- **Measurement Devices**: Comprehensive support for various measurement types and devices
- **Load Distribution**: Intelligent load allocation algorithms
- **State Estimation Comparator**: Compare measured vs. estimated values
- **Easy to Use**: Convenience methods and sensible defaults

## Performance

See [PERFORMANCE.md](docs/PERFORMANCE.md) for complete performance guide.

**Key Performance:**
- **10,000 Bus Systems**: 20-50x overall speedup, 100-500 ms per cycle (real-time capable)
- **GPU Acceleration**: 5-100x speedup, enabled by default with automatic CPU fallback
- **CPU Parallelization**: 4-8x speedup with OpenMP (when GPU unavailable)
- **Optimizations**: FMA operations, memory pool, SIMD vectorization, OpenMP threading, warp shuffles

## Requirements

- CUDA Toolkit 12.0 or higher (12.1+ recommended for better MSVC support)
- CMake 3.18 or higher
- C++17 compatible compiler
- NVIDIA GPU with compute capability 7.5 or higher

## Library Type

The project builds as a **shared library (DLL on Windows, .so on Linux)** with all public APIs properly exported. This allows:
- Dynamic linking at runtime
- Smaller executable sizes
- Easier library updates without recompiling dependent applications
- Proper symbol visibility on all platforms

## Building

See [BUILD_CUDA.md](docs/BUILD_CUDA.md) for complete build instructions.

## Usage

### Basic State Estimation

```cpp
#include <sle/StateEstimator.h>

// Load network model
auto network = ModelLoader::loadFromIEEE("network.dat");

// Load telemetry measurements
auto telemetry = MeasurementLoader::loadTelemetry("measurements.csv", network);

// Create estimator
StateEstimator estimator;
estimator.setNetwork(network);
estimator.setTelemetryData(telemetry);

// Run estimation
auto result = estimator.estimate();
```

### Robust Estimation

```cpp
#include <sle/math/RobustEstimator.h>

sle::math::RobustEstimator robust;
robust.setConfig({sle::math::RobustWeightFunction::HUBER, 1.345});
auto result = robust.estimate(state, network, telemetry);
```

### Load Flow

```cpp
#include <sle/math/LoadFlow.h>

sle::math::LoadFlow loadflow;
auto result = loadflow.solve(network);
```

### PMU Support

```cpp
#include <sle/io/PMUData.h>

auto frames = sle::io::pmu::PMUParser::parseFromFile("pmu_data.bin");
auto measurement = sle::io::pmu::PMUParser::convertToMeasurement(frames[0], busId);
```

### Multi-Area Estimation (3-Level Hierarchy)

```cpp
#include <sle/multiarea/MultiAreaEstimator.h>

sle::multiarea::MultiAreaEstimator multiArea;

// Create zones (lowest level)
sle::multiarea::Zone zone1;
zone1.name = "NorthZone";
zone1.areaName = "PJM";
// ... configure zone
multiArea.addZone(zone1);

// Create areas (middle level)
sle::multiarea::Area area1;
area1.name = "PJM";
area1.regionName = "Eastern";
// ... configure area
multiArea.addArea(area1);

// Create regions (highest level)
sle::multiarea::Region region1;
region1.name = "Eastern";
// ... configure region
multiArea.addRegion(region1);

// Run hierarchical estimation
auto result = multiArea.estimateHierarchical();
```

### Optimal Measurement Placement

```cpp
#include <sle/observability/OptimalPlacement.h>

sle::observability::OptimalPlacement placement;
auto placements = placement.findOptimalPlacement(network, existing, maxMeas, budget);
```

## Examples

The `examples/` directory contains consolidated setup examples:
- `offlinesetup.cpp` - **Offline analysis setup** with high accuracy, comprehensive validation, and detailed reporting
- `realtimesetup.cpp` - **Production real-time setup** with asynchronous telemetry processing, incremental estimation, monitoring, and comprehensive reporting
- `hybridsetup.cpp` - **Hybrid setup** combining fast WLS with periodic robust estimation and bad data detection
- `advancedsetup.cpp` - **Advanced features setup** including robust estimation with value extraction and WLS comparison, load flow, optimal placement, transformers, PMU, and multi-area

**See [EXAMPLES_COMPARISON.md](docs/EXAMPLES_COMPARISON.md) for detailed comparison of hybrid vs real-time setups.**

Test data files are in `examples/ieee14/`:
- `network.dat` - IEEE 14-bus test case
- `measurements.csv` - Sample measurements

## Documentation

Complete documentation is available in the `docs/` directory. See **[INDEX.md](docs/INDEX.md)** for the complete index.

### Production Readiness

See **[FEATURES.md](docs/FEATURES.md)** for comprehensive feature list and production requirements.

**Production Readiness: 85-90%**
- ✅ **Core Functionality**: Complete and robust
- ✅ **Real-Time Operation**: Production-ready
- ✅ **Performance**: GPU-accelerated, scalable to 50,000+ buses
- ✅ **API Integration**: Clean C++ API with shared library support
- ⚠️ **Enterprise Features**: Logging, monitoring, configuration management (partial)
- ❌ **User Interface**: No GUI (technical/API use only)

### Quick Navigation

**For New Users:**
- [EASY_USAGE.md](docs/EASY_USAGE.md) - Quick start guide with convenience methods
- [API.md](docs/API.md) - API documentation with examples

**For Developers:**
- [API.md](docs/API.md) - Complete API documentation
- [SETTERS_GETTERS.md](docs/SETTERS_GETTERS.md) - All setters and getters reference
- [PERFORMANCE.md](docs/PERFORMANCE.md) - GPU acceleration and optimizations
- [BUILD_CUDA.md](docs/BUILD_CUDA.md) - Complete CUDA build guide
- [DOXYGEN.md](docs/DOXYGEN.md) - Doxygen API documentation guide

**For System Integrators:**
- [REALTIME.md](docs/REALTIME.md) - Real-time operation guide
- [MODEL_FORMAT.md](docs/MODEL_FORMAT.md) - File format specifications
- [FEATURES.md](docs/FEATURES.md) - Feature list, production requirements, and implementation status

**For Performance Analysis:**
- [PERFORMANCE.md](docs/PERFORMANCE.md) - GPU acceleration and optimizations

## Performance Tuning

Modify `include/sle/utils/CompileTimeConfig.h` to adjust:
- Precision (double/float)
- CUDA optimization flags
- Block sizes
- Algorithm selection

## License

Copyright (c) 2024 AlexD Oleksandr Don

All rights reserved.
