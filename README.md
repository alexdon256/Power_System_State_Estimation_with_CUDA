# Power System State Estimation with CUDA

A comprehensive C/C++ CUDA-accelerated power system State Estimation program that processes telemetry data to estimate bus voltage magnitudes and phase angles. The system supports **real-time operation** with on-the-fly model and measurement updates, and includes **highly optimized code** using template metaprogramming.

## Getting Started

### Quick Setup (IDE Configuration)

To set up the project for development (generate `compile_commands.json` for IntelliSense):

**Windows:**
```powershell
.\setup_ide.bat
```

**Manual:**
```bash
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

After running the setup, **restart your IDE** or reload the window to ensure code navigation works.

## Features

- **State Estimation**: Weighted Least Squares (WLS) with Newton-Raphson iterative solver
- **Power Estimation**: Active and reactive power injection and flow measurements
- **Current Estimation**: Current magnitude and phasor measurements (including PMU support)
- **Robust Estimation**: M-estimators (Huber, Bi-square, Cauchy, Welsch) for bad data handling
- **Load Flow Integration**: Newton-Raphson load flow solver
- **CUDA Acceleration**: GPU-accelerated parallel computation with optimized kernels
  - **10-50x speedup** for large systems (10,000+ buses)
  - Real-time capable: 100-500 ms per cycle for 10K buses
  - **Zero-Copy Topology Reuse**: Minimizes PCIe transfers for static topologies
  - **Asynchronous Pipeline**: Overlaps computation and data transfer
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

- **CUDA Toolkit 12.0+** (12.1+ recommended for VS2022)
- **CMake 3.18+**
- **C++17 Compiler** (MSVC 2019+, GCC 7+, Clang 10+)
- **NVIDIA GPU** (Compute Capability 7.5+)

See [COMPILERS.md](docs/COMPILERS.md) for detailed compiler compatibility matrix.

## Building & Installation

See [BUILD_CUDA.md](docs/BUILD_CUDA.md) for complete build instructions.

### Windows Build
```bash
mkdir build && cd build
cmake .. -DCUDA_ARCH=sm_75
cmake --build . --config Release
```

### Linux Build
```bash
mkdir build && cd build
cmake .. -DCUDA_ARCH=sm_75
cmake --build . -j$(nproc)
```

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

### Multi-Area Estimation

```cpp
#include <sle/multiarea/MultiAreaEstimator.h>

sle::multiarea::MultiAreaEstimator multiArea;

// Add zones/areas/regions...
multiArea.addZone(zone1);
multiArea.addArea(area1);
multiArea.addRegion(region1);

// Run hierarchical estimation
auto result = multiArea.estimateHierarchical();
```

## Examples

The `examples/` directory contains consolidated setup examples:
- `offlinesetup.cpp` - **Offline analysis setup**
- `realtimesetup.cpp` - **Production real-time setup**
- `hybridsetup.cpp` - **Hybrid setup** (WLS + Robust)
- `advancedsetup.cpp` - **Advanced features setup**

**See [EXAMPLES_COMPARISON.md](docs/EXAMPLES_COMPARISON.md) for detailed comparison.**

## Documentation

Complete documentation is available in the `docs/` directory. See **[INDEX.md](docs/INDEX.md)** for the complete index.

## License

Copyright (c) 2024 AlexD Oleksandr Don. All rights reserved.
