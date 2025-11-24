# Documentation Index

Complete guide to all documentation in the State Estimation project.

## Getting Started

- **[README.md](../README.md)** - Project overview, features, quick start guide, and documentation navigation
- **[USAGE.md](USAGE.md)** - Complete usage guide from quick start to advanced operations
- **[API.md](API.md)** - Complete API documentation with detailed examples
- **[SETTERS_GETTERS.md](SETTERS_GETTERS.md)** - Complete reference for all setters and getters

## Core Features

- **[REALTIME.md](REALTIME.md)** - Real-time operation and dynamic network updates guide
- **[MODEL_FORMAT.md](MODEL_FORMAT.md)** - Model format specifications (IEEE, JSON, CSV, SCADA, PMU)
- **[IO_FORMATS.md](IO_FORMATS.md)** - File format specifications for measurements, devices, and network models
- **[MEASUREMENT_DEVICES.md](MEASUREMENT_DEVICES.md)** - Measurement device modeling (multimeters, voltmeters, CT/PT)
- **[MEASURED_VS_ESTIMATED.md](MEASURED_VS_ESTIMATED.md)** - Comparing measured values from devices with estimated values
- **[FEATURES.md](FEATURES.md)** - Complete feature list, production requirements, and implementation status

## Performance & Optimization

- **[PERFORMANCE.md](PERFORMANCE.md)** - GPU acceleration, CUDA optimizations, and performance tuning

## Building

- **[BUILD_CUDA.md](BUILD_CUDA.md)** - Complete guide for building with CUDA support
- **[COMPILERS.md](COMPILERS.md)** - Supported compilers and compatibility matrix

## Comparison & Assessment

- **[ETAP_COMPARISON.md](ETAP_COMPARISON.md)** - Comparison with ETAP State Load Estimation
- **[EXAMPLES_COMPARISON.md](EXAMPLES_COMPARISON.md)** - Comparison of all setup examples (offline, real-time, hybrid, advanced)

## Quick Reference

### Core Functionality
- **State Estimation**: WLS with Newton-Raphson, robust estimation (M-estimators)
- **Real-time**: On-the-fly model/measurement updates, incremental estimation
- **Observability**: Analysis, restoration, optimal placement
- **Bad Data Detection**: Chi-square and normalized residual tests

### Advanced Features
- **PMU Support**: Complete C37.118 implementation
- **Multi-Area**: 3-level hierarchy (Region → Area → Zone)
- **Transformer Modeling**: Tap ratio and phase shift support
- **Load Flow**: Newton-Raphson solver
- **Hybrid Approach**: WLS + robust estimation + bad data detection

### Performance
- **GPU Acceleration**: 5-100x speedup (see [PERFORMANCE.md](PERFORMANCE.md))
- **CPU Parallelization**: 4-8x speedup with OpenMP
- **10,000 Bus Systems**: 20-50x overall, 100-500 ms per cycle (real-time capable)
- **Memory Pool**: 100-500x faster allocations (reuses memory)

## Examples

All examples include detailed comments:
- `offlinesetup.cpp` - Offline analysis with high accuracy, comprehensive validation, and detailed reporting
- `realtimesetup.cpp` - Production real-time setup with asynchronous telemetry processing, incremental estimation, monitoring, and comprehensive reporting
- `hybridsetup.cpp` - Hybrid setup combining fast WLS with periodic robust estimation, bad data detection, pre-validation (data consistency, observability), optimal placement, and computed values extraction
- `advancedsetup.cpp` - Advanced features setup including robust estimation with value extraction and WLS comparison, load flow, optimal placement, transformers, PMU, and multi-area
- `compare_measured_estimated.cpp` - Compare measured values from devices with estimated values from state estimation
- `observability_example.cpp` - Observability analysis and restoration

## Build & Configuration

See [BUILD_CUDA.md](BUILD_CUDA.md) for complete CUDA build instructions. Compiler optimizations are automatically enabled.

