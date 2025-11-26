# Documentation Index

Complete guide to all documentation in the State Estimation project.

## Getting Started

- **[README.md](../README.md)** - Project overview, features, quick start guide
- **[USAGE.md](USAGE.md)** - Complete usage guide with concepts, examples, and API reference
- **[SETTERS_GETTERS.md](SETTERS_GETTERS.md)** - Complete API reference for all methods

## Core Features

- **[MODEL_FORMAT.md](MODEL_FORMAT.md)** - Network model format specifications (IEEE, JSON, CSV, SCADA, PMU)
- **[IO_FORMATS.md](IO_FORMATS.md)** - File format specifications for measurements, devices, and network models
- **[MEASUREMENT_DEVICES.md](MEASUREMENT_DEVICES.md)** - Measurement device modeling (multimeters, voltmeters, CT/PT)
- **[MEASURED_VS_ESTIMATED.md](MEASURED_VS_ESTIMATED.md)** - Comparing measured values from devices with estimated values
- **[FEATURES.md](FEATURES.md)** - Complete feature list and production readiness assessment

## Examples & Comparison

- **[EXAMPLES_COMPARISON.md](EXAMPLES_COMPARISON.md)** - Guide to choosing the right example for your needs

## Performance & Optimization

- **[PERFORMANCE.md](PERFORMANCE.md)** - GPU acceleration, performance characteristics, and tuning guide

## Building

- **[BUILD_CUDA.md](BUILD_CUDA.md)** - Complete guide for building with CUDA support
- **[COMPILERS.md](COMPILERS.md)** - Supported compilers and compatibility matrix

## Comparison & Assessment

- **[ETAP_COMPARISON.md](ETAP_COMPARISON.md)** - Comparison with ETAP State Load Estimation

## Quick Reference

### Core Concepts

**State Estimation**: Calculates voltage magnitude and angle at every bus using measurements and network topology. Uses Weighted Least Squares (WLS) with Newton-Raphson solver.

**Real-Time Operation**: Updates measurements and topology on-the-fly without full reload. Uses incremental estimation for speed (~300-500 ms) and full estimation when topology changes (~500-700 ms).

**Topology Changes**: Automatic detection via circuit breaker status changes. When a breaker opens/closes, the system automatically rebuilds the Jacobian structure.

**Observability**: Determines if all buses can be estimated with available measurements. If unobservable, suggests additional measurement placements.

**Bad Data Detection**: Finds faulty measurements using statistical tests (chi-square, normalized residual). Measurements with residuals > 3 standard deviations are flagged.

### Advanced Features

- **PMU Support**: Complete C37.118 phasor measurement unit implementation
- **Multi-Area**: 3-level hierarchy (Region → Area → Zone) for large systems
- **Transformer Modeling**: Accurate tap ratio and phase shift support
- **Load Flow**: Newton-Raphson solver for power flow analysis
- **Robust Estimation**: M-estimators (Huber, Bi-square, Cauchy, Welsch) for bad data handling

### Performance Characteristics

- **GPU Acceleration**: 5-100x speedup (enabled by default)
- **CPU Parallelization**: 4-8x speedup with OpenMP
- **10,000 Bus Systems**: 20-50x overall, 100-500 ms per cycle (real-time capable)
- **Memory Pool**: 100-500x faster allocations (reuses memory)
- **Key Optimizations**: Direct pointer linking, fused kernels, unified buffers, O(1) lookups

## Examples

All examples include detailed comments and demonstrate real-world usage:

- **`offlinesetup.cpp`** - Offline analysis with high accuracy and validation
- **`realtimesetup.cpp`** - Production real-time setup with automatic topology detection
- **`hybridsetup.cpp`** - Hybrid setup combining WLS with periodic robust estimation
- **`advancedsetup.cpp`** - Advanced features (PMU, multi-area, transformers, load flow)
- **`compare_measured_estimated.cpp`** - Compare measured vs estimated values
- **`observability_example.cpp`** - Observability analysis and restoration

## Build & Configuration

See [BUILD_CUDA.md](BUILD_CUDA.md) for complete CUDA build instructions.

## Learning Path

**New to state estimation?**
1. Start with [USAGE.md](USAGE.md) - Read "What is State Estimation?" section
2. Try `offlinesetup.cpp` example
3. Read [EXAMPLES_COMPARISON.md](EXAMPLES_COMPARISON.md) to understand differences

**Building a real-time system?**
1. Read [USAGE.md](USAGE.md) - "Real-Time Operation" section
2. Study `realtimesetup.cpp` example
3. Read [PERFORMANCE.md](PERFORMANCE.md) for optimization tips

**Need API reference?**
1. See [SETTERS_GETTERS.md](SETTERS_GETTERS.md) for complete method reference
2. See [USAGE.md](USAGE.md) - "API Reference" section for common patterns
