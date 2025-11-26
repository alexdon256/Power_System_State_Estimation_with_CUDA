# Performance Guide

Understanding GPU acceleration and performance characteristics.

## Performance Overview

**What to expect** for different system sizes:

| System Size | GPU Speedup | Time per Cycle | Status |
|-------------|-------------|----------------|--------|
| < 100 buses | 5-10x | < 10 ms | Very fast |
| 100-1000 buses | 10-50x | 10-50 ms | Fast |
| 1000-10000 buses | 50-100x | 50-200 ms | Real-time capable |
| > 10000 buses | 100x+ | 200-500 ms | Real-time capable |

**10,000 bus systems**: 20-50x overall speedup, 100-500 ms per cycle (suitable for real-time operation)

## How GPU Acceleration Works

**Why GPU?** Power system state estimation involves:
- Thousands of parallel calculations (measurement functions)
- Large sparse matrix operations (Jacobian computation)
- Linear system solving (Newton-Raphson iterations)

These operations are **perfectly suited for GPU parallelization**.

**What runs on GPU:**
- Measurement function evaluation (`h(x)`) - 10-100x speedup
- Jacobian matrix computation (`H(x)`) - 5-50x speedup
- Sparse matrix operations (cuSPARSE) - 3-20x speedup
- Linear system solving (cuSOLVER) - 5-30x speedup

**What runs on CPU:**
- Data loading and file I/O
- Result processing and reporting
- Control flow and coordination

## Key Optimizations

### Memory Pool (100-500x faster allocations)
**Problem**: Allocating GPU memory is slow (~milliseconds per allocation)

**Solution**: Reuse buffers across iterations instead of allocating each time

**Impact**: 100-500x faster memory operations

### Zero-Copy Topology Reuse (90%+ bandwidth reduction)
**Problem**: Uploading network topology to GPU is slow (~tens of milliseconds)

**Solution**: When topology doesn't change, reuse existing GPU data (`reuseStructure=true`)

**Impact**: 90%+ reduction in PCIe bandwidth usage

**When it helps**: Real-time systems where measurements change frequently but topology is static

### Incremental Estimation (~40% faster)
**Problem**: Rebuilding Jacobian structure is slow

**Solution**: Reuse previous structure when only measurements changed

**Impact**: ~300-500 ms vs ~500-700 ms (40% faster)

**When to use**: Measurement-only updates (no topology changes)

### Direct Pointer Linking (eliminates lookups)
**Problem**: Hash map lookups add overhead

**Solution**: Bus/Branch objects store direct pointers to measurement devices

**Impact**: O(1) direct access instead of O(log n) hash map lookup

### Fused Kernels (reduces overhead)
**Problem**: Multiple GPU kernel launches add overhead

**Solution**: Combine operations (e.g., `h(x)` + residual computation) in single kernel

**Impact**: 2-5% improvement by reducing launch overhead

## Real-Time Architecture

**How it works** for high-frequency updates:

**Initialization** (once):
1. Build network model and Jacobian structure
2. Allocate GPU buffers (persisted across iterations)
3. Initialize cuSPARSE/cuSOLVER handles

**Per-Cycle Update** (repeated):
1. Upload only changed measurement values (`z`) to GPU (~1-5 ms)
2. Reuse topology if unchanged (`reuseStructure=true`, saves ~20-40 ms)
3. Solve entirely on GPU (~50-200 ms)
4. Download only state vector (`v`, `θ`) to host (~1-5 ms)

**Total**: ~100-500 ms per cycle for 10K bus systems

**Why it's fast**: Minimizes host-device synchronization and PCIe transfers.

## Configuration

### Enable/Disable GPU

```cpp
sle::math::SolverConfig config;
config.useGPU = true;  // Default: true (automatic CPU fallback if GPU unavailable)
```

**When to disable**: Debugging, systems without GPU, or CPU-only requirements

### Set CUDA Architecture

```cmake
cmake .. -DCUDA_ARCH=sm_75  # Default: sm_75 (Turing+)
```

**Common architectures**:
- `sm_75` - Turing (RTX 20-series, GTX 16-series)
- `sm_80` - Ampere (RTX 30-series, A100)
- `sm_86` - Ampere (RTX 30-series mobile)
- `sm_89` - Ada Lovelace (RTX 40-series)

### Enable OpenMP (CPU parallelization)

```cmake
cmake .. -DUSE_OPENMP=ON  # Default: ON (if available)
```

**Impact**: 4-8x speedup on multi-core CPUs for host-side tasks

## Memory Usage

**For 2,000,000 measurements and 300,000 devices:**

| Component | RAM | VRAM |
|-----------|-----|------|
| **Total** | ~450-500 MB | ~580-600 MB |
| **Peak** | ~450 MB | ~620 MB |

**Scaling factors**:
- Measurements: ~48 bytes per measurement (RAM), ~12 bytes (VRAM)
- Devices: ~310 bytes per device (RAM)
- Buses: ~216 bytes per bus (RAM), ~24 bytes (VRAM)
- Branches: ~152 bytes per branch (RAM), ~48 bytes (VRAM)

**Memory optimizations**:
- Direct pointer linking reduces lookup overhead
- Unified pinned buffers reduce allocation overhead
- Memory pools reuse GPU buffers across iterations
- Sparse matrix formats minimize storage for large systems

## Performance Tuning

### For Maximum Speed

1. **Use incremental estimation** when topology doesn't change
2. **Enable GPU** (default)
3. **Use real-time mode** (`configureForRealTime()`)
4. **Batch measurement updates** when possible
5. **Reuse topology** (`reuseStructure=true`)

### For Maximum Accuracy

1. **Use offline mode** (`configureForOffline()`)
2. **Increase maxIterations** (default: 50)
3. **Tighten tolerance** (default: 1e-8)
4. **Run bad data detection** to remove outliers

## Profiling

**NVIDIA Nsight Systems** (timeline analysis):
```bash
nsys profile --trace=cuda,nvtx ./your_executable
```

**NVIDIA Nsight Compute** (kernel analysis):
```bash
ncu --kernel computePowerFlowPQ --set full ./your_executable
```

**What to look for**:
- Kernel execution time
- Memory transfer time
- PCIe bandwidth usage
- GPU utilization

## Requirements

- **CUDA Toolkit**: 12.0+ (12.1+ recommended)
- **GPU**: NVIDIA GPU with compute capability 7.5+ (Turing, Ampere, Ada, Hopper)
- **Memory**: 
  - RAM: 1 GB minimum, 2 GB recommended, 4 GB+ for production
  - VRAM: 1 GB minimum, 2 GB recommended, 4 GB+ for production

## Common Performance Issues

**Problem**: Estimation is slow
- **Check**: Is GPU enabled? (`config.useGPU = true`)
- **Check**: Are you using incremental estimation for measurement-only updates?
- **Check**: Is topology reuse enabled? (`reuseStructure=true`)

**Problem**: Memory usage is high
- **Check**: System size (measurements, buses, branches)
- **Solution**: Use sparse matrix formats (default)

**Problem**: GPU not detected
- **Check**: CUDA toolkit installed and in PATH
- **Check**: GPU compute capability (needs 7.5+)
- **Solution**: System automatically falls back to CPU
