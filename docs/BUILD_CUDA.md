# Building with CUDA

Complete guide for building the Power System State Estimation project with CUDA support.

## Prerequisites

### Required Software

1. **CUDA Toolkit 11.0 or higher**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Verify installation:
     ```bash
     nvcc --version
     nvidia-smi
     ```

2. **CMake 3.18 or higher**
   - Download from: https://cmake.org/download/
   - Verify installation:
     ```bash
     cmake --version
     ```

3. **OpenMP (Optional but Recommended)**
   - **Linux**: `sudo apt-get install libomp-dev`
   - **macOS**: `brew install libomp`
   - **Windows**: Included with Visual Studio 2019+
   - Provides 4-8x CPU speedup when GPU unavailable

3. **C++17 Compatible Compiler**
   - **Linux**: GCC 7+ or Clang 5+
   - **Windows**: Visual Studio 2017+ (MSVC 19.14+)
   - **macOS**: Clang 5+ (Xcode 9+)

4. **NVIDIA GPU**
   - Compute capability 7.5 or higher (Turing, Ampere, Ada, Hopper)
   - Check your GPU compute capability: https://developer.nvidia.com/cuda-gpus

### Verify GPU

```bash
# Check GPU and driver
nvidia-smi

# Should show:
# - GPU model
# - Driver version
# - CUDA version
```

## Build Steps

### 1. Clone and Prepare

```bash
# Navigate to project directory
cd Power_System_State_Estimation_with_CUDA

# Create build directory
mkdir build
cd build
```

### 2. Configure with CMake

#### Basic Configuration

```bash
cmake .. -DCUDA_ARCH=sm_75
```

#### Common GPU Architectures

| GPU Architecture | Compute Capability | CMake Flag |
|------------------|-------------------|------------|
| Turing (RTX 20xx) | 7.5 | `sm_75` |
| Ampere (RTX 30xx, A100) | 8.0, 8.6 | `sm_80` or `sm_86` |
| Ada (RTX 40xx) | 8.9 | `sm_89` |
| Hopper (H100) | 9.0 | `sm_90` |

**Find your GPU's compute capability:**
```bash
# Linux
nvidia-smi --query-gpu=compute_cap --format=csv

# Or check: https://developer.nvidia.com/cuda-gpus
```

#### Advanced Configuration Options

```bash
cmake .. \
  -DCUDA_ARCH=sm_75 \
  -DUSE_CUSOLVER=ON \
  -DUSE_OPENMP=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_TESTS=ON
```

**Options:**
- `CUDA_ARCH`: CUDA compute capability (default: `sm_75`)
- `USE_CUSOLVER`: Enable cuSOLVER for sparse linear systems (default: `ON`)
- `USE_OPENMP`: Enable OpenMP for CPU parallelization (default: `ON` if found)
- `BUILD_EXAMPLES`: Build example programs (default: `ON`)
- `BUILD_TESTS`: Build test suite (default: `ON`)

### 3. Build

```bash
# Build the project
cmake --build . --config Release

# Or on Linux/macOS:
make -j$(nproc)  # Use all CPU cores
```

### 4. Verify Build

```bash
# Check if library was created
# Shared library (DLL on Windows, .so on Linux)
ls -la lib/libSLE.so*  # Linux (shared library)
ls -la bin/libSLE.dll  # Windows (DLL)
# or
dir lib\SLE.lib         # Windows (import library)
dir bin\SLE.dll         # Windows (DLL)

# Run an example
./examples/basic_example  # Linux/macOS
# or
examples\basic_example.exe  # Windows
```

**Note:** The project builds as a **shared library (DLL on Windows, .so on Linux)** by default. The DLL/.so file contains the actual code, while the .lib file (Windows) is the import library needed for linking.

## Platform-Specific Instructions

### Linux

```bash
# Install CUDA Toolkit (Ubuntu/Debian)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build
mkdir build && cd build
cmake .. -DCUDA_ARCH=sm_75
cmake --build . -j$(nproc)
```

### Windows

```bash
# Install CUDA Toolkit from NVIDIA website
# Ensure Visual Studio 2017+ is installed
# Note: VS2022 14.44+ requires -allow-unsupported-compiler flag (automatically set)

# Open Developer Command Prompt for VS
# Navigate to project
cd Power_System_State_Estimation_with_CUDA

# Build
mkdir build
cd build
cmake .. -DCUDA_ARCH=sm_75 -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Output files:
# - bin/Release/SLE.dll (DLL - runtime)
# - lib/Release/SLE.lib (import library - link time)
```

### macOS

**Note:** Modern macOS doesn't support NVIDIA GPUs. This project requires Linux or Windows with NVIDIA GPU.

## Troubleshooting

### Issue: CMake can't find CUDA

**Solution:**
```bash
# Set CUDA path explicitly
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Or set environment variable
export CUDA_PATH=/usr/local/cuda
cmake ..
```

### Issue: Wrong CUDA Architecture

**Error:** `nvcc fatal: Unsupported gpu architecture 'sm_XX'`

**Solution:**
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Use correct architecture
cmake .. -DCUDA_ARCH=sm_XX  # Replace XX with your GPU's compute capability
```

### Issue: CUDA Version Mismatch

**Error:** `CUDA version X.X is required, but found Y.Y`

**Solution:**
- Install CUDA Toolkit 11.0 or higher
- Update NVIDIA drivers: `nvidia-smi` shows required driver version

### Issue: Compilation Errors in CUDA Files

**Solution:**
```bash
# Clean build
rm -rf build/*
cmake .. -DCUDA_ARCH=sm_75
cmake --build . --clean-first
```

### Issue: IDE Shows Errors in CUDA Files

**Note:** IDEs may show false errors in `.cu` files because they use C++ parser, not nvcc. The code will compile correctly with nvcc. This is because:
- IDEs use regular C++ compiler for parsing (doesn't define `__CUDACC__`)
- Actual compilation uses nvcc (defines `__CUDACC__` and `__CUDA_ARCH__`)
- The code uses three-tier macro detection to handle both cases

**Solution:**
- Install CUDA extension for your IDE:
  - VS Code: "CUDA" extension
  - Visual Studio: CUDA Toolkit integration
- Verify actual compilation works (ignore IDE warnings)
- IDE errors are false positives - code compiles correctly with nvcc

### Issue: Runtime Error: "CUDA driver version is insufficient"

**Solution:**
- Update NVIDIA drivers: https://www.nvidia.com/drivers
- Check compatibility: `nvidia-smi` shows CUDA version

### Issue: Out of Memory During Compilation

**Solution:**
```bash
# Reduce parallel jobs
cmake --build . -j2  # Use 2 cores instead of all
```

### Issue: OpenMP Not Found

**Error:** `OpenMP not found - CPU parallelization disabled`

**Solution:**
```bash
# Linux (Ubuntu/Debian)
sudo apt-get install libomp-dev

# Linux (Fedora/RHEL)
sudo dnf install libgomp-devel

# macOS
brew install libomp

# Windows
# OpenMP is included with Visual Studio 2019+
# For MinGW, install: mingw-w64-x86_64-openmp
```

## Verification

### 1. Check CUDA Libraries

```bash
# Verify CUDA libraries are linked
ldd lib/libSLE.so | grep cuda  # Linux (shared library)
ldd bin/libSLE.dll | grep cuda  # Windows (DLL)
otool -L lib/libSLE.dylib | grep cuda  # macOS (if applicable)
```

**Note:** For shared libraries, use `ldd` (Linux) or `dumpbin /dependents` (Windows) to check dependencies.

### 2. Run Example

```bash
# Run basic example
./examples/basic_example

# Should output:
# - State estimation results
# - Convergence information
# - No CUDA errors
```

### 3. Check GPU Usage

```bash
# Monitor GPU during execution
watch -n 1 nvidia-smi

# Should show:
# - GPU utilization > 0%
# - Memory usage
# - Process name
```

## Library Type

The project builds as a **shared library (DLL/.so)** by default, which allows:
- Dynamic linking at runtime
- Smaller executable sizes
- Easier updates without recompiling dependent applications
- Proper symbol export/import on Windows

### Using the DLL/Shared Library

**Windows:**
- The DLL (`SLE.dll`) must be in the same directory as your executable or in the system PATH
- Link against the import library (`SLE.lib`) during compilation
- All public APIs are exported using `SLE_API` macro

**Linux:**
- The shared library (`libSLE.so`) must be in a library path (e.g., `/usr/local/lib` or set `LD_LIBRARY_PATH`)
- Link against the shared library during compilation: `-lSLE`
- All public APIs are exported with default visibility

### Export Macros

All public API classes and functions use the `SLE_API` macro for proper DLL export/import:
- When building the library: symbols are exported (`__declspec(dllexport)`)
- When using the library: symbols are imported (`__declspec(dllimport)`)
- On Linux: uses `__attribute__((visibility("default")))`

## Build Configuration Reference

### CMake Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_ARCH` | `sm_75` | CUDA compute capability |
| `USE_CUSOLVER` | `ON` | Enable cuSOLVER |
| `USE_OPENMP` | `ON` | Enable OpenMP for CPU parallelization |
| `BUILD_EXAMPLES` | `ON` | Build examples |
| `BUILD_TESTS` | `ON` | Build tests |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_PATH` | CUDA Toolkit installation path |
| `CUDA_HOME` | Alternative to CUDA_PATH |
| `LD_LIBRARY_PATH` | Library search path (Linux) |

## Performance Tuning

### Optimize for Your GPU

```bash
# For RTX 3080 (Ampere, sm_86)
cmake .. -DCUDA_ARCH=sm_86

# For RTX 4090 (Ada, sm_89)
cmake .. -DCUDA_ARCH=sm_89

# For A100 (Ampere, sm_80)
cmake .. -DCUDA_ARCH=sm_80
```

### Build Type

```bash
# Release build (optimized)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Debug build (with symbols)
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

## Linking Against the Library

### Windows (MSVC)

```cpp
// In your CMakeLists.txt or project settings:
target_link_libraries(your_target PRIVATE SLE)

// The DLL (SLE.dll) must be:
// - In the same directory as your executable, OR
// - In a directory in your system PATH
```

### Linux

```bash
# Link against the shared library
g++ your_app.cpp -lSLE -L/path/to/lib -I/path/to/include

# Or set library path:
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
```

### CMake Integration

```cmake
# Find the library (if installed)
find_library(SLE_LIBRARY SLE PATHS ${CMAKE_PREFIX_PATH}/lib)
find_path(SLE_INCLUDE_DIR sle/StateEstimator.h PATHS ${CMAKE_PREFIX_PATH}/include)

# Link your target
target_link_libraries(your_target PRIVATE ${SLE_LIBRARY})
target_include_directories(your_target PRIVATE ${SLE_INCLUDE_DIR})
```

## Next Steps

After successful build:
1. See [EASY_USAGE.md](EASY_USAGE.md) for quick start
2. Check [PERFORMANCE.md](PERFORMANCE.md) for performance tuning
3. Review [API.md](API.md) for API documentation

## Support

For build issues:
1. Check CUDA Toolkit installation: `nvcc --version`
2. Verify GPU compatibility: `nvidia-smi`
3. Review CMake output for errors
4. Check [Troubleshooting](#troubleshooting) section above

