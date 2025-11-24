# Building with CUDA

Complete guide for building the Power System State Estimation project with CUDA support and setting up your IDE.

## Prerequisites

### Required Software

1. **CUDA Toolkit 12.0 or higher** (12.1+ recommended)
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

3. **C++17 Compatible Compiler**
   - **Windows**: Visual Studio 2019, 2022, 2025, 2026+ (MSVC 19.20+)
   - **Linux**: GCC 7+ or Clang 10+
   - See [COMPILERS.md](COMPILERS.md) for detailed compiler compatibility matrix

4. **NVIDIA GPU**
   - Compute capability 7.5 or higher (Turing, Ampere, Ada, Hopper)
   - Check your GPU compute capability: https://developer.nvidia.com/cuda-gpus

## IDE Setup (Recommended)

To enable IntelliSense, code navigation ("Go to Definition"), and proper CUDA support in your IDE (VS Code, Visual Studio, Cursor), run the automated setup script:

### Windows

```powershell
.\setup_ide.bat
```

This script will:
1. Auto-detect your Visual Studio installation (2019-2026+).
2. Set up the correct environment variables (vcvars64).
3. Run CMake to generate `compile_commands.json`.
4. Configure the project for your IDE.

**After running the script, restart your IDE or reload the window.**

### Manual Setup

If you prefer manual configuration or are on Linux:

```bash
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
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

#### Basic Configuration (Windows/Linux)

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

# Run an example
./examples/basic_example  # Linux
examples\basic_example.exe  # Windows
```

## Troubleshooting

### Issue: CMake can't find CUDA

**Solution:**
```bash
# Set CUDA path explicitly
cmake .. -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x"
```

### Issue: "No CMAKE_CXX_COMPILER could be found" (Windows)

**Solution:**
Ensure Visual Studio C++ workload is installed. If you are not in a Developer Command Prompt, run `setup_ide.bat` which will automatically find your Visual Studio installation and set up the environment.

### Issue: Wrong CUDA Architecture

**Error:** `nvcc fatal: Unsupported gpu architecture 'sm_XX'`

**Solution:**
Check your GPU compute capability and update the flag:
```bash
cmake .. -DCUDA_ARCH=sm_86  # Example for RTX 3080
```

### Issue: OpenMP Not Found

**Solution:**
- **Linux**: `sudo apt-get install libomp-dev`
- **Windows**: Installed with Visual Studio (ensure C++ workload is present)

## Performance Tuning

Modify `include/sle/utils/CompileTimeConfig.h` to adjust precision (double/float) and block sizes.

Rebuild after changes:
```bash
cmake --build . --config Release --clean-first
```
