# Supported Compilers

This document lists all supported compilers for building the Power System State Estimation project with CUDA support.

## Requirements

- **C++ Standard**: C++17 (required)
- **CUDA Toolkit**: 12.0 or higher (12.1+ recommended)
- **CMake**: 3.18 or higher

## Windows Compilers

### Microsoft Visual C++ (MSVC)

| Visual Studio Version | MSVC Version | Status | Notes |
|----------------------|--------------|--------|-------|
| Visual Studio 2019 | 19.20 - 19.29 | ✅ Fully Supported | Recommended for CUDA 12.0 |
| Visual Studio 2022 | 19.30 - 19.39 | ✅ Fully Supported | Best compatibility with CUDA 12.1+ |
| Visual Studio 2025 | 19.40+ | ✅ Supported | Requires `-allow-unsupported-compiler` flag |
| Visual Studio 2026 | 19.50+ | ✅ Supported | Requires `-allow-unsupported-compiler` flag, CUDA 12.1+ recommended |
| Visual Studio 2026 | 19.50.35718.0 | ✅ Fully Tested | Specifically tested and supported, works with CUDA 12.1+ |

**Notes:**
- All editions supported: Community, Professional, Enterprise, Preview, BuildTools
- CUDA 12.0+ officially supports MSVC 2019 and 2022
- Newer MSVC versions (2025, 2026) work with the `-allow-unsupported-compiler` flag (automatically added by CMakeLists.txt)
- **MSVC 19.50.35718.0 (VS 2026)** is specifically tested and supported
- CUDA 12.1+ recommended for MSVC 19.50+ for best compatibility
- Ensure "Desktop development with C++" workload is installed

**Setup:**
```batch
# Run setup script (auto-detects VS installation)
.\setup_ide.bat

# Or manually from Developer Command Prompt
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Linux Compilers

### GCC (GNU Compiler Collection)

| GCC Version | Status | Notes |
|-------------|--------|-------|
| GCC 7.x | ✅ Supported | Minimum version for C++17 |
| GCC 8.x | ✅ Supported | Good compatibility |
| GCC 9.x | ✅ Supported | Recommended |
| GCC 10.x | ✅ Supported | Excellent compatibility |
| GCC 11.x | ✅ Supported | Excellent compatibility |
| GCC 12.x | ✅ Supported | Excellent compatibility |
| GCC 13.x | ✅ Supported | Latest stable |

**Installation (Ubuntu/Debian):**
```bash
# Install GCC 11 (recommended)
sudo apt-get update
sudo apt-get install gcc-11 g++-11

# Set as default (optional)
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

**Build:**
```bash
# Specify compiler explicitly
cmake -B build -DCMAKE_CXX_COMPILER=g++-11 -DCUDA_ARCH=sm_75
cmake --build build -j$(nproc)
```

### Clang

| Clang Version | Status | Notes |
|---------------|--------|-------|
| Clang 10.x | ✅ Supported | Minimum recommended |
| Clang 11.x | ✅ Supported | Good compatibility |
| Clang 12.x | ✅ Supported | Good compatibility |
| Clang 13.x | ✅ Supported | Excellent compatibility |
| Clang 14.x | ✅ Supported | Excellent compatibility |
| Clang 15.x | ✅ Supported | Excellent compatibility |
| Clang 16.x | ✅ Supported | Excellent compatibility |
| Clang 17.x | ✅ Supported | Latest stable |

**Installation (Ubuntu/Debian):**
```bash
# Install Clang 15 (recommended)
sudo apt-get update
sudo apt-get install clang-15 libc++-15-dev libc++abi-15-dev

# Set as default (optional)
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
```

**Build:**
```bash
# Specify compiler explicitly
cmake -B build -DCMAKE_CXX_COMPILER=clang++-15 -DCUDA_ARCH=sm_75
cmake --build build -j$(nproc)
```

## CUDA Compiler Compatibility

### CUDA Toolkit 12.0

| Host Compiler | Windows | Linux | Notes |
|---------------|---------|-------|-------|
| MSVC 2019 | ✅ | N/A | Official support |
| MSVC 2022 | ✅ | N/A | Official support |
| GCC 7-12 | N/A | ✅ | Official support |
| Clang 10-16 | N/A | ✅ | Official support |

### CUDA Toolkit 12.1+

| Host Compiler | Windows | Linux | Notes |
|---------------|---------|-------|-------|
| MSVC 2019 | ✅ | N/A | Official support |
| MSVC 2022 | ✅ | N/A | Official support |
| MSVC 2025+ | ⚠️ | N/A | Requires `-allow-unsupported-compiler` |
| MSVC 19.50.35718.0 (VS 2026) | ✅ | N/A | Tested with CUDA 12.1+, flag auto-added |
| GCC 7-13 | N/A | ✅ | Official support |
| Clang 10-17 | N/A | ✅ | Official support |

## Compiler Feature Support

### Required C++17 Features

All supported compilers must support:
- Structured bindings
- `if constexpr`
- Fold expressions
- `std::optional`, `std::variant`, `std::any`
- Parallel algorithms (optional, for OpenMP)
- Class template argument deduction (CTAD)

### CUDA-Specific Features

- Unified Memory (CUDA 6.0+)
- Cooperative Groups (CUDA 9.0+)
- cuSPARSE API 11.0+ (for sparse matrix operations)
- cuSOLVER API (for linear system solving)

## Recommended Compiler Configurations

### Windows (Production)

**Best Choice:** Visual Studio 2022 Community Edition
- MSVC 19.30+
- CUDA 12.1+
- Excellent IDE integration
- Full debugging support

**Alternative:** Visual Studio 2026 Community Edition
- MSVC 19.50.35718.0+ (tested and supported)
- CUDA 12.1+ required
- Latest features and improvements
- Requires `-allow-unsupported-compiler` (automatically handled by CMakeLists.txt)

### Linux (Production)

**Best Choice:** GCC 11 or Clang 15
- Excellent CUDA compatibility
- Good optimization
- Stable performance

### Development/Testing

**Windows:** Visual Studio 2022 or 2025
**Linux:** GCC 12+ or Clang 16+

## Troubleshooting

### Issue: "Unsupported compiler version"

**Solution:**
- For MSVC 2025+ (19.40+) and MSVC 2026 (19.50+): The `-allow-unsupported-compiler` flag is automatically added by CMakeLists.txt
- MSVC 19.50.35718.0 is specifically tested and supported with CUDA 12.1+
- For older CUDA versions: Upgrade to CUDA 12.1+ or use an officially supported compiler
- Verify the flag is present: Check CMakeLists.txt line 125 for `-allow-unsupported-compiler`

### Issue: "C++17 features not available"

**Solution:**
- Ensure compiler version meets minimum requirements
- Check `CMAKE_CXX_STANDARD` is set to 17 in CMakeLists.txt
- Verify compiler supports C++17 (GCC 7+, Clang 5+, MSVC 2017+)

### Issue: CUDA compilation fails

**Solution:**
1. Verify CUDA Toolkit version: `nvcc --version`
2. Check compiler compatibility with CUDA version
3. Ensure compiler is in PATH
4. For MSVC: Run from Developer Command Prompt or use `setup_ide.bat`

## Compiler Detection

The build system automatically detects compilers. To verify:

```bash
# Check detected compiler
cmake -B build
# Look for: "CXX compiler: /path/to/compiler"

# Check CUDA compiler
nvcc --version
```

## Performance Considerations

- **GCC**: Generally produces faster code, better optimization
- **Clang**: Faster compilation, good optimization, better error messages
- **MSVC**: Excellent Windows integration, good optimization, best debugging experience

Choose based on your platform and development workflow preferences.

