cmake_minimum_required(VERSION 3.18)

# Allow unsupported compiler versions (needed for CUDA 11.x with newer VS2022)
# Note: CUDA 12.1+ has better MSVC support and may not need this flag
# Must be set before project() so it applies during compiler detection
# Set as normal variable first (for compiler identification), then cache (for later use)

# Read from environment variable first (set by CI/CD), then cache variable, then default
# Set as normal variable (not cached) so CMake uses it during compiler identification
if(DEFINED ENV{CMAKE_CUDA_FLAGS_INIT})
    set(CMAKE_CUDA_FLAGS_INIT "$ENV{CMAKE_CUDA_FLAGS_INIT}")
elseif(DEFINED CMAKE_CUDA_FLAGS_INIT)
    # Already set via -D on command line, use as-is
    set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT}")
else()
    # Set default value as normal variable
    set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
endif()

# Ensure flag is present (append if not already there)
if(NOT CMAKE_CUDA_FLAGS_INIT MATCHES "-allow-unsupported-compiler")
    set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler")
endif()

# Also set as cache variable for later use (after project() call)
set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT}" CACHE STRING "Initial CUDA compiler flags" FORCE)

# Also set for regular compilation
if(DEFINED ENV{CMAKE_CUDA_FLAGS})
    set(CMAKE_CUDA_FLAGS "$ENV{CMAKE_CUDA_FLAGS}" CACHE STRING "CUDA compiler flags" FORCE)
elseif(DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" CACHE STRING "CUDA compiler flags" FORCE)
else()
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "CUDA compiler flags" FORCE)
endif()

# Ensure flag is present in regular flags too
get_property(_cuda_flags CACHE CMAKE_CUDA_FLAGS PROPERTY VALUE)
if(NOT _cuda_flags MATCHES "-allow-unsupported-compiler")
    set(CMAKE_CUDA_FLAGS "${_cuda_flags} -allow-unsupported-compiler" CACHE STRING "CUDA compiler flags" FORCE)
endif()

project(SLE VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA configuration
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (user can override with -DCUDA_ARCH)
# CMake 3.18+ expects just the compute capability number (e.g., "75") not "sm_75"
# If user provides "sm_XX", extract just the number
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "75" CACHE STRING "CUDA architecture (compute capability number, e.g., 75 for sm_75)")
endif()

# Extract compute capability number if user provided "sm_XX" format
string(REGEX MATCH "^sm_([0-9]+)$" _match "${CUDA_ARCH}")
if(_match)
    # User provided "sm_XX" format, extract just the number
    string(REGEX REPLACE "^sm_([0-9]+)$" "\\1" CUDA_ARCH_NUM "${CUDA_ARCH}")
    set(CUDA_ARCH "${CUDA_ARCH_NUM}" CACHE STRING "CUDA architecture (compute capability number)" FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# Build options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_DOCS "Build Doxygen documentation" OFF)
option(USE_CUSOLVER "Use cuSOLVER for sparse linear systems" ON)
option(USE_OPENMP "Use OpenMP for CPU parallelization" ON)

# Include directories (for backward compatibility)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source files
set(SOURCES
    src/model/Bus.cpp
    src/model/Branch.cpp
    src/model/NetworkModel.cpp
    src/model/MeasurementModel.cpp
    src/model/StateVector.cpp
    src/model/TelemetryData.cpp
    src/interface/ModelLoader.cpp
    src/interface/MeasurementLoader.cpp
    src/interface/TelemetryProcessor.cpp
    src/interface/StateEstimator.cpp
    src/interface/Results.cpp
    src/io/IEEEFormatParser.cpp
    src/io/JSONParser.cpp
    src/io/SCADAParser.cpp
    src/io/PMUParser.cpp
    src/io/OutputFormatter.cpp
    src/io/ComparisonReport.cpp
    src/observability/ObservabilityAnalyzer.cpp
    src/baddata/BadDataDetector.cpp
    src/baddata/DataConsistencyChecker.cpp
    src/measurements/VirtualMeasurementGenerator.cpp
    src/measurements/PseudoMeasurementGenerator.cpp
    src/utils/LoadDistributor.cpp
    src/utils/EstimationComparator.cpp
    src/multiarea/MultiAreaEstimator.cpp
    src/observability/OptimalPlacement.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/math/MeasurementFunctions.cu
    src/math/JacobianMatrix.cu
    src/math/SparseMatrix.cu
    src/math/Solver.cu
    src/math/RobustEstimator.cu
    src/math/CuSOLVERIntegration.cu
    src/math/LoadFlow.cu
    src/cuda/CudaSparseOps.cu
    src/cuda/CudaMemoryMonitor.cu
    src/cuda/CudaSolverOps.cu
    src/cuda/CudaReduction.cu
    src/cuda/CudaKernels.cu
)

# Create library as shared (DLL)
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CUDA_SOURCES})

# Define export macro when building the DLL
target_compile_definitions(${PROJECT_NAME} PRIVATE SLE_EXPORTS)

# Define USE_CUDA macro for compile-time conditional compilation
# This allows code to check if CUDA support is available at compile time
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)

# Set include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Note: Executables are in examples/ directory

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)

if(USE_CUSOLVER)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cusolver)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUSOLVER)
endif()

# OpenMP support for CPU parallelization
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_options(${PROJECT_NAME} PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
        )
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENMP)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found - CPU parallelization disabled")
        message(WARNING "Install OpenMP: sudo apt-get install libomp-dev (Linux) or brew install libomp (macOS)")
    endif()
endif()

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>  # Enable optimizations
)

# Install
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

# Tests
if(BUILD_TESTS)
    enable_testing()
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/tests AND EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
        add_subdirectory(tests)
    else()
        message(STATUS "Tests directory or CMakeLists.txt not found - skipping tests")
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        # Update Doxyfile paths for build directory
        set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/Doxyfile)
        set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)
        
        # Copy Doxyfile to build directory
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} COPYONLY)
        
        # Update OUTPUT_DIRECTORY in the copied Doxyfile
        file(READ ${DOXYGEN_OUT} DOXYFILE_CONTENT)
        string(REPLACE "OUTPUT_DIRECTORY       = docs/doxygen" 
                      "OUTPUT_DIRECTORY       = ${CMAKE_BINARY_DIR}/docs/doxygen" 
                      DOXYFILE_CONTENT "${DOXYFILE_CONTENT}")
        file(WRITE ${DOXYGEN_OUT} "${DOXYFILE_CONTENT}")
        
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        message(STATUS "Doxygen documentation target 'docs' available")
        message(STATUS "Build documentation with: cmake --build . --target docs")
    else()
        message(WARNING "Doxygen not found. Install Doxygen to build documentation.")
    endif()
endif()

# Set library output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

