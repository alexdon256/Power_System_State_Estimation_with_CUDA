cmake_minimum_required(VERSION 3.18)

# Allow unsupported compiler versions (needed for newer VS2022 with CUDA 11.8)
# Must be set before project() so it applies during compiler detection
# This is set via -DCMAKE_CUDA_FLAGS in the workflow, but ensure it's set here too
if(CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler" CACHE STRING "CUDA compiler flags" FORCE)
else()
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "CUDA compiler flags" FORCE)
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
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "sm_75" CACHE STRING "CUDA architecture")
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
    src/model/NetworkModel.cpp
    src/model/MeasurementModel.cpp
    src/model/DeviceModel.cpp
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
    src/utils/Logger.cpp
    src/utils/Config.cpp
    src/utils/Validation.cpp
    src/utils/LoadDistributor.cpp
    src/utils/EstimationComparator.cpp
    src/cuda/CudaMemoryManager.cpp
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
    src/cuda/CudaMeasurementKernels.cu
    src/cuda/CudaJacobianKernels.cu
    src/cuda/CudaVectorOps.cu
    src/cuda/CudaWeightedOps.cu
)

# Create library
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${CUDA_SOURCES})

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
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
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

# Set library output
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

