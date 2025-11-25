cmake_minimum_required(VERSION 3.18)

# Define project and languages
project(SLE VERSION 1.0.0 LANGUAGES CXX CUDA)

# Standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Configuration
find_package(CUDAToolkit REQUIRED)

# CUDA Architecture Configuration
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "75" CACHE STRING "CUDA architecture (compute capability number, e.g., 75 for sm_75)")
endif()

# Normalize CUDA_ARCH (handle "sm_XX" format)
if(CUDA_ARCH MATCHES "^sm_([0-9]+)$")
    string(REGEX REPLACE "^sm_([0-9]+)$" "\\1" CUDA_ARCH "${CUDA_ARCH}")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# Build Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_CUSOLVER "Use cuSOLVER for sparse linear systems" ON)
option(USE_OPENMP "Use OpenMP for CPU parallelization" ON)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source Files
set(SOURCES
    src/model/Bus.cpp
    src/model/Branch.cpp
    src/model/NetworkModel.cpp
    src/model/MeasurementModel.cpp
    src/model/MeasurementDevice.cpp
    src/model/StateVector.cpp
    src/model/TelemetryData.cpp
    src/interface/ModelLoader.cpp
    src/interface/MeasurementLoader.cpp
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
    src/utils/LoadDistributor.cpp
    src/multiarea/MultiAreaEstimator.cpp
    src/observability/OptimalPlacement.cpp
)

set(CUDA_SOURCES
    src/math/MeasurementFunctions.cu
    src/math/JacobianMatrix.cu
    src/math/SparseMatrix.cu
    src/math/Solver.cu
    src/math/RobustEstimator.cu
    src/math/LoadFlow.cu
    src/cuda/CudaSparseOps.cu
    src/cuda/CudaSolverOps.cu
    src/cuda/CudaReduction.cu
    src/cuda/CudaKernels.cu
    src/cuda/CudaNetworkKernels.cu
)

# Library Target
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CUDA_SOURCES})

# Target Configuration
target_compile_definitions(${PROJECT_NAME} PRIVATE SLE_EXPORTS USE_CUDA)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link Libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)

if(USE_CUSOLVER)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cusolver)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUSOLVER)
endif()

# OpenMP Support
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENMP)
        
        # Add OpenMP flags to CUDA compilation if needed
        if(NOT MSVC)
            target_compile_options(${PROJECT_NAME} PRIVATE 
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
            )
        else()
             target_compile_options(${PROJECT_NAME} PRIVATE 
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>
            )
        endif()
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP not found - CPU parallelization disabled")
    endif()
endif()

# Compiler Compatibility Check and Flags
if(MSVC)
    # Check MSVC version
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.50")
        message(STATUS "Detected MSVC ${CMAKE_CXX_COMPILER_VERSION} (Visual Studio 2026)")
        # MSVC 19.50+ requires CUDA 12.1+ for best compatibility
        if(CUDAToolkit_VERSION VERSION_LESS "12.1")
            message(WARNING "MSVC 19.50+ (VS 2026) works best with CUDA 12.1+. Current CUDA version: ${CUDAToolkit_VERSION}")
            message(WARNING "Using -allow-unsupported-compiler flag. Some features may not work correctly.")
        endif()
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.40")
        message(STATUS "Detected MSVC ${CMAKE_CXX_COMPILER_VERSION} (Visual Studio 2025)")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.30")
        message(STATUS "Detected MSVC ${CMAKE_CXX_COMPILER_VERSION} (Visual Studio 2022)")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.20")
        message(STATUS "Detected MSVC ${CMAKE_CXX_COMPILER_VERSION} (Visual Studio 2019)")
    endif()
    
    # MSVC 19.40+ (VS 2025+) and 19.50+ (VS 2026+) require -allow-unsupported-compiler flag
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.40")
        message(STATUS "Enabling CUDA unsupported compiler flag for MSVC ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

# Compiler Flags
# Note: -allow-unsupported-compiler is required for MSVC 19.40+ (VS 2025+) and MSVC 19.50+ (VS 2026+)
# This includes MSVC 19.50.35718.0 and newer versions
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Thrust check (header only, usually part of Toolkit)
find_path(THRUST_INCLUDE_DIR NAMES thrust/version.h PATHS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
if(NOT THRUST_INCLUDE_DIR)
    message(WARNING "Thrust headers not found in standard CUDA paths")
endif()

# Installation
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

# Subdirectories
if(BUILD_TESTS)
    enable_testing()
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
        add_subdirectory(tests)
    endif()
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
