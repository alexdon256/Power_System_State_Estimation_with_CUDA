# CMake Toolchain file for CUDA with unsupported compiler support
# This file is loaded before project() call, allowing flags to be set early

# Set CUDA flags for unsupported compiler versions (needed for CUDA 11.x with newer VS2022)
# Note: CUDA 12.1+ has better MSVC support and may not need this flag
# These must be set as initial values before project() processes CUDA language
if(NOT DEFINED CMAKE_CUDA_FLAGS_INIT)
    set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler" CACHE STRING "Initial CUDA compiler flags" FORCE)
elseif(NOT CMAKE_CUDA_FLAGS_INIT MATCHES "-allow-unsupported-compiler")
    set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler" CACHE STRING "Initial CUDA compiler flags" FORCE)
endif()

# Also set for regular compilation
if(NOT DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "CUDA compiler flags" FORCE)
elseif(NOT CMAKE_CUDA_FLAGS MATCHES "-allow-unsupported-compiler")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler" CACHE STRING "CUDA compiler flags" FORCE)
endif()

