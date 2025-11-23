/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Memory usage monitoring utilities for CUDA operations
 */

#ifndef SLE_CUDA_CUDAMEMORYMONITOR_H
#define SLE_CUDA_CUDAMEMORYMONITOR_H

#include <sle/Types.h>
#include <cuda_runtime.h>
#include <string>

namespace sle {
namespace cuda {

// Memory usage statistics
struct MemoryUsageStats {
    size_t totalGPUMemory = 0;        // Total GPU memory (bytes)
    size_t freeGPUMemory = 0;         // Free GPU memory (bytes)
    size_t usedGPUMemory = 0;         // Used GPU memory (bytes)
    size_t allocatedBuffers = 0;      // Number of allocated buffers
    double utilizationPercent = 0.0;   // GPU memory utilization percentage
};

// Get current GPU memory usage statistics
MemoryUsageStats getGPUMemoryUsage();

// Get formatted memory usage string
std::string formatMemoryUsage(const MemoryUsageStats& stats);

// Print memory usage to console
void printMemoryUsage(const std::string& context = "");

// Check if sufficient GPU memory is available
bool checkGPUMemoryAvailable(size_t requiredBytes, size_t* availableBytes = nullptr);

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDAMEMORYMONITOR_H


