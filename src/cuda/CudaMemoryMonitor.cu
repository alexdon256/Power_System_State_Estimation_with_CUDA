/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Memory usage monitoring utilities for CUDA operations
 */

#include <sle/cuda/CudaMemoryMonitor.h>
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace sle {
namespace cuda {

MemoryUsageStats getGPUMemoryUsage() {
    MemoryUsageStats stats;
    
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    
    if (err == cudaSuccess) {
        stats.totalGPUMemory = totalBytes;
        stats.freeGPUMemory = freeBytes;
        stats.usedGPUMemory = totalBytes - freeBytes;
        if (totalBytes > 0) {
            stats.utilizationPercent = 100.0 * (static_cast<double>(stats.usedGPUMemory) / static_cast<double>(totalBytes));
        }
    }
    
    return stats;
}

std::string formatMemoryUsage(const MemoryUsageStats& stats) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    auto formatBytes = [](size_t bytes) -> std::string {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit = 0;
        double size = static_cast<double>(bytes);
        while (size >= 1024.0 && unit < 4) {
            size /= 1024.0;
            unit++;
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return oss.str();
    };
    
    oss << "GPU Memory: " << formatBytes(stats.usedGPUMemory) 
        << " / " << formatBytes(stats.totalGPUMemory)
        << " (" << stats.utilizationPercent << "% used)";
    
    return oss.str();
}

void printMemoryUsage(const std::string& context) {
    MemoryUsageStats stats = getGPUMemoryUsage();
    std::string formatted = formatMemoryUsage(stats);
    
    if (!context.empty()) {
        std::cout << "[" << context << "] " << formatted << std::endl;
    } else {
        std::cout << formatted << std::endl;
    }
}

bool checkGPUMemoryAvailable(size_t requiredBytes, size_t* availableBytes) {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    
    if (err == cudaSuccess) {
        if (availableBytes) {
            *availableBytes = freeBytes;
        }
        return freeBytes >= requiredBytes;
    }
    
    if (availableBytes) {
        *availableBytes = 0;
    }
    return false;
}

} // namespace cuda
} // namespace sle


