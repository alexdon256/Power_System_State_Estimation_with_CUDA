/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * Utility functions for converting network data to GPU device structures
 */

#ifndef SLE_CUDA_CUDANETWORKUTILS_H
#define SLE_CUDA_CUDANETWORKUTILS_H

#include <sle/Types.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations
namespace sle {
namespace model {
    class NetworkModel;
}
}

namespace sle {
namespace cuda {

// Build device bus structures from network model
void buildDeviceBuses(const model::NetworkModel& network, 
                     std::vector<DeviceBus>& deviceBuses);

// Build device branch structures from network model
void buildDeviceBranches(const model::NetworkModel& network,
                        std::vector<DeviceBranch>& deviceBranches);

// Build CSR format adjacency lists for GPU (CPU version - for backward compatibility)
void buildCSRAdjacencyLists(const model::NetworkModel& network,
                            std::vector<Index>& branchFromBus,
                            std::vector<Index>& branchFromBusRowPtr,
                            std::vector<Index>& branchToBus,
                            std::vector<Index>& branchToBusRowPtr);

// GPU-accelerated CSR adjacency list building (eliminates CPU-GPU transfer)
// Builds CSR format directly on GPU from DeviceBranch array
// Returns true on success, false on failure
bool buildCSRAdjacencyListsGPU(
    const DeviceBranch* d_branches,
    Index* d_branchFromBus, Index* d_branchToBus,
    Index* d_branchFromBusRowPtr, Index* d_branchToBusRowPtr,
    Index nBranches, Index nBuses,
    cudaStream_t stream = nullptr);

// Map MeasurementType enum to index (0-5)
Index mapMeasurementTypeToIndex(MeasurementType type);

// Find branch index by from/to bus IDs
Index findBranchIndex(const model::NetworkModel& network, 
                     BusId fromBus, BusId toBus);

} // namespace cuda
} // namespace sle

#endif // SLE_CUDA_CUDANETWORKUTILS_H

