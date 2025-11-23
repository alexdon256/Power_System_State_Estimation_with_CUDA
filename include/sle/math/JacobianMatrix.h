/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_JACOBIANMATRIX_H
#define SLE_MATH_JACOBIANMATRIX_H

#include <sle/Types.h>
#include <sle/math/SparseMatrix.h>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

// Forward declaration
namespace sle {
namespace cuda {
    class CudaDataManager;
}
}

// Forward declarations
namespace sle {
namespace model {
    class StateVector;
    class NetworkModel;
    class TelemetryData;
}
}

namespace sle {
namespace math {

class JacobianMatrix {
public:
    JacobianMatrix();
    ~JacobianMatrix();
    
    // Get sparse matrix representation
    const SparseMatrix& getMatrix() const { return matrix_; }
    SparseMatrix& getMatrix() { return matrix_; }
    
    // Get dimensions
    Index getNRows() const { return nRows_; }
    Index getNCols() const { return nCols_; }
    
    // Build on GPU
    void buildGPU(const model::StateVector& state, const model::NetworkModel& network,
                  const model::TelemetryData& telemetry);
    
    // Build on GPU using shared CudaDataManager (avoids duplicate allocations)
    void buildGPU(const model::StateVector& state, const model::NetworkModel& network,
                  const model::TelemetryData& telemetry, cuda::CudaDataManager* dataManager);
    
    // Get host CSR data (for debugging/compatibility only - values copied from GPU)
    // CUDA-EXCLUSIVE: Values stay on GPU, this method copies them if needed
    void getHostCSR(std::vector<Real>& values, std::vector<Index>& rowPtr, std::vector<Index>& colInd) const;
    
    // Build Jacobian structure (called once before iterations, structure doesn't change)
    void buildStructure(const model::NetworkModel& network, const model::TelemetryData& telemetry);
    
    // CUDA-EXCLUSIVE: Build using GPU pointers (reuses GPU data from CudaDataManager)
    // stream: Optional CUDA stream for asynchronous execution
    void buildGPUFromPointers(const Real* d_v, const Real* d_theta,
                              const sle::cuda::DeviceBus* d_buses, const sle::cuda::DeviceBranch* d_branches,
                              const Index* d_branchFromBus, const Index* d_branchFromBusRowPtr,
                              const Index* d_branchToBus, const Index* d_branchToBusRowPtr,
                              const Index* d_measurementTypes, const Index* d_measurementLocations,
                              const Index* d_measurementBranches,
                              Index nBuses, Index nBranches, Index nMeasurements,
                              cudaStream_t stream = nullptr);
    
private:
    SparseMatrix matrix_;
    Index nRows_;
    Index nCols_;
    
    // Host CSR structure (values stay on GPU, only structure stored for compatibility)
    std::vector<Index> hostRowPtr_;
    std::vector<Index> hostColInd_;
    bool structureBuilt_;  // Track if structure is already built (avoids rebuilding every iteration)
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_JACOBIANMATRIX_H

