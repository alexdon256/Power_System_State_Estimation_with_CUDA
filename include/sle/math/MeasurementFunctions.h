/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MATH_MEASUREMENTFUNCTIONS_H
#define SLE_MATH_MEASUREMENTFUNCTIONS_H

#include <sle/Types.h>
#include <vector>
#include <memory>

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

class MeasurementFunctions {
public:
    MeasurementFunctions();
    ~MeasurementFunctions();
    
    // Set shared CudaDataManager (optional - creates own if not set)
    void setDataManager(cuda::CudaDataManager* dataManager);
    
    // CUDA-EXCLUSIVE: Evaluate measurement functions h(x) on GPU
    // Returns GPU pointer to hx (data stays on GPU)
    // reuseTopology: If true, skips re-uploading network/measurement structure (assumes unchanged)
    // stream: Optional CUDA stream for asynchronous execution
    Real* evaluateGPU(const model::StateVector& state, const model::NetworkModel& network,
                     const model::TelemetryData& telemetry, bool reuseTopology = false, 
                     cudaStream_t stream = nullptr);
    
    // Legacy: Evaluate and copy to host (for backward compatibility)
    void evaluate(const model::StateVector& state, const model::NetworkModel& network,
                  const model::TelemetryData& telemetry, std::vector<Real>& hx);
    
    // Get CudaDataManager (for sharing with JacobianMatrix)
    cuda::CudaDataManager* getDataManager() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace math
} // namespace sle

#endif // SLE_MATH_MEASUREMENTFUNCTIONS_H

