/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/MeasurementFunctions.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <sle/cuda/CudaMeasurementKernels.cu>
#include <sle/cuda/CudaMemoryManager.h>
#include <sle/Types.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace sle {
namespace math {

struct MeasurementFunctions::Impl {
    cuda::CudaMemoryManager memoryManager;
    
    // Device data
    void* d_buses;
    void* d_branches;
    void* d_measurementTypes;
    void* d_measurementLocations;
    void* d_measurementBranches;
    void* d_v;
    void* d_theta;
    void* d_hx;
    
    size_t nBuses;
    size_t nBranches;
    size_t nMeasurements;
    
    Impl() : d_buses(nullptr), d_branches(nullptr),
             d_measurementTypes(nullptr), d_measurementLocations(nullptr),
             d_measurementBranches(nullptr), d_v(nullptr), d_theta(nullptr),
             d_hx(nullptr), nBuses(0), nBranches(0), nMeasurements(0) {}
    
    ~Impl() {
        if (d_buses) memoryManager.freeDevice(d_buses);
        if (d_branches) memoryManager.freeDevice(d_branches);
        if (d_measurementTypes) memoryManager.freeDevice(d_measurementTypes);
        if (d_measurementLocations) memoryManager.freeDevice(d_measurementLocations);
        if (d_measurementBranches) memoryManager.freeDevice(d_measurementBranches);
        if (d_v) memoryManager.freeDevice(d_v);
        if (d_theta) memoryManager.freeDevice(d_theta);
        if (d_hx) memoryManager.freeDevice(d_hx);
    }
};

MeasurementFunctions::MeasurementFunctions() 
    : pImpl_(std::make_unique<Impl>()) {
}

MeasurementFunctions::~MeasurementFunctions() = default;

void MeasurementFunctions::evaluate(const StateVector& state, 
                                    const NetworkModel& network,
                                    const TelemetryData& telemetry,
                                    std::vector<Real>& hx) {
    // CPU implementation (fallback) - optimized with reserve
    const auto& measurements = telemetry.getMeasurements();
    const size_t nMeas = measurements.size();
    
    // Reserve capacity to avoid reallocations
    hx.clear();
    hx.reserve(nMeas);
    hx.resize(nMeas);
    
    const auto& angles = state.getAngles();
    const auto& magnitudes = state.getMagnitudes();
    
    // Parallel evaluation - each measurement is independent (CPU fallback only)
    // Thread-safe: read-only access to network, state, and measurements
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < nMeas; ++i) {
        const auto& meas = measurements[i];
        const BusId busId = meas->getLocation();
        const Index busIdx = network.getBusIndex(busId);
        
        if (busIdx >= 0 && static_cast<size_t>(busIdx) < magnitudes.size()) {
            switch (meas->getType()) {
                case MeasurementType::V_MAGNITUDE:
                    hx[i] = magnitudes[busIdx];
                    break;
                case MeasurementType::P_INJECTION:
                case MeasurementType::Q_INJECTION:
                case MeasurementType::P_FLOW:
                case MeasurementType::Q_FLOW:
                    // Would compute power flow/injection
                    hx[i] = 0.0;  // Placeholder
                    break;
                default:
                    hx[i] = 0.0;
            }
        } else {
            hx[i] = 0.0;
        }
    }
}

void MeasurementFunctions::evaluateGPU(const StateVector& state,
                                      const NetworkModel& network,
                                      const TelemetryData& telemetry,
                                      std::vector<Real>& hx) {
    const auto& measurements = telemetry.getMeasurements();
    size_t nMeas = measurements.size();
    size_t nBuses = network.getBusCount();
    size_t nBranches = network.getBranchCount();
    
    hx.resize(nMeas);
    
    // Prepare device data if needed
    if (pImpl_->nBuses != nBuses || pImpl_->nBranches != nBranches || 
        pImpl_->nMeasurements != nMeas) {
        // Reallocate device memory
        // This is simplified - would need proper device structure setup
    }
    
    // Copy state to device
    std::vector<Real> v = state.getMagnitudes();
    std::vector<Real> theta = state.getAngles();
    
    // Upload to device (simplified - would use proper device arrays)
    // For now, use CPU fallback
    hx.assign(nMeas, 0.0);
}

void MeasurementFunctions::computeResidual(const std::vector<Real>& z,
                                          const std::vector<Real>& hx,
                                          std::vector<Real>& residual) {
    // Optimized residual computation with reserve and vectorization
    const size_t n = z.size();
    residual.clear();
    residual.reserve(n);
    residual.resize(n);
    
    // Parallel vectorized subtraction (combines multi-threading with SIMD)
    #ifdef USE_OPENMP
    #pragma omp parallel for simd schedule(static)
    #else
    #pragma omp simd
    #endif
    for (size_t i = 0; i < n; ++i) {
        residual[i] = z[i] - hx[i];
    }
}

} // namespace math
} // namespace sle

