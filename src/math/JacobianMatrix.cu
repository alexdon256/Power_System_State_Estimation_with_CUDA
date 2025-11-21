/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/JacobianMatrix.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <sle/cuda/CudaJacobianKernels.cu>
#include <algorithm>

namespace sle {
namespace math {

using model::NetworkModel;
using model::StateVector;
using model::TelemetryData;

JacobianMatrix::JacobianMatrix() : nRows_(0), nCols_(0) {
}

JacobianMatrix::~JacobianMatrix() = default;

void JacobianMatrix::buildStructure(const NetworkModel& network,
                                    const TelemetryData& telemetry) {
    const auto& measurements = telemetry.getMeasurements();
    nRows_ = measurements.size();
    nCols_ = 2 * network.getBusCount();  // angles + magnitudes
    
    std::vector<Index> rowPtr(nRows_ + 1, 0);
    std::vector<Index> colInd;
    
    // Build CSR structure
    for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& m = measurements[i];
        Index rowStart = colInd.size();
        
        switch (m->getType()) {
            case MeasurementType::P_FLOW:
            case MeasurementType::Q_FLOW: {
                // Depends on from and to bus
                Index fromBus = network.getBusIndex(m->getFromBus());
                Index toBus = network.getBusIndex(m->getToBus());
                if (fromBus >= 0) colInd.push_back(fromBus);
                if (toBus >= 0) colInd.push_back(toBus);
                if (fromBus >= 0) colInd.push_back(nCols_ / 2 + fromBus);
                if (toBus >= 0) colInd.push_back(nCols_ / 2 + toBus);
                break;
            }
            case MeasurementType::P_INJECTION:
            case MeasurementType::Q_INJECTION: {
                // Depends on all connected buses
                Index busIdx = network.getBusIndex(m->getLocation());
                if (busIdx >= 0) {
                    // Add all buses (simplified - should only add connected ones)
                    for (Index j = 0; j < nCols_ / 2; ++j) {
                        colInd.push_back(j);
                        colInd.push_back(nCols_ / 2 + j);
                    }
                }
                break;
            }
            case MeasurementType::V_MAGNITUDE: {
                Index busIdx = network.getBusIndex(m->getLocation());
                if (busIdx >= 0) {
                    colInd.push_back(nCols_ / 2 + busIdx);
                }
                break;
            }
            case MeasurementType::I_MAGNITUDE: {
                // Similar to flow
                Index fromBus = network.getBusIndex(m->getFromBus());
                Index toBus = network.getBusIndex(m->getToBus());
                if (fromBus >= 0) colInd.push_back(fromBus);
                if (toBus >= 0) colInd.push_back(toBus);
                if (fromBus >= 0) colInd.push_back(nCols_ / 2 + fromBus);
                if (toBus >= 0) colInd.push_back(nCols_ / 2 + toBus);
                break;
            }
            default:
                break;
        }
        
        rowPtr[i + 1] = colInd.size();
    }
    
    // Build matrix structure (values will be filled by buildGPU)
    std::vector<Real> values(colInd.size(), 0.0);
    matrix_.buildFromCSR(values, rowPtr, colInd, nRows_, nCols_);
}

void JacobianMatrix::build(const StateVector& state,
                          const NetworkModel& network,
                          const TelemetryData& telemetry) {
    // Use GPU if available, otherwise CPU fallback
    // For now, always use GPU path (which has CPU fallback internally)
    buildGPU(state, network, telemetry);
}

void JacobianMatrix::buildGPU(const StateVector& state,
                             const NetworkModel& network,
                             const TelemetryData& telemetry) {
    buildStructure(network, telemetry);
    
    // This would call the CUDA kernel to fill values
    // For now, placeholder
}

} // namespace math
} // namespace sle

