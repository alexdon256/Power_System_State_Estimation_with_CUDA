/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/MeasurementFunctions.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/TelemetryData.h>
#include <sle/cuda/CudaDataManager.h>
#include <sle/cuda/CudaPowerFlow.h>
#include <sle/cuda/CudaNetworkUtils.h>
#include <sle/Types.h>
#include <cuda_runtime.h>

namespace sle {
namespace math {

using model::NetworkModel;
using model::StateVector;
using model::TelemetryData;

struct MeasurementFunctions::Impl {
    cuda::CudaDataManager* sharedDataManager = nullptr;  // Shared from Solver (optional)
    std::unique_ptr<cuda::CudaDataManager> ownDataManager;  // Own instance if not shared
    
    // Cached measurement data (updated when measurements change)
    std::vector<Index> measurementTypes_;
    std::vector<Index> measurementLocations_;
    std::vector<Index> measurementBranches_;
    
    // Cached network data (updated when network changes)
    std::vector<cuda::DeviceBus> deviceBuses_;
    std::vector<cuda::DeviceBranch> deviceBranches_;
    std::vector<Index> branchFromBus_;
    std::vector<Index> branchFromBusRowPtr_;
    std::vector<Index> branchToBus_;
    std::vector<Index> branchToBusRowPtr_;
    
    size_t nBuses = 0;
    size_t nBranches = 0;
    size_t nMeasurements = 0;
    bool initialized = false;
    
    cuda::CudaDataManager* getDataManager() {
        return sharedDataManager ? sharedDataManager : ownDataManager.get();
    }
};

MeasurementFunctions::MeasurementFunctions() 
    : pImpl_(std::make_unique<Impl>()) {
}

MeasurementFunctions::~MeasurementFunctions() = default;

void MeasurementFunctions::setDataManager(cuda::CudaDataManager* dataManager) {
    pImpl_->sharedDataManager = dataManager;
    // Clear own manager if using shared one
    if (dataManager) {
        pImpl_->ownDataManager.reset();
    }
}

cuda::CudaDataManager* MeasurementFunctions::getDataManager() const {
    return pImpl_->getDataManager();
}

Real* MeasurementFunctions::evaluateGPU(const StateVector& state, 
                                       const NetworkModel& network,
                                       const TelemetryData& telemetry,
                                       bool reuseTopology,
                                       cudaStream_t stream,
                                       const Real* z,
                                       const Real* weights,
                                       Real* residual,
                                       Real* weightedResidual) {
    auto measurements = telemetry.getMeasurements();
    const size_t nMeas = measurements.size();
    const size_t nBuses = network.getBusCount();
    const size_t nBranches = network.getBranchCount();
    
    cuda::CudaDataManager* dataManager = pImpl_->getDataManager();
    
    // Initialize data manager if needed
    // If reuseTopology is true, we assume sizes match and initialized
    if (!dataManager || 
        (!reuseTopology && (
            pImpl_->nBuses != nBuses || 
            pImpl_->nBranches != nBranches || 
            pImpl_->nMeasurements != nMeas))) {
        
        if (!dataManager) {
            pImpl_->ownDataManager = std::make_unique<cuda::CudaDataManager>();
            dataManager = pImpl_->ownDataManager.get();
        }
        
        dataManager->initialize(
            static_cast<Index>(nBuses), 
            static_cast<Index>(nBranches), 
            static_cast<Index>(nMeas));
        
        pImpl_->nBuses = nBuses;
        pImpl_->nBranches = nBranches;
        pImpl_->nMeasurements = nMeas;
        pImpl_->initialized = true;
        
        // Force update if reinitialized
        reuseTopology = false;
    }
    
    const auto& v = state.getMagnitudes();
    const auto& theta = state.getAngles();
    dataManager->updateState(v.data(), theta.data(), static_cast<Index>(nBuses));
    
    // Update network data only if not reusing topology
    if (!reuseTopology) {
        if (pImpl_->deviceBuses_.size() != nBuses || pImpl_->deviceBranches_.size() != nBranches) {
            cuda::buildDeviceBuses(network, pImpl_->deviceBuses_);
            cuda::buildDeviceBranches(network, pImpl_->deviceBranches_);
            cuda::buildCSRAdjacencyLists(network, 
                                         pImpl_->branchFromBus_,
                                         pImpl_->branchFromBusRowPtr_,
                                         pImpl_->branchToBus_,
                                         pImpl_->branchToBusRowPtr_);
            
            dataManager->updateNetwork(
                pImpl_->deviceBuses_.data(), 
                pImpl_->deviceBranches_.data(),
                static_cast<Index>(nBuses), 
                static_cast<Index>(nBranches));
            
            dataManager->updateAdjacency(
                pImpl_->branchFromBus_.data(), 
                pImpl_->branchFromBusRowPtr_.data(),
                pImpl_->branchToBus_.data(), 
                pImpl_->branchToBusRowPtr_.data(),
                static_cast<Index>(nBuses),
                static_cast<Index>(pImpl_->branchFromBus_.size()),
                static_cast<Index>(pImpl_->branchToBus_.size()));
        }
        
        if (pImpl_->measurementTypes_.size() != nMeas) {
            pImpl_->measurementTypes_.clear();
            pImpl_->measurementLocations_.clear();
            pImpl_->measurementBranches_.clear();
            
            pImpl_->measurementTypes_.reserve(nMeas);
            pImpl_->measurementLocations_.reserve(nMeas);
            pImpl_->measurementBranches_.reserve(nMeas);
            
            for (const auto* meas : measurements) {
                if (!meas) continue;
                pImpl_->measurementTypes_.push_back(cuda::mapMeasurementTypeToIndex(meas->getType()));
                pImpl_->measurementLocations_.push_back(network.getBusIndex(meas->getLocation()));
                pImpl_->measurementBranches_.push_back(cuda::findBranchIndex(network, 
                                                                             meas->getFromBus(), 
                                                                             meas->getToBus()));
            }
            
            dataManager->updateMeasurements(
                pImpl_->measurementTypes_.data(),
                pImpl_->measurementLocations_.data(),
                pImpl_->measurementBranches_.data(),
                static_cast<Index>(nMeas));
        }
    }
    
    // OPTIMIZATION: Use stream for asynchronous execution
    sle::cuda::computeMeasurementsGPU(
        dataManager->getStateV(),
        dataManager->getStateTheta(),
        dataManager->getBuses(),
        dataManager->getBranches(),
        dataManager->getBranchFromBus(),
        dataManager->getBranchFromBusRowPtr(),
        dataManager->getBranchToBus(),
        dataManager->getBranchToBusRowPtr(),
        dataManager->getMeasurementTypes(),
        dataManager->getMeasurementLocations(),
        dataManager->getMeasurementBranches(),
        dataManager->getPInjection(),
        dataManager->getQInjection(),
        dataManager->getPFlow(),
        dataManager->getQFlow(),
        dataManager->getHx(),
        static_cast<Index>(nBuses),
        static_cast<Index>(nBranches),
        static_cast<Index>(nMeas),
        stream,
        z, weights, residual, weightedResidual);
    
    return dataManager->getHx();
}

void MeasurementFunctions::evaluate(const StateVector& state, 
                                    const NetworkModel& network,
                                    const TelemetryData& telemetry,
                                    std::vector<Real>& hx) {
    Real* d_hx = evaluateGPU(state, network, telemetry);
    const size_t nMeas = telemetry.getMeasurements().size();
    hx.resize(nMeas);
    cudaMemcpy(hx.data(), d_hx, nMeas * sizeof(Real), cudaMemcpyDeviceToHost);
}

} // namespace math
} // namespace sle

