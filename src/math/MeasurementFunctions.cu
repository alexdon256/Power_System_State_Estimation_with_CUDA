/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/math/MeasurementFunctions.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/StateVector.h>
#include <sle/model/Branch.h>
#include <sle/model/TelemetryData.h>
#include <sle/cuda/CudaMemoryManager.h>
#include <sle/Types.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <unordered_map>

namespace sle {
namespace math {

using model::NetworkModel;
using model::StateVector;
using model::TelemetryData;

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
                                    std::vector<Real>& hx,
                                    bool useGPU) {
    // Implementation - optimized to compute power injections/flows once
    // If useGPU=true, relies on NetworkModel's CUDA path for heavy computations
    const auto& measurements = telemetry.getMeasurements();
    const size_t nMeas = measurements.size();
    
    // Reserve capacity to avoid reallocations
    hx.clear();
    hx.reserve(nMeas);
    hx.resize(nMeas);
    
    const auto& angles = state.getAngles();
    const auto& magnitudes = state.getMagnitudes();
    
    // Determine which computed quantities are required
    bool needInjections = false;
    bool needFlows = false;
    for (const auto& meas : measurements) {
        if (meas->getType() == MeasurementType::P_INJECTION || 
            meas->getType() == MeasurementType::Q_INJECTION) {
            needInjections = true;
        }
        if (meas->getType() == MeasurementType::P_FLOW || 
            meas->getType() == MeasurementType::Q_FLOW ||
            meas->getType() == MeasurementType::I_MAGNITUDE) {
            needFlows = true;
        }
    }
    
    // Compute injections/flows once (values stored directly in Bus/Branch objects)
    NetworkModel& modifiableNetwork = const_cast<NetworkModel&>(network);
    if (needInjections) {
        modifiableNetwork.computePowerInjections(state, useGPU);
    }
    if (needFlows) {
        modifiableNetwork.computePowerFlows(state, useGPU);
    }
    
    // Build lookup map for quick branch access (fromBus,toBus)->branch pointer
    std::unordered_map<BusId, std::unordered_map<BusId, const Branch*>> branchLookup;
    if (needFlows) {
        auto branches = network.getBranches();
        for (const auto* branch : branches) {
            if (!branch) continue;
            branchLookup[branch->getFromBus()][branch->getToBus()] = branch;
            branchLookup[branch->getToBus()][branch->getFromBus()] = branch;
        }
    }
    
    auto buses = network.getBuses();
    auto branchesList = network.getBranches();
    
    // Now evaluate each measurement (just extract from pre-computed data)
    #if defined(USE_OPENMP) && !defined(__CUDACC__)
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < nMeas; ++i) {
        const auto& meas = measurements[i];
        const BusId busId = meas->getLocation();
        const Index busIdx = network.getBusIndex(busId);
        
        switch (meas->getType()) {
            case MeasurementType::V_MAGNITUDE:
                if (busIdx >= 0 && static_cast<size_t>(busIdx) < magnitudes.size()) {
                    hx[i] = magnitudes[busIdx];
                } else {
                    hx[i] = 0.0;
                }
                break;
                
            case MeasurementType::P_INJECTION:
                if (busIdx >= 0 && static_cast<size_t>(busIdx) < buses.size() && buses[busIdx]) {
                    hx[i] = buses[busIdx]->getPInjection();
                } else {
                    hx[i] = 0.0;
                }
                break;
                
            case MeasurementType::Q_INJECTION:
                if (busIdx >= 0 && static_cast<size_t>(busIdx) < buses.size() && buses[busIdx]) {
                    hx[i] = buses[busIdx]->getQInjection();
                } else {
                    hx[i] = 0.0;
                }
                break;
                
            case MeasurementType::I_MAGNITUDE:
            case MeasurementType::P_FLOW:
            case MeasurementType::Q_FLOW: {
                BusId fromBus = meas->getFromBus();
                BusId toBus = meas->getToBus();
                auto it1 = branchLookup.find(fromBus);
                if (it1 != branchLookup.end()) {
                    auto it2 = it1->second.find(toBus);
                    if (it2 != it1->second.end() && it2->second) {
                        const Branch* branch = it2->second;
                        switch (meas->getType()) {
                            case MeasurementType::P_FLOW:
                                hx[i] = (branch->getFromBus() == fromBus && branch->getToBus() == toBus)
                                        ? branch->getPFlow()
                                        : -branch->getPFlow();
                                break;
                            case MeasurementType::Q_FLOW:
                                hx[i] = (branch->getFromBus() == fromBus && branch->getToBus() == toBus)
                                        ? branch->getQFlow()
                                        : -branch->getQFlow();
                                break;
                            case MeasurementType::I_MAGNITUDE:
                                hx[i] = branch->getIPU();
                                break;
                            default:
                                hx[i] = 0.0;
                        }
                    } else {
                        hx[i] = 0.0;
                    }
                } else {
                    hx[i] = 0.0;
                }
                break;
            }
                
            case MeasurementType::P_FLOW: {
                BusId fromBus = meas->getFromBus();
                BusId toBus = meas->getToBus();
                auto it1 = branchFlowMap.find(fromBus);
                if (it1 != branchFlowMap.end()) {
                    auto it2 = it1->second.find(toBus);
                    if (it2 != it1->second.end() && it2->second < pFlow.size()) {
                        // Get the branch to check direction
                        auto branches = network.getBranches();
                        if (it2->second < branches.size()) {
                            const auto* branch = branches[it2->second];
                            // If measurement direction matches branch direction, use flow as-is
                            // If reversed, negate the flow
                            if (branch->getFromBus() == fromBus && branch->getToBus() == toBus) {
                                hx[i] = pFlow[it2->second];
                            } else {
                                // Reverse direction: negate flow
                                hx[i] = -pFlow[it2->second];
                            }
                        } else {
                            hx[i] = 0.0;
                        }
                    } else {
                        hx[i] = 0.0;
                    }
                } else {
                    hx[i] = 0.0;
                }
                break;
            }
            
            case MeasurementType::Q_FLOW: {
                BusId fromBus = meas->getFromBus();
                BusId toBus = meas->getToBus();
                auto it1 = branchFlowMap.find(fromBus);
                if (it1 != branchFlowMap.end()) {
                    auto it2 = it1->second.find(toBus);
                    if (it2 != it1->second.end() && it2->second < qFlow.size()) {
                        // Get the branch to check direction
                        auto branches = network.getBranches();
                        if (it2->second < branches.size()) {
                            const auto* branch = branches[it2->second];
                            // If measurement direction matches branch direction, use flow as-is
                            // If reversed, negate the flow
                            if (branch->getFromBus() == fromBus && branch->getToBus() == toBus) {
                                hx[i] = qFlow[it2->second];
                            } else {
                                // Reverse direction: negate flow
                                hx[i] = -qFlow[it2->second];
                            }
                        } else {
                            hx[i] = 0.0;
                        }
                    } else {
                        hx[i] = 0.0;
                    }
                } else {
                    hx[i] = 0.0;
                }
                break;
            }
            
            default:
                hx[i] = 0.0;
        }
    }
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
    #if defined(USE_OPENMP) && !defined(__CUDACC__)
    #pragma omp parallel for simd schedule(static)
    #elif !defined(__CUDACC__)
    #pragma omp simd
    #endif
    for (size_t i = 0; i < n; ++i) {
        residual[i] = z[i] - hx[i];
    }
}

} // namespace math
} // namespace sle

