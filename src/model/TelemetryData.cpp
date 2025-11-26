/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementDevice.h>
#include <sle/model/NetworkModel.h>
#include <sle/cuda/CudaUtils.h>
#include <algorithm>
#include <unordered_set>
#include <functional>
#include <cuda_runtime.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace sle {
namespace model {

TelemetryData::TelemetryData() 
    : network_(nullptr), latestTimestamp_(0), cachedMeasurementCount_(0), measurementCountDirty_(true),
      h_pinned_z_(nullptr), h_pinned_stdDev_(nullptr), h_pinned_weights_(nullptr),
      soaArraySize_(0), soaArrayCapacity_(0), soaArraysDirty_(true) {
}

std::vector<const MeasurementModel*> TelemetryData::getMeasurements() const {
    std::vector<const MeasurementModel*> result;
    
    // Pre-allocate if we have cached count
    if (!measurementCountDirty_) {
        result.reserve(cachedMeasurementCount_);
    }
    
    // Iterate through ordered devices for stable output
    for (const auto* device : orderedDevices_) {
        // Use iterator access instead of getMeasurements() to avoid vector creation
        for (auto it = device->begin(); it != device->end(); ++it) {
            result.push_back(it->get());
        }
    }
    
    return result;
}

size_t TelemetryData::getMeasurementCount() const {
    if (measurementCountDirty_) {
        cachedMeasurementCount_ = 0;
        Index currentIndex = 0;
        for (const auto* device : orderedDevices_) {
            cachedMeasurementCount_ += device->size();
            // Update global indices for stable mapping (e.g. for direct pinned buffer updates)
            for (auto it = device->begin(); it != device->end(); ++it) {
                if (*it) {
                    (*it)->setGlobalIndex(currentIndex++);
                }
            }
        }
        measurementCountDirty_ = false;
    }
    return cachedMeasurementCount_;
}

void TelemetryData::addMeasurementToDevice(const std::string& deviceId, std::unique_ptr<MeasurementModel> measurement) {
    if (!measurement || deviceId.empty()) return;
    
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end() || !deviceIt->second) {
        return;
    }
    
    // Add measurement to device (device takes ownership)
    deviceIt->second->addMeasurement(std::move(measurement));
    measurementCountDirty_ = true;
}

MeasurementModel* TelemetryData::addMeasurement(MeasurementDevice* device, std::unique_ptr<MeasurementModel> measurement) {
    if (!device || !measurement) return nullptr;
    
    MeasurementModel* meas = device->addMeasurement(std::move(measurement));
    measurementCountDirty_ = true;
    return meas;
}

MeasurementDevice* TelemetryData::getDevice(const std::string& deviceId) {
    auto it = devices_.find(deviceId);
    return (it != devices_.end()) ? it->second.get() : nullptr;
}

const MeasurementDevice* TelemetryData::getDevice(const std::string& deviceId) const {
    auto it = devices_.find(deviceId);
    return (it != devices_.end()) ? it->second.get() : nullptr;
}

void TelemetryData::getMeasurementVector(std::vector<Real>& z) const {
    z.clear();
    // OPTIMIZATION: Iterate directly through devices without creating temporary vector
    if (!measurementCountDirty_) {
        z.reserve(cachedMeasurementCount_);
    }
    
    // OPTIMIZATION: Parallelize device iteration with OpenMP for large systems
#ifdef USE_OPENMP
    // Use stable orderedDevices_ list
    // Parallelize measurement extraction
    if (orderedDevices_.size() > 100) {  // Only parallelize for large systems
        std::vector<std::vector<Real>> threadLocalZ(omp_get_max_threads());
        #pragma omp parallel for
        for (size_t i = 0; i < orderedDevices_.size(); ++i) {
            int tid = omp_get_thread_num();
            const auto* device = orderedDevices_[i];
            for (auto it = device->begin(); it != device->end(); ++it) {
                const MeasurementModel* m = it->get();
                if (m) {
                    threadLocalZ[tid].push_back(m->getValue());
                }
            }
        }
        
        // Merge thread-local results
        size_t totalSize = 0;
        for (const auto& local : threadLocalZ) {
            totalSize += local.size();
        }
        z.reserve(totalSize);
        for (const auto& local : threadLocalZ) {
            z.insert(z.end(), local.begin(), local.end());
        }
    } else {
        // Sequential for small systems
        for (const auto* device : orderedDevices_) {
            for (auto it = device->begin(); it != device->end(); ++it) {
                const MeasurementModel* m = it->get();
                if (m) {
                    z.push_back(m->getValue());
                }
            }
        }
    }
#else
    // Sequential fallback
    for (const auto* device : orderedDevices_) {
        for (auto it = device->begin(); it != device->end(); ++it) {
            const MeasurementModel* m = it->get();
            if (m) {
                z.push_back(m->getValue());
            }
        }
    }
#endif
}

void TelemetryData::getWeightMatrix(std::vector<Real>& weights) const {
    weights.clear();
    // OPTIMIZATION: Iterate directly through devices without creating temporary vector
    if (!measurementCountDirty_) {
        weights.reserve(cachedMeasurementCount_);
    }
    
    // OPTIMIZATION: Parallelize device iteration with OpenMP for large systems
#ifdef USE_OPENMP
    // Use stable orderedDevices_ list
    // Parallelize weight extraction
    if (orderedDevices_.size() > 100) {  // Only parallelize for large systems
        std::vector<std::vector<Real>> threadLocalWeights(omp_get_max_threads());
        #pragma omp parallel for
        for (size_t i = 0; i < orderedDevices_.size(); ++i) {
            int tid = omp_get_thread_num();
            const auto* device = orderedDevices_[i];
            for (auto it = device->begin(); it != device->end(); ++it) {
                const MeasurementModel* m = it->get();
                if (m) {
                    threadLocalWeights[tid].push_back(m->getWeight());
                }
            }
        }
        
        // Merge thread-local results
        size_t totalSize = 0;
        for (const auto& local : threadLocalWeights) {
            totalSize += local.size();
        }
        weights.reserve(totalSize);
        for (const auto& local : threadLocalWeights) {
            weights.insert(weights.end(), local.begin(), local.end());
        }
    } else {
        // Sequential for small systems
        for (const auto* device : orderedDevices_) {
            for (auto it = device->begin(); it != device->end(); ++it) {
                const MeasurementModel* m = it->get();
                if (m) {
                    weights.push_back(m->getWeight());
                }
            }
        }
    }
#else
    // Sequential fallback
    for (const auto* device : orderedDevices_) {
        for (auto it = device->begin(); it != device->end(); ++it) {
            const MeasurementModel* m = it->get();
            if (m) {
                weights.push_back(m->getWeight());
            }
        }
    }
#endif
}

bool TelemetryData::removeMeasurement(const std::string& deviceId, MeasurementType type) {
    if (deviceId.empty()) return false;
    
    // Find device
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end() || !deviceIt->second) {
        return false;
    }
    
    // Remove measurement from device (device owns it)
    return deviceIt->second->removeMeasurement(type);
}

size_t TelemetryData::removeAllMeasurementsFromDevice(const std::string& deviceId) {
    if (deviceId.empty()) return 0;
    
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end() || !deviceIt->second) {
        return 0;
    }
    
    // Get count before removal
    size_t count = deviceIt->second->size();
    
    // Get all measurement types and remove them
    std::vector<MeasurementType> typesToRemove;
    typesToRemove.reserve(count);
    for (auto it = deviceIt->second->begin(); it != deviceIt->second->end(); ++it) {
        if (*it) {
            typesToRemove.push_back((*it)->getType());
        }
    }
    
    // Remove all measurements
    for (MeasurementType type : typesToRemove) {
        deviceIt->second->removeMeasurement(type);
    }
    
    return count;
}

bool TelemetryData::updateMeasurement(const std::string& deviceId, MeasurementType type, Real value, Real stdDev, int64_t timestamp) {
    if (deviceId.empty()) return false;
    
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end() || !deviceIt->second) {
        return false;
    }
    
    // Get measurement from device
    MeasurementModel* m = deviceIt->second->getMeasurement(type);
    if (!m) {
        return false;
    }
    
    m->setValue(value);
    m->setStdDev(stdDev);
    if (timestamp >= 0) {
        m->setTimestamp(timestamp);
    }
    
    // OPTIMIZATION: Mark SoA arrays as dirty when measurement is updated
    markArraysDirty();
    
    return true;
}


void TelemetryData::addDevice(std::unique_ptr<MeasurementDevice> device) {
    if (!device) return;
    
    const std::string& deviceId = device->getId();
    if (deviceId.empty() || devices_.find(deviceId) != devices_.end()) {
        return;
    }
    
    MeasurementDevice* devicePtr = device.get();
    devices_[deviceId] = std::move(device);
    orderedDevices_.push_back(devicePtr);  // Maintain stable order
    updateDeviceIndices(devicePtr);
    measurementCountDirty_ = true;
    markArraysDirty();  // OPTIMIZATION: Mark arrays dirty when device added
}


std::vector<const MeasurementDevice*> TelemetryData::getDevicesByBus(BusId busId) const {
    std::vector<const MeasurementDevice*> result;
    auto it = busToDevices_.find(busId);
    if (it != busToDevices_.end()) {
        result.reserve(it->second.size());
        for (MeasurementDevice* dev : it->second) {
            result.push_back(dev);
        }
    }
    return result;
}

std::vector<const MeasurementDevice*> TelemetryData::getDevicesByBranch(BusId fromBus, BusId toBus) const {
    std::vector<const MeasurementDevice*> result;
    // Try both orderings
    auto key1 = std::make_pair(fromBus, toBus);
    auto key2 = std::make_pair(toBus, fromBus);
    
    auto it1 = branchToDevices_.find(key1);
    if (it1 != branchToDevices_.end()) {
        result.reserve(it1->second.size());
        for (MeasurementDevice* dev : it1->second) {
            result.push_back(dev);
        }
    }
    
    auto it2 = branchToDevices_.find(key2);
    if (it2 != branchToDevices_.end() && it2 != it1) {
        for (MeasurementDevice* dev : it2->second) {
            result.push_back(dev);
        }
    }
    
    return result;
}

void TelemetryData::updateDeviceIndices(MeasurementDevice* device) {
    if (!device) return;
    
    const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(device);
    if (voltmeter) {
        BusId busId = voltmeter->getBusId();
        busToDevices_[busId].push_back(device);
        // OPTIMIZATION: Add direct pointer to Bus
        if (network_) {
            Bus* bus = network_->getBus(busId);
            if (bus) bus->addAssociatedDevice(device);
        }
        return;
    }
    
    const Multimeter* multimeter = dynamic_cast<const Multimeter*>(device);
    if (multimeter) {
        auto key = std::make_pair(multimeter->getFromBus(), multimeter->getToBus());
        branchToDevices_[key].push_back(device);
        // OPTIMIZATION: Add direct pointer to Branch
        if (network_) {
            Branch* branch = network_->getBranchByBuses(multimeter->getFromBus(), multimeter->getToBus());
            if (branch) branch->addAssociatedDevice(device);
        }
    }
}

void TelemetryData::removeDeviceFromIndices(MeasurementDevice* device) {
    if (!device) return;
    
    const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(device);
    if (voltmeter) {
        BusId busId = voltmeter->getBusId();
        auto& devices = busToDevices_[busId];
        devices.erase(std::remove(devices.begin(), devices.end(), device), devices.end());
        if (devices.empty()) {
            busToDevices_.erase(busId);
        }
        // OPTIMIZATION: Remove direct pointer from Bus
        if (network_) {
            Bus* bus = network_->getBus(busId);
            if (bus) bus->removeAssociatedDevice(device);
        }
        return;
    }
    
    const Multimeter* multimeter = dynamic_cast<const Multimeter*>(device);
    if (multimeter) {
        auto key = std::make_pair(multimeter->getFromBus(), multimeter->getToBus());
        auto& devices = branchToDevices_[key];
        devices.erase(std::remove(devices.begin(), devices.end(), device), devices.end());
        if (devices.empty()) {
            branchToDevices_.erase(key);
        }
        // OPTIMIZATION: Remove direct pointer from Branch
        if (network_) {
            Branch* branch = network_->getBranchByBuses(multimeter->getFromBus(), multimeter->getToBus());
            if (branch) branch->removeAssociatedDevice(device);
        }
    }
}

void TelemetryData::setNetworkModel(NetworkModel* network) {
    network_ = network;
    // If devices already exist, link them to the network
    if (network_) {
        for (auto* device : orderedDevices_) {
            const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(device);
            if (voltmeter) {
                Bus* bus = network_->getBus(voltmeter->getBusId());
                if (bus) bus->addAssociatedDevice(device);
                continue;
            }
            const Multimeter* multimeter = dynamic_cast<const Multimeter*>(device);
            if (multimeter) {
                Branch* branch = network_->getBranchByBuses(multimeter->getFromBus(), multimeter->getToBus());
                if (branch) branch->addAssociatedDevice(device);
            }
        }
    }
}

void TelemetryData::setTopologyChangeCallback(std::function<void()> callback) {
    onTopologyChange_ = callback;
}

void TelemetryData::updateMeasurement(const TelemetryUpdate& update) {
    applyUpdate(update);
}

void TelemetryData::addMeasurement(const TelemetryUpdate& update) {
    applyUpdate(update);
}

void TelemetryData::updateMeasurements(const std::vector<TelemetryUpdate>& updates) {
    for (const auto& update : updates) {
        applyUpdate(update);
    }
}

void TelemetryData::applyUpdate(const TelemetryUpdate& update) {
    // Find device
    MeasurementDevice* device = nullptr;
    if (!update.deviceId.empty()) {
        auto it = devices_.find(update.deviceId);
        if (it != devices_.end()) {
            device = it->second.get();
        }
    }
    
    // Try updating existing measurement
    if (device && updateMeasurement(update.deviceId, update.type, update.value, update.stdDev, update.timestamp)) {
        latestTimestamp_ = update.timestamp;
        return;
    }
    
    // Create new measurement and add to device
    // Note: Location information is stored in the device, not in the measurement
    if (device) {
        auto measurement = std::make_unique<MeasurementModel>(
            update.type, update.value, update.stdDev);
        measurement->setTimestamp(update.timestamp);
        latestTimestamp_ = update.timestamp;
        
        // Add to device (device takes ownership)
        device->addMeasurement(std::move(measurement));
    }
}

// OPTIMIZATION: Build SoA arrays in pinned memory
void TelemetryData::buildSoAArrays() const {
    if (!soaArraysDirty_ && soaArraySize_ > 0) {
        return;  // Arrays are up-to-date
    }
    
    // Get current measurement count
    size_t nMeas = getMeasurementCount();
    if (nMeas == 0) {
        soaArraySize_ = 0;
        soaArraysDirty_ = false;
        return;
    }
    
    // Allocate/reallocate pinned memory if needed
    cuda::allocatePinnedBuffer(h_pinned_z_, soaArrayCapacity_, nMeas);
    cuda::allocatePinnedBuffer(h_pinned_stdDev_, soaArrayCapacity_, nMeas);
    cuda::allocatePinnedBuffer(h_pinned_weights_, soaArrayCapacity_, nMeas);
    
    if (!h_pinned_z_ || !h_pinned_stdDev_ || !h_pinned_weights_) {
        soaArraySize_ = 0;
        return;
    }
    
    // OPTIMIZATION: Parallel extraction for large systems
#ifdef USE_OPENMP
    if (orderedDevices_.size() > 100) {
        // Parallel extraction with thread-local arrays
        std::vector<std::vector<Real>> threadLocalZ(omp_get_max_threads());
        std::vector<std::vector<Real>> threadLocalStdDev(omp_get_max_threads());
        std::vector<std::vector<Real>> threadLocalWeights(omp_get_max_threads());
        
        #pragma omp parallel for
        for (size_t i = 0; i < orderedDevices_.size(); ++i) {
            int tid = omp_get_thread_num();
            const auto* device = orderedDevices_[i];
            for (auto it = device->begin(); it != device->end(); ++it) {
                const MeasurementModel* m = it->get();
                if (m) {
                    threadLocalZ[tid].push_back(m->getValue());
                    threadLocalStdDev[tid].push_back(m->getStdDev());
                    threadLocalWeights[tid].push_back(m->getWeight());  // Uses cached weight
                }
            }
        }
        
        // Merge thread-local results
        size_t offset = 0;
        for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
            const auto& localZ = threadLocalZ[tid];
            const auto& localStdDev = threadLocalStdDev[tid];
            const auto& localWeights = threadLocalWeights[tid];
            
            std::copy(localZ.begin(), localZ.end(), h_pinned_z_ + offset);
            std::copy(localStdDev.begin(), localStdDev.end(), h_pinned_stdDev_ + offset);
            std::copy(localWeights.begin(), localWeights.end(), h_pinned_weights_ + offset);
            offset += localZ.size();
        }
    } else {
#endif
        // Sequential extraction for small systems
        size_t idx = 0;
        for (const auto* device : orderedDevices_) {
            for (auto it = device->begin(); it != device->end(); ++it) {
                const MeasurementModel* m = it->get();
                if (m) {
                    h_pinned_z_[idx] = m->getValue();
                    h_pinned_stdDev_[idx] = m->getStdDev();
                    h_pinned_weights_[idx] = m->getWeight();  // Uses cached weight
                    ++idx;
                }
            }
        }
#ifdef USE_OPENMP
    }
#endif
    
    soaArraySize_ = nMeas;
    soaArraysDirty_ = false;
}

const Real* TelemetryData::getMeasurementValuesArray() const {
    buildSoAArrays();
    return h_pinned_z_;
}

const Real* TelemetryData::getStdDevArray() const {
    buildSoAArrays();
    return h_pinned_stdDev_;
}

const Real* TelemetryData::getWeightsArray() const {
    buildSoAArrays();
    return h_pinned_weights_;
}

size_t TelemetryData::getMeasurementArraySize() const {
    buildSoAArrays();
    return soaArraySize_;
}

void TelemetryData::markArraysDirty() {
    soaArraysDirty_ = true;
}

void TelemetryData::clear() {
    // Free pinned memory
    if (h_pinned_z_) {
        cudaFreeHost(h_pinned_z_);
        h_pinned_z_ = nullptr;
    }
    if (h_pinned_stdDev_) {
        cudaFreeHost(h_pinned_stdDev_);
        h_pinned_stdDev_ = nullptr;
    }
    if (h_pinned_weights_) {
        cudaFreeHost(h_pinned_weights_);
        h_pinned_weights_ = nullptr;
    }
    soaArraySize_ = 0;
    soaArrayCapacity_ = 0;
    soaArraysDirty_ = true;
    
    // Unlink from network first to avoid dangling pointers
    if (network_) {
        for (auto* device : orderedDevices_) {
            const Voltmeter* voltmeter = dynamic_cast<const Voltmeter*>(device);
            if (voltmeter) {
                Bus* bus = network_->getBus(voltmeter->getBusId());
                if (bus) bus->removeAssociatedDevice(device);
                continue;
            }
            const Multimeter* multimeter = dynamic_cast<const Multimeter*>(device);
            if (multimeter) {
                Branch* branch = network_->getBranchByBuses(multimeter->getFromBus(), multimeter->getToBus());
                if (branch) branch->removeAssociatedDevice(device);
            }
        }
    }
    
    // Clear all devices (devices own measurements, so clearing devices clears measurements)
    devices_.clear();
    orderedDevices_.clear();
    busToDevices_.clear();
    branchToDevices_.clear();
    cachedMeasurementCount_ = 0;
    measurementCountDirty_ = false;
    latestTimestamp_ = 0;
}

} // namespace model
} // namespace sle

