/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_TELEMETRYDATA_H
#define SLE_MODEL_TELEMETRYDATA_H

#include <sle/Types.h>
#include <sle/model/MeasurementModel.h>
#include <sle/model/MeasurementDevice.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

namespace sle {
namespace model {

// Forward declaration
class NetworkModel;
class Branch;

// Real-time telemetry update structure
struct TelemetryUpdate {
    std::string deviceId;
    MeasurementType type;
    Real value;
    Real stdDev;
    BusId busId;
    BusId fromBus;
    BusId toBus;
    int64_t timestamp;
};

class TelemetryData {
public:
    TelemetryData();
    
    // Get all measurements (iterates through all devices)
    // Returns vector of pointers to measurements owned by devices
    // NOTE: Creates temporary vector - use device iteration for better performance
    std::vector<const MeasurementModel*> getMeasurements() const;
    
    // Get total measurement count (O(n) first call, O(1) cached after)
    size_t getMeasurementCount() const;
    
    // Remove measurement by device ID and type
    bool removeMeasurement(const std::string& deviceId, MeasurementType type);
    
    // Remove all measurements from a device
    size_t removeAllMeasurementsFromDevice(const std::string& deviceId);
    
    // Update measurement by device ID and type (returns true if found and updated)
    bool updateMeasurement(const std::string& deviceId, MeasurementType type, Real value, Real stdDev, int64_t timestamp = -1);
    
    // Get measurement vector z (iterates through devices)
    void getMeasurementVector(std::vector<Real>& z) const;
    
    // Get weight matrix R⁻¹ (diagonal, iterates through devices)
    void getWeightMatrix(std::vector<Real>& weights) const;
    
    // Measurement device management
    void addDevice(std::unique_ptr<MeasurementDevice> device);
    const std::unordered_map<std::string, std::unique_ptr<MeasurementDevice>>& getDevices() const {
        return devices_;
    }
    std::vector<const MeasurementDevice*> getDevicesByBus(BusId busId) const;
    std::vector<const MeasurementDevice*> getDevicesByBranch(BusId fromBus, BusId toBus) const;
    
    // Real-time update processing
    void setNetworkModel(NetworkModel* network);
    void setTopologyChangeCallback(std::function<void()> callback);
    
    // Process telemetry updates
    void updateMeasurement(const TelemetryUpdate& update);
    void addMeasurement(const TelemetryUpdate& update);
    void updateMeasurements(const std::vector<TelemetryUpdate>& updates);
    
    // Add measurement to a device by ID
    // Device must be created from topology first, then measurements added
    void addMeasurementToDevice(const std::string& deviceId, std::unique_ptr<MeasurementModel> measurement);
    
    // Add measurement directly through device pointer (more efficient)
    // Returns pointer to added measurement
    MeasurementModel* addMeasurement(MeasurementDevice* device, std::unique_ptr<MeasurementModel> measurement);
    
    // Get latest timestamp
    int64_t getLatestTimestamp() const { return latestTimestamp_; }
    
    void clear();
    
private:
    // Devices own measurements exclusively
    // Measurements are accessed through devices only
    std::unordered_map<std::string, std::unique_ptr<MeasurementDevice>> devices_;
    
    // Index maps for fast bus/branch device queries (updated on device add/remove)
    std::unordered_map<BusId, std::vector<MeasurementDevice*>> busToDevices_;
    
    // Hash function for BusId pair
    struct BusPairHash {
        std::size_t operator()(const std::pair<BusId, BusId>& p) const {
            return std::hash<BusId>{}(p.first) ^ (std::hash<BusId>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<BusId, BusId>, std::vector<MeasurementDevice*>, BusPairHash> branchToDevices_;
    mutable size_t cachedMeasurementCount_;  // Cache measurement count
    mutable bool measurementCountDirty_;
    
    // Real-time processing support
    NetworkModel* network_;
    std::function<void()> onTopologyChange_;
    int64_t latestTimestamp_;
    
    void applyUpdate(const TelemetryUpdate& update);
    void updateDeviceIndices(MeasurementDevice* device);
    void removeDeviceFromIndices(MeasurementDevice* device);
    
    // Ordered list of devices for stable iteration
    std::vector<MeasurementDevice*> orderedDevices_;
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_TELEMETRYDATA_H

