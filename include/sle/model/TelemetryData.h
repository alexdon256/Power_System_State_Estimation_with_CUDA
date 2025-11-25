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
    
    void addMeasurement(std::unique_ptr<MeasurementModel> measurement, const std::string& deviceId = "");
    const std::vector<std::unique_ptr<MeasurementModel>>& getMeasurements() const {
        return measurements_;
    }
    
    size_t getMeasurementCount() const { return measurements_.size(); }
    
    // Remove measurement by device ID and type
    bool removeMeasurement(const std::string& deviceId, MeasurementType type);
    
    // Remove all measurements from a device
    size_t removeAllMeasurementsFromDevice(const std::string& deviceId);
    
    // Update measurement by device ID and type (returns true if found and updated)
    bool updateMeasurement(const std::string& deviceId, MeasurementType type, Real value, Real stdDev, int64_t timestamp = -1);
    
    // Get measurement vector z
    void getMeasurementVector(std::vector<Real>& z) const;
    
    // Get weight matrix R⁻¹ (diagonal)
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
    
    // Get latest timestamp
    int64_t getLatestTimestamp() const { return latestTimestamp_; }
    
    void clear();
    
private:
    // Primary storage: maintains order, enables O(measurements) sequential access
    // Used by: getMeasurements(), getMeasurementVector(), getWeightMatrix(), Jacobian building
    std::vector<std::unique_ptr<MeasurementModel>> measurements_;
    
    // Device index: enables O(1) device lookups and O(device_measurements) device-specific queries
    // Used by: updateMeasurement(), removeMeasurement(), Bus/Branch device queries
    // Note: MeasurementDevice::measurements_ stores pointers to measurements owned by measurements_
    std::unordered_map<std::string, std::unique_ptr<MeasurementDevice>> devices_;
    
    // Real-time processing support
    NetworkModel* network_;
    std::function<void()> onTopologyChange_;
    int64_t latestTimestamp_;
    
    void applyUpdate(const TelemetryUpdate& update);
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_TELEMETRYDATA_H

