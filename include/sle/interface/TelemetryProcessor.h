/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_INTERFACE_TELEMETRYPROCESSOR_H
#define SLE_INTERFACE_TELEMETRYPROCESSOR_H

#include <sle/Export.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/Types.h>
#include <string>
#include <memory>
#include <vector>

#include <functional>

namespace sle {
namespace interface {

// Forward declaration
namespace model {
    class NetworkModel;
}

// Real-time telemetry update
struct SLE_API TelemetryUpdate {
    std::string deviceId;
    MeasurementType type;
    Real value;
    Real stdDev;
    BusId busId;
    BusId fromBus;
    BusId toBus;
    int64_t timestamp;
};

class SLE_API TelemetryProcessor {
public:
    TelemetryProcessor();
    ~TelemetryProcessor();
    
    // Set telemetry data container
    void setTelemetryData(model::TelemetryData* telemetry);
    void setNetworkModel(model::NetworkModel* network);
    void setTopologyChangeCallback(std::function<void()> callback);
    
    // Add/update measurement in real-time
    void updateMeasurement(const TelemetryUpdate& update);
    void addMeasurement(const TelemetryUpdate& update);
    // Remove all measurements from a device
    void removeAllMeasurementsFromDevice(const std::string& deviceId);
    
    // Batch updates
    void updateMeasurements(const std::vector<TelemetryUpdate>& updates);
    
    // Get latest timestamp
    int64_t getLatestTimestamp() const { return latestTimestamp_; }
    
private:
    model::TelemetryData* telemetry_;
    model::NetworkModel* network_;
    std::function<void()> onTopologyChange_;
    int64_t latestTimestamp_;
    
    void applyUpdate(const TelemetryUpdate& update);
};

} // namespace interface
} // namespace sle

#endif // SLE_INTERFACE_TELEMETRYPROCESSOR_H

