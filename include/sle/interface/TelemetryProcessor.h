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
#include <atomic>
#include <thread>
#include <queue>

namespace sle {
namespace interface {

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
    
    // Add/update measurement in real-time
    void updateMeasurement(const TelemetryUpdate& update);
    void addMeasurement(const TelemetryUpdate& update);
    void removeMeasurement(const std::string& deviceId);
    
    // Batch updates
    void updateMeasurements(const std::vector<TelemetryUpdate>& updates);
    
    // Process updates from queue (for async processing)
    void processUpdateQueue();
    
    // Start/stop real-time processing thread
    void startRealTimeProcessing();
    void stopRealTimeProcessing();
    
    // Check if updates are pending
    bool hasPendingUpdates() const;
    
    // Get latest timestamp
    int64_t getLatestTimestamp() const { return latestTimestamp_.load(); }
    
private:
    model::TelemetryData* telemetry_;
    
    std::queue<TelemetryUpdate> updateQueue_;
    std::condition_variable queueCondition_;
    
    std::atomic<bool> running_;
    std::thread processingThread_;
    std::atomic<int64_t> latestTimestamp_;
    
    void processingLoop();
    void applyUpdate(const TelemetryUpdate& update);
};

} // namespace interface
} // namespace sle

#endif // SLE_INTERFACE_TELEMETRYPROCESSOR_H

