/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/interface/TelemetryProcessor.h>
#include <sle/model/TelemetryData.h>
#include <sle/model/MeasurementModel.h>
#include <algorithm>
#include <chrono>

namespace sle {
namespace interface {

TelemetryProcessor::TelemetryProcessor() 
    : telemetry_(nullptr), running_(false), latestTimestamp_(0) {
}

TelemetryProcessor::~TelemetryProcessor() {
    stopRealTimeProcessing();
}

void TelemetryProcessor::setTelemetryData(model::TelemetryData* telemetry) {
    std::lock_guard<std::mutex> lock(telemetryMutex_);
    telemetry_ = telemetry;
}

void TelemetryProcessor::updateMeasurement(const TelemetryUpdate& update) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    updateQueue_.push(update);
    queueCondition_.notify_one();
}

void TelemetryProcessor::addMeasurement(const TelemetryUpdate& update) {
    updateMeasurement(update);
}

void TelemetryProcessor::removeMeasurement(const std::string& deviceId) {
    std::lock_guard<std::mutex> lock(telemetryMutex_);
    if (!telemetry_) return;
    
    // Remove measurement by device ID
    // This would need TelemetryData to support removal
}

void TelemetryProcessor::updateMeasurements(const std::vector<TelemetryUpdate>& updates) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    for (const auto& update : updates) {
        updateQueue_.push(update);
    }
    queueCondition_.notify_one();
}

void TelemetryProcessor::processUpdateQueue() {
    std::unique_lock<std::mutex> lock(queueMutex_);
    
    while (!updateQueue_.empty()) {
        TelemetryUpdate update = updateQueue_.front();
        updateQueue_.pop();
        lock.unlock();
        
        applyUpdate(update);
        
        lock.lock();
    }
}

void TelemetryProcessor::applyUpdate(const TelemetryUpdate& update) {
    std::lock_guard<std::mutex> lock(telemetryMutex_);
    if (!telemetry_) return;
    
    // Find existing measurement by device ID and update, or create new
    // For now, always create new (would need device ID lookup in TelemetryData)
    auto measurement = std::make_unique<model::MeasurementModel>(
        update.type, update.value, update.stdDev, update.deviceId);
    
    if (update.busId >= 0) {
        measurement->setLocation(update.busId);
    }
    if (update.fromBus >= 0 && update.toBus >= 0) {
        measurement->setBranchLocation(update.fromBus, update.toBus);
    }
    
    measurement->setTimestamp(update.timestamp);
    latestTimestamp_.store(update.timestamp);
    
    telemetry_->addMeasurement(std::move(measurement));
}

void TelemetryProcessor::startRealTimeProcessing() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    processingThread_ = std::thread(&TelemetryProcessor::processingLoop, this);
}

void TelemetryProcessor::stopRealTimeProcessing() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    queueCondition_.notify_all();
    
    if (processingThread_.joinable()) {
        processingThread_.join();
    }
}

bool TelemetryProcessor::hasPendingUpdates() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    return !updateQueue_.empty();
}

void TelemetryProcessor::processingLoop() {
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(queueMutex_);
        
        queueCondition_.wait(lock, [this] {
            return !updateQueue_.empty() || !running_.load();
        });
        
        if (!running_.load()) {
            break;
        }
        
        while (!updateQueue_.empty()) {
            TelemetryUpdate update = updateQueue_.front();
            updateQueue_.pop();
            lock.unlock();
            
            applyUpdate(update);
            
            lock.lock();
        }
    }
}

} // namespace interface
} // namespace sle

