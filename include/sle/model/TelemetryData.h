/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_TELEMETRYDATA_H
#define SLE_MODEL_TELEMETRYDATA_H

#include <sle/Types.h>
#include <sle/model/MeasurementModel.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace sle {
namespace model {

class TelemetryData {
public:
    TelemetryData();
    
    void addMeasurement(std::unique_ptr<MeasurementModel> measurement);
    const std::vector<std::unique_ptr<MeasurementModel>>& getMeasurements() const {
        return measurements_;
    }
    
    size_t getMeasurementCount() const { return measurements_.size(); }
    
    // Filter measurements by type
    std::vector<const MeasurementModel*> getMeasurementsByType(MeasurementType type) const;
    
    // Filter measurements by bus
    std::vector<const MeasurementModel*> getMeasurementsByBus(BusId busId) const;
    
    // Filter measurements by branch
    std::vector<const MeasurementModel*> getMeasurementsByBranch(BusId fromBus, BusId toBus) const;
    
    // Find measurement by device ID (O(1) average case)
    MeasurementModel* findMeasurementByDeviceId(const std::string& deviceId);
    const MeasurementModel* findMeasurementByDeviceId(const std::string& deviceId) const;
    
    // Remove measurement by device ID
    bool removeMeasurement(const std::string& deviceId);
    
    // Update measurement by device ID (returns true if found and updated)
    bool updateMeasurement(const std::string& deviceId, Real value, Real stdDev, int64_t timestamp = -1);
    
    // Get measurement vector z
    void getMeasurementVector(std::vector<Real>& z) const;
    
    // Get weight matrix R⁻¹ (diagonal)
    void getWeightMatrix(std::vector<Real>& weights) const;
    
    void clear();
    
private:
    
    std::vector<std::unique_ptr<MeasurementModel>> measurements_;
    std::unordered_map<std::string, size_t> deviceIdIndex_;  // Device ID -> index mapping for O(1) lookup
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_TELEMETRYDATA_H

