/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_TYPES_H
#define SLE_TYPES_H

#include <cstdint>
#include <string>
#include <vector>
#include <complex>
#include <memory>

namespace sle {

// Basic types
using Real = double;
using Complex = std::complex<Real>;
using Index = int32_t;
using BusId = int32_t;
using BranchId = int32_t;
using DeviceId = std::string;

// Bus types
enum class BusType {
    PQ = 1,      // Load bus
    PV = 2,      // Generator bus
    Slack = 3,   // Slack/reference bus
    Isolated = 4
};

// Measurement types
enum class MeasurementType {
    P_FLOW,           // Active power flow
    Q_FLOW,           // Reactive power flow
    P_INJECTION,      // Active power injection
    Q_INJECTION,      // Reactive power injection
    V_MAGNITUDE,      // Voltage magnitude
    I_MAGNITUDE,      // Current magnitude
    V_ANGLE,          // Voltage angle
    V_PHASOR,         // Voltage phasor (PMU)
    I_PHASOR,         // Current phasor (PMU)
    PSEUDO,           // Pseudo measurement (forecast)
    BREAKER_STATUS    // Circuit breaker status (1.0=Closed, 0.0=Open)
};

// Measurement status
enum class MeasurementStatus {
    VALID,
    SUSPECT,
    BAD,
    MISSING
};

// Forward declaration for device status (defined in MeasurementDevice.h)
namespace model {
    enum class DeviceStatus;
}

} // namespace sle

#endif // SLE_TYPES_H

