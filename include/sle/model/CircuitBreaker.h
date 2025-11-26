/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_CIRCUITBREAKER_H
#define SLE_MODEL_CIRCUITBREAKER_H

#include <sle/Types.h>
#include <string>
#include <functional>

namespace sle {
namespace model {

// Forward declaration
class Branch;

/**
 * CircuitBreaker: Represents a circuit breaker that can open/close a branch
 * This is a separate component from measurement devices
 * 
 * When status changes, automatically updates associated branch status
 * and notifies registered callbacks for topology change detection
 */
class CircuitBreaker {
public:
    // Callback type: void(BranchId branchId, bool newStatus)
    using StatusChangeCallback = std::function<void(BranchId, bool)>;
    
    CircuitBreaker(const std::string& id, BranchId branchId, BusId fromBus, BusId toBus, const std::string& name = "");
    
    const std::string& getId() const { return id_; }
    const std::string& getName() const { return name_; }
    
    BranchId getBranchId() const { return branchId_; }
    BusId getFromBus() const { return fromBus_; }
    BusId getToBus() const { return toBus_; }
    
    // Status: true = Closed (allows flow), false = Open (blocks flow)
    // When status changes, automatically calls registered callback
    void setStatus(bool closed);
    bool getStatus() const { return status_; }
    bool isClosed() const { return status_; }
    bool isOpen() const { return !status_; }
    
    // Register callback for status change notifications
    // Callback signature: void(BranchId branchId, bool newStatus)
    void setStatusChangeCallback(StatusChangeCallback callback) { statusChangeCallback_ = callback; }

private:
    std::string id_;
    std::string name_;
    BranchId branchId_;
    BusId fromBus_;
    BusId toBus_;
    bool status_;  // true = Closed, false = Open
    StatusChangeCallback statusChangeCallback_;  // Called when status changes
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_CIRCUITBREAKER_H

