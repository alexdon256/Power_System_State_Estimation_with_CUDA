/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_UTILS_LOADDISTRIBUTOR_H
#define SLE_UTILS_LOADDISTRIBUTOR_H

#include <sle/model/NetworkModel.h>
#include <sle/Types.h>
#include <vector>

namespace sle {
namespace utils {

class LoadDistributor {
public:
    // Distribute loads proportionally when measurements unavailable
    static void distributeLoadsProportional(model::NetworkModel& network,
                                           const std::vector<Real>& totalLoad);
    
    // Distribute loads based on historical patterns
    static void distributeLoadsHistorical(model::NetworkModel& network,
                                          const std::vector<Real>& historicalFactors);
};

} // namespace utils
} // namespace sle

#endif // SLE_UTILS_LOADDISTRIBUTOR_H

