/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_IO_PMUPARSER_H
#define SLE_IO_PMUPARSER_H

#include <sle/model/TelemetryData.h>
#include <string>
#include <memory>

namespace sle {
namespace io {

class PMUParser {
public:
    // Parse PMU data in C37.118 format
    static std::unique_ptr<model::TelemetryData> parse(const std::string& filepath);
};

} // namespace io
} // namespace sle

#endif // SLE_IO_PMUPARSER_H

