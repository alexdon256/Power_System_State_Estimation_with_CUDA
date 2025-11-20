/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_IO_JSONPARSER_H
#define SLE_IO_JSONPARSER_H

#include <sle/model/NetworkModel.h>
#include <string>
#include <memory>

namespace sle {
namespace io {

class JSONParser {
public:
    static std::unique_ptr<model::NetworkModel> parseNetwork(const std::string& filepath);
    // Add other parse methods as needed
};

} // namespace io
} // namespace sle

#endif // SLE_IO_JSONPARSER_H

