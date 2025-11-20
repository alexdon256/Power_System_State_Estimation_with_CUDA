/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_INTERFACE_MODELLOADER_H
#define SLE_INTERFACE_MODELLOADER_H

#include <sle/model/NetworkModel.h>
#include <string>
#include <memory>

namespace sle {
namespace interface {

class ModelLoader {
public:
    // Load from IEEE Common Format
    static std::unique_ptr<model::NetworkModel> loadFromIEEE(const std::string& filepath);
    
    // Load from JSON format
    static std::unique_ptr<model::NetworkModel> loadFromJSON(const std::string& filepath);
    
    // Load from MATPOWER format
    static std::unique_ptr<model::NetworkModel> loadFromMATPOWER(const std::string& filepath);
    
    // Auto-detect format and load
    static std::unique_ptr<model::NetworkModel> load(const std::string& filepath);
    
private:
    static std::unique_ptr<model::NetworkModel> parseIEEE(const std::string& filepath);
    static std::unique_ptr<model::NetworkModel> parseJSON(const std::string& filepath);
    static std::unique_ptr<model::NetworkModel> parseMATPOWER(const std::string& filepath);
};

} // namespace interface
} // namespace sle

#endif // SLE_INTERFACE_MODELLOADER_H

