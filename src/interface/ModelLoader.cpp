/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/interface/ModelLoader.h>
#include <sle/io/IEEEFormatParser.h>
#include <sle/io/JSONParser.h>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace sle {
namespace interface {

std::unique_ptr<model::NetworkModel> ModelLoader::loadFromIEEE(const std::string& filepath) {
    return parseIEEE(filepath);
}

std::unique_ptr<model::NetworkModel> ModelLoader::loadFromJSON(const std::string& filepath) {
    return parseJSON(filepath);
}

std::unique_ptr<model::NetworkModel> ModelLoader::loadFromMATPOWER(const std::string& filepath) {
    return parseMATPOWER(filepath);
}

std::unique_ptr<model::NetworkModel> ModelLoader::load(const std::string& filepath) {
    // Auto-detect format based on extension
    std::string ext = filepath.substr(filepath.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "dat" || ext == "raw") {
        return loadFromIEEE(filepath);
    } else if (ext == "json") {
        return loadFromJSON(filepath);
    } else if (ext == "m" || ext == "mat") {
        return loadFromMATPOWER(filepath);
    } else {
        // Try IEEE format as default
        return loadFromIEEE(filepath);
    }
}

std::unique_ptr<model::NetworkModel> ModelLoader::parseIEEE(const std::string& filepath) {
    return sle::io::IEEEFormatParser::parse(filepath);
}

std::unique_ptr<model::NetworkModel> ModelLoader::parseJSON(const std::string& filepath) {
    return sle::io::JSONParser::parseNetwork(filepath);
}

std::unique_ptr<model::NetworkModel> ModelLoader::parseMATPOWER(const std::string& filepath) {
    // MATPOWER parser would go here
    // For now, return empty model
    return std::make_unique<model::NetworkModel>();
}

} // namespace interface
} // namespace sle

