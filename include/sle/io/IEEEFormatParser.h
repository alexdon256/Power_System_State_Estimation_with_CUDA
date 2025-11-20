/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_IO_IEEEFORMATPARSER_H
#define SLE_IO_IEEEFORMATPARSER_H

#include <sle/model/NetworkModel.h>
#include <string>
#include <memory>

namespace sle {
namespace io {

class IEEEFormatParser {
public:
    static std::unique_ptr<model::NetworkModel> parse(const std::string& filepath);
    
private:
    static void parseBusData(std::istringstream& is, model::NetworkModel& network);
    static void parseBranchData(std::istringstream& is, model::NetworkModel& network);
    static void parseGeneratorData(std::istringstream& is, model::NetworkModel& network);
};

} // namespace io
} // namespace sle

#endif // SLE_IO_IEEEFORMATPARSER_H

