/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_IO_OUTPUTFORMATTER_H
#define SLE_IO_OUTPUTFORMATTER_H

#include <sle/Export.h>
#include <sle/interface/Results.h>
#include <sle/model/StateVector.h>
#include <string>

namespace sle {
namespace io {

class SLE_API OutputFormatter {
public:
    // Format results to string
    static std::string formatJSON(const interface::Results& results);
    static std::string formatCSV(const interface::Results& results);
    static std::string formatXML(const interface::Results& results);
    static std::string formatHumanReadable(const interface::Results& results);
    
    // Write to file
    static void writeToFile(const std::string& filepath, const interface::Results& results,
                           const std::string& format = "json");
};

} // namespace io
} // namespace sle

#endif // SLE_IO_OUTPUTFORMATTER_H

