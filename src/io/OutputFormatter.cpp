/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/io/OutputFormatter.h>
#include <sle/interface/Results.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace sle {
namespace io {

std::string OutputFormatter::formatJSON(const interface::Results& results) {
    return results.toJSON();
}

std::string OutputFormatter::formatCSV(const interface::Results& results) {
    return results.toCSV();
}

std::string OutputFormatter::formatXML(const interface::Results& results) {
    // Simplified XML formatter
    std::ostringstream oss;
    oss << "<?xml version=\"1.0\"?>\n";
    oss << "<StateEstimationResult>\n";
    oss << "  <Converged>" << (results.converged() ? "true" : "false") << "</Converged>\n";
    oss << "  <Iterations>" << results.getIterations() << "</Iterations>\n";
    oss << "  <FinalNorm>" << results.getFinalNorm() << "</FinalNorm>\n";
    oss << "  <ObjectiveValue>" << results.getObjectiveValue() << "</ObjectiveValue>\n";
    oss << "  <Message>" << results.getMessage() << "</Message>\n";
    oss << "</StateEstimationResult>\n";
    return oss.str();
}

std::string OutputFormatter::formatHumanReadable(const interface::Results& results) {
    return results.toString();
}

void OutputFormatter::writeToFile(const std::string& filepath, 
                                  const interface::Results& results,
                                  const std::string& format) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    
    std::string content;
    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);
    
    if (fmt == "json") {
        content = formatJSON(results);
    } else if (fmt == "csv") {
        content = formatCSV(results);
    } else if (fmt == "xml") {
        content = formatXML(results);
    } else {
        content = formatHumanReadable(results);
    }
    
    file << content;
}

} // namespace io
} // namespace sle

