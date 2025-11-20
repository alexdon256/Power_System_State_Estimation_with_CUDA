/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_IO_PMUDATA_H
#define SLE_IO_PMUDATA_H

#include <sle/Types.h>
#include <cstdint>
#include <vector>
#include <complex>
#include <string>

namespace sle {
namespace io {
namespace pmu {

// C37.118 Data Frame Structure
struct PMUFrame {
    uint16_t sync;           // Sync word (0xAA11)
    uint16_t frameSize;      // Frame size in bytes
    uint16_t idCode;         // PMU ID code
    uint32_t soc;            // Seconds of century
    uint32_t fracsec;        // Fractional seconds
    uint8_t stat;            // Status word
    uint16_t format;         // Data format
    uint16_t numPhasors;     // Number of phasors
    uint16_t numFreq;        // Number of frequency measurements
    uint16_t numAnalog;      // Number of analog values
    uint16_t numDigital;     // Number of digital words
    
    std::vector<Complex> phasors;      // Voltage/current phasors
    std::vector<Real> frequencies;     // Frequency measurements
    std::vector<Real> dfreq;           // Rate of change of frequency
    std::vector<Real> analog;          // Analog values
    std::vector<uint16_t> digital;     // Digital words
};

// PMU Configuration Frame
struct PMUConfig {
    uint16_t idCode;
    std::string stationName;
    uint16_t dataFormat;
    uint16_t numPhasors;
    uint16_t numFormats;
    std::vector<std::string> channelNames;
    std::vector<uint16_t> phasorTypes;  // 0=voltage, 1=current
    std::vector<uint16_t> analogTypes;
    std::vector<uint16_t> digitalTypes;
    Real nominalFreq;        // Nominal frequency (50 or 60 Hz)
    Real cfgChangeCount;     // Configuration change count
};

// PMU Measurement Data
struct PMUMeasurement {
    int64_t timestamp;       // Timestamp in microseconds
    BusId busId;             // Associated bus ID
    Complex voltagePhasor;   // Voltage phasor (magnitude, angle)
    Complex currentPhasor;   // Current phasor (if available)
    Real frequency;          // Frequency measurement
    Real dfreq;              // Rate of change of frequency
    Real stdDev;             // Measurement uncertainty
    bool valid;              // Data validity flag
};

class PMUParser {
public:
    // Parse C37.118 data frame
    static bool parseDataFrame(const uint8_t* buffer, size_t length, PMUFrame& frame);
    
    // Parse C37.118 configuration frame
    static bool parseConfigFrame(const uint8_t* buffer, size_t length, PMUConfig& config);
    
    // Convert PMU frame to measurement
    static PMUMeasurement convertToMeasurement(const PMUFrame& frame, BusId busId);
    
    // Parse from file (binary C37.118 format)
    static std::vector<PMUFrame> parseFromFile(const std::string& filepath);
    
    // Validate frame checksum
    static bool validateChecksum(const uint8_t* buffer, size_t length);
};

} // namespace pmu
} // namespace io
} // namespace sle

#endif // SLE_IO_PMUDATA_H

