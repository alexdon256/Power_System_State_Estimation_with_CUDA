/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/io/PMUData.h>
#include <sle/model/MeasurementModel.h>
#include <sle/model/TelemetryData.h>
#include <sle/Types.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iterator>

namespace sle {
namespace io {
namespace pmu {

bool PMUParser::parseDataFrame(const uint8_t* buffer, size_t length, PMUFrame& frame) {
    if (length < 18) return false;  // Minimum frame size
    
    size_t offset = 0;
    
    // Parse header
    frame.sync = (buffer[offset] << 8) | buffer[offset + 1];
    offset += 2;
    
    if (frame.sync != 0xAA11) return false;  // Invalid sync word
    
    frame.frameSize = (buffer[offset] << 8) | buffer[offset + 1];
    offset += 2;
    
    if (frame.frameSize > length) return false;
    
    frame.idCode = (buffer[offset] << 8) | buffer[offset + 1];
    offset += 2;
    
    frame.soc = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                (buffer[offset + 2] << 8) | buffer[offset + 3];
    offset += 4;
    
    frame.fracsec = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                    (buffer[offset + 2] << 8) | buffer[offset + 3];
    offset += 4;
    
    frame.stat = buffer[offset++];
    frame.format = (buffer[offset] << 8) | buffer[offset + 1];
    offset += 2;
    
    frame.numPhasors = (buffer[offset] << 8) | buffer[offset + 1];
    offset += 2;
    
    frame.numFreq = (buffer[offset] << 8) | buffer[offset + 1];
    offset += 2;
    
    frame.numAnalog = (buffer[offset] << 8) | buffer[offset + 1];
    offset += 2;
    
    frame.numDigital = (buffer[offset] << 8) | buffer[offset + 1];
    offset += 2;
    
    // Parse phasors
    frame.phasors.clear();
    bool isPolar = (frame.format & 0x0001) == 0;  // Bit 0: 0=polar, 1=rectangular
    
    for (uint16_t i = 0; i < frame.numPhasors; ++i) {
        if (isPolar) {
            // Polar format: magnitude (4 bytes) + angle (4 bytes)
            uint32_t magInt = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                             (buffer[offset + 2] << 8) | buffer[offset + 3];
            offset += 4;
            
            int32_t angleInt = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                              (buffer[offset + 2] << 8) | buffer[offset + 3];
            offset += 4;
            
            // Use memcpy to avoid strict aliasing violation
            Real magnitude;
            std::memcpy(&magnitude, &magInt, sizeof(Real));
            Real angle = *reinterpret_cast<const int32_t*>(&angleInt) * 1e-7;  // Convert to radians
            angle = angle * M_PI / 180.0;  // Convert from degrees to radians
            
            frame.phasors.push_back(Complex(magnitude * cos(angle), magnitude * sin(angle)));
        } else {
            // Rectangular format: real (4 bytes) + imaginary (4 bytes)
            uint32_t realInt = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                              (buffer[offset + 2] << 8) | buffer[offset + 3];
            offset += 4;
            
            uint32_t imagInt = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                              (buffer[offset + 2] << 8) | buffer[offset + 3];
            offset += 4;
            
            // Use memcpy to avoid strict aliasing violation
            Real real, imag;
            std::memcpy(&real, &realInt, sizeof(Real));
            std::memcpy(&imag, &imagInt, sizeof(Real));
            
            frame.phasors.push_back(Complex(real, imag));
        }
    }
    
    // Parse frequencies
    frame.frequencies.clear();
    for (uint16_t i = 0; i < frame.numFreq; ++i) {
        uint32_t freqInt = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                          (buffer[offset + 2] << 8) | buffer[offset + 3];
        offset += 4;
        // Use memcpy to avoid strict aliasing violation
        Real freq;
        std::memcpy(&freq, &freqInt, sizeof(Real));
        frame.frequencies.push_back(freq);
    }
    
    // Parse rate of change of frequency
    frame.dfreq.clear();
    for (uint16_t i = 0; i < frame.numFreq; ++i) {
        uint32_t dfreqInt = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                           (buffer[offset + 2] << 8) | buffer[offset + 3];
        offset += 4;
        // Use memcpy to avoid strict aliasing violation
        Real dfreq;
        std::memcpy(&dfreq, &dfreqInt, sizeof(Real));
        frame.dfreq.push_back(dfreq);
    }
    
    return validateChecksum(buffer, frame.frameSize);
}

bool PMUParser::parseConfigFrame(const uint8_t* buffer, size_t length, PMUConfig& config) {
    // Simplified config frame parsing
    // Full implementation would parse all C37.118 config fields
    if (length < 20) return false;
    
    config.idCode = (buffer[2] << 8) | buffer[3];
    // Additional parsing would go here
    
    return true;
}

PMUMeasurement PMUParser::convertToMeasurement(const PMUFrame& frame, BusId busId) {
    PMUMeasurement meas;
    meas.timestamp = static_cast<int64_t>(frame.soc) * 1000000LL + 
                     static_cast<int64_t>(frame.fracsec);
    meas.busId = busId;
    meas.valid = (frame.stat & 0x01) == 0;  // Bit 0: data valid
    
    if (frame.phasors.size() > 0) {
        meas.voltagePhasor = frame.phasors[0];
    }
    if (frame.phasors.size() > 1) {
        meas.currentPhasor = frame.phasors[1];
    }
    if (frame.frequencies.size() > 0) {
        meas.frequency = frame.frequencies[0];
    }
    if (frame.dfreq.size() > 0) {
        meas.dfreq = frame.dfreq[0];
    }
    
    meas.stdDev = 0.001;  // Default PMU uncertainty (0.1%)
    
    return meas;
}

std::vector<PMUFrame> PMUParser::parseFromFile(const std::string& filepath) {
    std::vector<PMUFrame> frames;
    std::ifstream file(filepath, std::ios::binary);
    
    if (!file.is_open()) {
        return frames;
    }
    
    std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(file), {});
    
    size_t offset = 0;
    while (offset < buffer.size()) {
        PMUFrame frame;
        if (parseDataFrame(buffer.data() + offset, buffer.size() - offset, frame)) {
            frames.push_back(frame);
            offset += frame.frameSize;
        } else {
            break;
        }
    }
    
    return frames;
}

bool PMUParser::validateChecksum(const uint8_t* buffer, size_t length) {
    if (length < 2) return false;
    
    uint16_t checksum = 0;
    for (size_t i = 0; i < length - 2; ++i) {
        checksum += buffer[i];
    }
    
    uint16_t receivedChecksum = (buffer[length - 2] << 8) | buffer[length - 1];
    return (checksum & 0xFFFF) == receivedChecksum;
}

} // namespace pmu
} // namespace io
} // namespace sle
