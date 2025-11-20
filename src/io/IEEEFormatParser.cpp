/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#include <sle/io/IEEEFormatParser.h>
#include <sle/model/NetworkModel.h>
#include <sle/model/Bus.h>
#include <sle/model/Branch.h>
#include <sle/Types.h>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdexcept>

namespace sle {
namespace io {

std::unique_ptr<model::NetworkModel> IEEEFormatParser::parse(const std::string& filepath) {
    auto network = std::make_unique<model::NetworkModel>();
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::string line;
    std::string section;
    
    while (std::getline(file, line)) {
        // Remove comments
        size_t commentPos = line.find('/');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty()) continue;
        
        // Check for section headers
        if (line.find("BUS DATA") != std::string::npos || 
            line.find("BUS,") != std::string::npos) {
            section = "BUS";
            continue;
        } else if (line.find("BRANCH DATA") != std::string::npos ||
                   line.find("BRANCH,") != std::string::npos) {
            section = "BRANCH";
            continue;
        } else if (line.find("GENERATOR DATA") != std::string::npos ||
                   line.find("GEN,") != std::string::npos) {
            section = "GENERATOR";
            continue;
        }
        
        // Parse based on section
        std::istringstream lineStream(line);
        if (section == "BUS") {
            parseBusData(lineStream, *network);
        } else if (section == "BRANCH") {
            parseBranchData(lineStream, *network);
        } else if (section == "GENERATOR") {
            parseGeneratorData(lineStream, *network);
        }
    }
    
    return network;
}

void IEEEFormatParser::parseBusData(std::istringstream& iss, model::NetworkModel& network) {
    BusId busId;
    std::string name;
    Real baseKV, vMag, vAngle, pLoad, qLoad, pGen, qGen, gShunt, bShunt;
    int busType;
    
    if (iss >> busId >> name >> baseKV >> busType >> pLoad >> qLoad >> pGen >> qGen 
        >> gShunt >> bShunt >> vMag >> vAngle) {
        
        auto* bus = network.addBus(busId, name);
        bus->setBaseKV(baseKV);
        bus->setType(static_cast<BusType>(busType));
        bus->setLoad(pLoad, qLoad);
        bus->setGeneration(pGen, qGen);
        bus->setShunt(gShunt, bShunt);
        bus->setVoltage(vMag, vAngle);
        
        if (busType == 3) {  // Slack bus
            network.setReferenceBus(busId);
        }
    }
}

void IEEEFormatParser::parseBranchData(std::istringstream& iss, model::NetworkModel& network) {
    BranchId branchId;
    BusId fromBus, toBus;
    Real r, x, b, rateA, rateB, rateC, ratio, angle;
    int status;
    
    if (iss >> fromBus >> toBus >> r >> x >> b >> rateA >> ratio >> angle >> status) {
        branchId = network.getBranchCount() + 1;
        auto* branch = network.addBranch(branchId, fromBus, toBus);
        branch->setImpedance(r, x);
        branch->setCharging(b);
        branch->setRating(rateA);
        branch->setTapRatio(ratio);
        branch->setPhaseShift(angle);
    }
}

void IEEEFormatParser::parseGeneratorData(std::istringstream& iss, model::NetworkModel& network) {
    BusId busId;
    Real pGen, qGen, qMax, qMin, vSet, mBase;
    int status;
    
    if (iss >> busId >> pGen >> qGen >> qMax >> qMin >> vSet >> mBase >> status) {
        auto* bus = network.getBus(busId);
        if (bus) {
            bus->setGeneration(pGen, qGen);
            bus->setVoltage(vSet, bus->getVoltageAngle());
        }
    }
}

} // namespace io
} // namespace sle

