/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 */

#ifndef SLE_MODEL_BUS_H
#define SLE_MODEL_BUS_H

#include <sle/Types.h>
#include <string>

namespace sle {
namespace model {

class Bus {
public:
    Bus(BusId id, const std::string& name = "");
    
    BusId getId() const { return id_; }
    const std::string& getName() const { return name_; }
    
    void setType(BusType type) { type_ = type; }
    BusType getType() const { return type_; }
    
    void setBaseKV(Real baseKV) { baseKV_ = baseKV; }
    Real getBaseKV() const { return baseKV_; }
    
    void setVoltage(Real magnitude, Real angle = 0.0);
    Real getVoltageMagnitude() const { return voltageMag_; }
    Real getVoltageAngle() const { return voltageAngle_; }
    
    void setLoad(Real pLoad, Real qLoad);
    Real getPLoad() const { return pLoad_; }
    Real getQLoad() const { return qLoad_; }
    
    void setGeneration(Real pGen, Real qGen);
    Real getPGeneration() const { return pGen_; }
    Real getQGeneration() const { return qGen_; }
    
    void setShunt(Real gShunt, Real bShunt);
    Real getGShunt() const { return gShunt_; }
    Real getBShunt() const { return bShunt_; }
    
    void setVoltageLimits(Real vMin, Real vMax);
    Real getVMin() const { return vMin_; }
    Real getVMax() const { return vMax_; }
    
    bool isZeroInjection() const;
    
private:
    BusId id_;
    std::string name_;
    BusType type_;
    Real baseKV_;
    
    Real voltageMag_;
    Real voltageAngle_;
    
    Real pLoad_;
    Real qLoad_;
    Real pGen_;
    Real qGen_;
    
    Real gShunt_;
    Real bShunt_;
    
    Real vMin_;
    Real vMax_;
};

} // namespace model
} // namespace sle

#endif // SLE_MODEL_BUS_H

