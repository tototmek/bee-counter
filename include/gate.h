#ifndef GATE_H
#define GATE_H

#include <cstdint>

namespace bee_counter {

typedef struct GateConfig gate_config_t;
typedef struct GateCalibration gate_calibration_t;
typedef struct GateReading gate_reading_t;

struct GateCalibration {
    int32_t resistanceL;
    int32_t timeOffsetL;
    int32_t resistanceR;
    int32_t timeOffsetR;
};

struct GateConfig {
    uint8_t chargePin;
    uint8_t measurePinL;
    uint8_t measurePinR;
    bool invertDirection;
    gate_calibration_t calibration;
};

struct GateReading {
    int32_t timeRawL;
    int32_t timeL;
    int32_t TimeRawR;
    int32_t timeR;
    int32_t timeDelta;
};

class Gate {
  private:
    static constexpr uint32_t kChargeTimeout_ = 0xffff;
    const gate_config_t config_;

  public:
    Gate(gate_config_t config);
    void initialize();
    gate_reading_t measure();
};

} // namespace bee_counter

#endif // GATE_H