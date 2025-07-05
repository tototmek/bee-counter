#ifndef GATE_H
#define GATE_H

#include <cstdint>

namespace bee_counter {

constexpr uint8_t kNumGates = 4;

typedef struct GateConfig gate_config_t;
typedef struct GateReading gate_reading_t;

struct GateConfig {
    uint8_t chargePin;
    uint8_t measurePinL;
    uint8_t measurePinR;
    bool invertDirection;
};

struct GateReading {
    uint8_t gateId;
    int32_t timeRawL;
    int32_t timeRawR;
};

class Gate {
  private:
    static constexpr uint32_t kChargeTimeout_ = 0xffff;
    const gate_config_t config_;

  public:
    Gate(gate_config_t config);
    void initialize();
    gate_reading_t measure();
    gate_reading_t measureSeparately();
};

} // namespace bee_counter

#endif // GATE_H