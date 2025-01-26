#include "fmt/core.h"
#include <Arduino.h>

#include "gate.h"

#define CHARGE_PIN 21
#define MEASURE_L_PIN 3
#define MEASURE_R_PIN 20

bee_counter::GateConfig gateConfig{.chargePin = 21,
                                   .measurePinL = 3,
                                   .measurePinR = 20,
                                   .invertDirection = true,
                                   .calibration = {.timeOffsetL = 0, .timeOffsetR = 0}};
bee_counter::Gate gate{gateConfig};

void setup() {
    Serial.begin(115200);
    gate.initialize();
}

double timeDeltaFiltered = 0;
double filter = 0.95;
void loop() {
    bee_counter::gate_reading_t measurement = gate.measure();
    timeDeltaFiltered = filter * double(timeDeltaFiltered) + (1 - filter) * double(measurement.timeDelta);
    std::string message = fmt::format("LeftRaw:{}, Left:{}, RightRaw:{}, Right:{}, Delta:{}, DeltaFiltered:{}",
                                      measurement.timeRawL, measurement.timeL, measurement.TimeRawR, measurement.timeR,
                                      measurement.timeDelta, int32_t(timeDeltaFiltered));
    Serial.println(message.c_str());
    delay(50);
}
