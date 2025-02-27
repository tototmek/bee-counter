#include "fmt/core.h"
#include <Arduino.h>

#include "gate.h"

bee_counter::GateConfig gateConfig[4] = {{.chargePin = 9,
                                          .measurePinL = 8,
                                          .measurePinR = 10,
                                          .invertDirection = false,
                                          .calibration = {.timeOffsetL = 0, .timeOffsetR = 0}},
                                         {.chargePin = 21,
                                          .measurePinL = 3,
                                          .measurePinR = 20,
                                          .invertDirection = true,
                                          .calibration = {.timeOffsetL = 0, .timeOffsetR = 0}},
                                         {.chargePin = 6,
                                          .measurePinL = 5,
                                          .measurePinR = 7,
                                          .invertDirection = false,
                                          .calibration = {.timeOffsetL = 0, .timeOffsetR = 0}},
                                         {.chargePin = 1,
                                          .measurePinL = 0,
                                          .measurePinR = 2,
                                          .invertDirection = true,
                                          .calibration = {.timeOffsetL = 0, .timeOffsetR = 0}}};
bee_counter::Gate gate[4] = {gateConfig[0], gateConfig[1], gateConfig[2], gateConfig[3]};

void setup() {
    Serial.begin(115200);
    gate[0].initialize();
    gate[1].initialize();
    gate[2].initialize();
    gate[3].initialize();
}
float prevMeasurement = 0.0f;
void loop() {

    auto measurements = gate[2].measureSeparately();

    // if (!millis() % 100) {
        std::string message = fmt::format("{},{},{}", millis(), measurements.timeRawL, measurements.TimeRawR);
        Serial.print(message.c_str());
        Serial.println();
    // }
    while (millis() % 10) {
        ;
    }
}
