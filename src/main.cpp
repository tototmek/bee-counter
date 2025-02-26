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

void loop() {
    for (int g; g < 4; ++g) {
        const int numMeasurements = 20;
        bee_counter::gate_reading_t measurements[numMeasurements];
        bee_counter::gate_reading_t sumMeasurements = {0, 0, 0, 0, 0};

        for (int i = 0; i < numMeasurements; ++i) {
            measurements[i] = gate[g].measure();
            sumMeasurements.timeRawL += measurements[i].timeRawL;
            sumMeasurements.timeL += measurements[i].timeL;
            sumMeasurements.TimeRawR += measurements[i].TimeRawR;
            sumMeasurements.timeR += measurements[i].timeR;
            sumMeasurements.timeDelta += measurements[i].timeDelta;
            delay(3);
        }

        bee_counter::gate_reading_t avgMeasurement;
        avgMeasurement.timeRawL = sumMeasurements.timeRawL / numMeasurements;
        avgMeasurement.timeL = sumMeasurements.timeL / numMeasurements;
        avgMeasurement.TimeRawR = sumMeasurements.TimeRawR / numMeasurements;
        avgMeasurement.timeR = sumMeasurements.timeR / numMeasurements;
        avgMeasurement.timeDelta = sumMeasurements.timeDelta / numMeasurements;

        std::string message =
            fmt::format("{} - AvgLeftRaw:{}, AvgRightRaw:{} ----", g, avgMeasurement.timeRawL, avgMeasurement.TimeRawR);
        Serial.print(message.c_str());
    }
    Serial.println();
    delay(1);
}
