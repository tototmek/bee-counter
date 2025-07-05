#include "fmt/core.h"
#include <Arduino.h>

#include "gate.h"
#include "connection.h"

#define TALKER
// #define LISTENER

bee_counter::connection::Connection connection{4};

bee_counter::GateConfig gateConfig[bee_counter::kNumGates] = {{.chargePin = 9,
                                                  .measurePinL = 20,
                                                  .measurePinR = 10,
                                                  .invertDirection = false},
                                                 {.chargePin = 4,
                                                  .measurePinL = 0,
                                                  .measurePinR = 21,
                                                  .invertDirection = true},
                                                 {.chargePin = 6,
                                                  .measurePinL = 5,
                                                  .measurePinR = 7,
                                                  .invertDirection = false},
                                                 {.chargePin = 2,
                                                  .measurePinL = 3,
                                                  .measurePinR = 1,
                                                  .invertDirection = true}};
bee_counter::Gate gate[4] = {gateConfig[0], gateConfig[1], gateConfig[2], gateConfig[3]};

void setup() {
    Serial.begin(115200);
    // for (int i = 0; i < bee_counter::kNumGates; ++i) {
    //     gate[i].initialize();
    // }
    connection.initialize();
}

// bee_counter::gate_reading_t measurements[4] = {0};

bee_counter::gate_reading_t reading {
    .gateId = 0,
    .timeRawL=21,
    .timeRawR=37};

void loop() {
    // connection.receiveReading(&reading, 300);
    // std::string message = fmt::format("{}, {}, {}", reading.gateId, reading.timeRawL, reading.timeRawR);
    // Serial.print(message.c_str());
    // Serial.println();

    
    connection.sendReading(&reading);
    reading.gateId++;
    delay(15);


    // for (int i = 0; i < 4; ++i) {
    //     measurements[i] = gate[i].measure();
    // }

    // std::string message =
    //     fmt::format("{},{},{},{},{},{},{},{},{}", millis(), measurements[0].timeRawL, measurements[0].timeRawR,
    //                 measurements[1].timeRawL, measurements[1].timeRawR, measurements[2].timeRawL,
    //                 measurements[2].timeRawR, measurements[3].timeRawL, measurements[3].timeRawR);
    // Serial.print(message.c_str());
    // Serial.println();
    // while (millis() % 10) {
    //     ;
    // }
}
