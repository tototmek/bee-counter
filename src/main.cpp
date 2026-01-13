#include "fmt/core.h"
#include <Arduino.h>

#include "gate.h"
#include "connection.h"
#include "detection.h"


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
bee_counter::Gate gate[bee_counter::kNumGates] = {gateConfig[0], gateConfig[1], gateConfig[2], gateConfig[3]};
bee_counter::gate_reading_t gate_reading[bee_counter::kNumGates*2] = {0};
int delta[bee_counter::kNumGates*2] = {0};

FsmDetector detectors[bee_counter::kNumGates*2];
int beeCount[bee_counter::kNumGates*2] = {0};

bee_counter::connection::Receiver receiver{8};
bee_counter::connection::Transmitter transmitter{8};

hw_timer_t *timer = NULL;
bool timerTick = false;
void onTimer() {timerTick = true;}

void primaryMcuLoop();
void secondaryMcuLoop();


void setup() {
    for (int i = 0; i < bee_counter::kNumGates; ++i) {
        gate[i].initialize();
    }
    if (bee_counter::connection::getMode() == MODE_PRIMARY) { // Primary MCU mode
        Serial.begin(115200);
        receiver.initialize();
    } else if (bee_counter::connection::getMode() == MODE_SECONDARY) { // Secondary MCU mode
        transmitter.initialize();
    } else {
        Serial.println("MCU Connection mode not set in the EEPROM.");
        while (true) {;}
    }

    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 10000, true);
    timerAlarmEnable(timer);
}

bee_counter::gate_reading_t single_reading;

void loop() {
    if (!timerTick) return; // Ensure further code is executed only after timer interrupt
    timerTick = false;
    if (bee_counter::connection::getMode() == MODE_PRIMARY) {
        primaryMcuLoop();
    } else {
        secondaryMcuLoop();
    }
}

void primaryMcuLoop() {
    // Pomiar bramek 0-3 i odbiór bramek 4-7
    for (int i = 0; i < bee_counter::kNumGates; ++i) {
        gate_reading[i] = gate[i].measure();
        receiver.receiveReading(&single_reading, 5);
        gate_reading[single_reading.gateId] = single_reading;
    }
    // Obliczenie sygnałów różnicowych i dostosowanie numeracji tuneli
    delta[0] = gate_reading[7].timeRawR - gate_reading[7].timeRawL;
    delta[1] = gate_reading[6].timeRawR - gate_reading[6].timeRawL;
    delta[2] = gate_reading[5].timeRawR - gate_reading[5].timeRawL;
    delta[3] = gate_reading[4].timeRawR - gate_reading[4].timeRawL;
    delta[4] = gate_reading[0].timeRawL - gate_reading[0].timeRawR;
    delta[7] = gate_reading[3].timeRawL - gate_reading[3].timeRawR;
    delta[6] = gate_reading[2].timeRawL - gate_reading[2].timeRawR;
    delta[5] = gate_reading[1].timeRawL - gate_reading[1].timeRawR;
    // Run detection algorithm
    for (int i = 0; i < bee_counter::kNumGates*2; ++i) {
        beeCount[i] += detectors[i].update(delta[i]);
    }
    // Report data to pc
    std::string message =
        fmt::format("{},{},{},{},{},{},{},{},{}\n", millis(),
                    beeCount[0], beeCount[1], beeCount[2], beeCount[3],
                    beeCount[4], beeCount[5], beeCount[6], beeCount[7]);
    Serial.print(message.c_str());
}

void secondaryMcuLoop() {
    // Measure channels 0-3 and send to primary mcu as channels 4-7
    for (int i = 0; i < bee_counter::kNumGates; ++i) {
        single_reading = gate[i].measure();
        single_reading.gateId = i+bee_counter::kNumGates;
        transmitter.transmitReading(&single_reading);
    }
}
