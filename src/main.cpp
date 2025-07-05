#include "fmt/core.h"
#include <Arduino.h>

#include "gate.h"
#include "connection.h"


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
uint32_t measurement[bee_counter::kNumGates] = {0};

bee_counter::connection::Receiver receiver{8};
bee_counter::connection::Transmitter transmitter{8};


void setup() {
    Serial.begin(115200);

    for (int i = 0; i < bee_counter::kNumGates; ++i) {
        gate[i].initialize();
    }

    if (bee_counter::connection::getMode() == MODE_PRIMARY) {
        // Primary MCU mode
        receiver.initialize();

    } else if (bee_counter::connection::getMode() == MODE_SECONDARY) {
        // Secondary MCU mode
        transmitter.initialize();

    } else {
        Serial.println("MCU Connection mode not set in the EEPROM.");
        while (true) {;}
    }
    
    
}

bee_counter::gate_reading_t single_reading;
void loop() {
    if (bee_counter::connection::getMode() == MODE_PRIMARY) {
        // Primary MCU mode

        for (int i = 0; i < bee_counter::kNumGates; ++i) {
            gate_reading[i] = gate[i].measure();
        }
        if (receiver.receiveReading(&single_reading, 100)) {
            gate_reading[single_reading.gateId] = single_reading;
        }
        if (receiver.receiveReading(&single_reading, 100)) {
            gate_reading[single_reading.gateId] = single_reading;
        }
        if (receiver.receiveReading(&single_reading, 100)) {
            gate_reading[single_reading.gateId] = single_reading;
        }
        if (receiver.receiveReading(&single_reading, 100)) {
            gate_reading[single_reading.gateId] = single_reading;
        }
        std::string message =
            fmt::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", millis(),
                        gate_reading[0].timeRawL, gate_reading[0].timeRawR,
                        gate_reading[1].timeRawL, gate_reading[1].timeRawR,
                        gate_reading[2].timeRawL, gate_reading[2].timeRawR,
                        gate_reading[3].timeRawL, gate_reading[3].timeRawR,
                        gate_reading[4].timeRawL, gate_reading[4].timeRawR,
                        gate_reading[5].timeRawL, gate_reading[5].timeRawR,
                        gate_reading[6].timeRawL, gate_reading[6].timeRawR,
                        gate_reading[7].timeRawL, gate_reading[7].timeRawR);
        Serial.print(message.c_str());
        Serial.println();


        delay(8);

    } else {

        // Secondary MCU mode
        for (int i = 0; i < bee_counter::kNumGates; ++i) {
            single_reading = gate[i].measure();
            single_reading.gateId = i+bee_counter::kNumGates;
            transmitter.transmitReading(&single_reading);
        }

        delay(10); // TODO: use timer interrupts instead
    }
}