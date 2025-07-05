#include "connection.h"

#include <Arduino.h>
#include <EEPROM.h>

namespace bee_counter::connection {

void Transmitter::initialize() {
    Serial1.begin(kBaudrate, SERIAL_8N1, -1, 4);
}

void Transmitter::transmitReading(const gate_reading_t* reading) {
    Serial1.write(0xfe);
    Serial1.write(0xfd);
    Serial1.write((const uint8_t*)reading, sizeof(gate_reading_t));
}

void Receiver::initialize() {
    Serial1.begin(kBaudrate, SERIAL_8N1, 4, -1);
}

bool Receiver::receiveReading(gate_reading_t* reading, uint32_t timeoutMs) {
    uint8_t bytes[sizeof(gate_reading_t)];
    uint32_t timeoutTime = millis() + timeoutMs;
    uint8_t state = 0;
    int data_idx = 0;
    int b = 0;
    while (millis() < timeoutTime) {
        if (!Serial1.available()) continue;
        b = Serial1.read();
        if (b == -1) continue; // No bytes received, continue spinning
        switch (state) {
          case 0:
            if (b == 0xfe) state = 1;
            break;
          case 1:
            if (b == 0xfd) state = 2;
            else state = 0;
            break;
          case 2:
            bytes[data_idx++] = b;
            if (data_idx == sizeof(gate_reading_t)) {
                memcpy((uint8_t*)reading, bytes, sizeof(gate_reading_t));
                return true; // All data received
            }
            break;
        }
    }
    return false;
}
    

void setModePersistent(uint8_t mode) {
    EEPROM.begin(1);
    EEPROM.write(0, mode);
    EEPROM.commit();
    EEPROM.end();
}

uint8_t getMode() {
    if (cached_mode != NO_MODE) {
        return cached_mode;
    }
    EEPROM.begin(1);
    cached_mode = EEPROM.read(0);
    EEPROM.end();
    return cached_mode;
}

} //namespace bee_counter::connection