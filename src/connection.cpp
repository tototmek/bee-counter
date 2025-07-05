#include "connection.h"

#include <Arduino.h>

namespace bee_counter::connection {

void Connection::sendByte(uint8_t byte) {
    pinMode(pin_, OUTPUT);

    digitalWrite(pin_, 0); // start bit
    delayMicroseconds(kBitTimeUs);

    for (int j = 0; j < 8; ++j) {
        uint8_t bit = (byte >> j) & 1;
        digitalWrite(pin_, bit); // i*j-th data bit
        delayMicroseconds(kBitTimeUs);
    }
    digitalWrite(pin_, 1); // stop bit
    delayMicroseconds(kBitTimeUs);
}

uint8_t Connection::receiveByte(uint32_t timeoutMs) {
    pinMode(pin_, INPUT);
    while (digitalRead(pin_)){
        ;
    }
    // Received start bit
    delayMicroseconds(kBitTimeUs);
    delayMicroseconds(kBitTimeUs/2); // Wait to be in the middle of the bit

    uint8_t result = 0;

    for (int j = 0; j < 8; ++j) {
        uint8_t bit = digitalRead(pin_);
        result |= bit << j; // read the i*j-th data bit
        delayMicroseconds(kBitTimeUs);
    }
    return result;
}

void Connection::sendReading(const gate_reading_t* reading) {

    uint32_t dataLen = sizeof(gate_reading_t);
    uint8_t data[dataLen];
    memcpy(data, reading, dataLen);

    for (int i = 0; i < dataLen; ++i) {
        sendByte(data[i]);
    }
}


void Connection::receiveReading(gate_reading_t* reading, uint32_t timeoutMs) {

    uint32_t dataLen = sizeof(gate_reading_t);
    uint8_t data[dataLen] = {0};

    for (int i = 0; i < dataLen; ++i) {
        data[i] = receiveByte(timeoutMs);
    }

    memcpy(reading, data, dataLen);
}


} // namespace bee_counter::connection