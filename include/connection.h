#ifndef CONNECTION_H
#define CONNECTION_H

#include "gate.h"

#define MODE_PRIMARY 0
#define MODE_SECONDARY 1
#define NO_MODE 0xff

namespace {
    static uint8_t cached_mode = NO_MODE;
}

namespace bee_counter::connection {

constexpr int kBaudrate = 115200;


class Transmitter{
  public:
    uint8_t pin_;

    Transmitter(uint8_t pin) : pin_(pin) {};
    void initialize();
    void transmitReading(const gate_reading_t* reading);
};

class Receiver{
  public:
    uint8_t pin_;

    Receiver(uint8_t pin) : pin_(pin) {};
    void initialize();
    bool receiveReading(gate_reading_t* reading, uint32_t timeoutMs);
};

void setModePersistent(uint8_t mode);
uint8_t getMode();

} // namespace bee_counter::connection

#endif // CONNECTION_H