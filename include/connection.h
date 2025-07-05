#ifndef CONNECTION_H
#define CONNECTION_H

#include "gate.h"

namespace bee_counter::connection {

constexpr int kBitTimeUs = 20;


class Connection{
  public:
    gate_reading_t readingBuffer_[kNumGates] = {0};
    uint8_t pin_;

    Connection(uint8_t pin) : pin_(pin) {};
    bool initialize() {return true;};
    void sendReading(const gate_reading_t* reading);
    void receiveReading(gate_reading_t* reading, uint32_t timeoutMs);

  private:
    void sendByte(uint8_t byte);
    uint8_t receiveByte(uint32_t timeoutMs);
};



} // namespace bee_counter::connection

#endif // CONNECTION_H