#include "gate.h"

#include <Arduino.h>

namespace bee_counter {

Gate::Gate(gate_config_t config) : config_(config) {}

void Gate::initialize() {
    pinMode(config_.chargePin, OUTPUT);
    pinMode(config_.measurePinL, INPUT);
    pinMode(config_.measurePinR, INPUT);
    gpio_set_drive_capability((gpio_num_t)config_.measurePinL, GPIO_DRIVE_CAP_3);
    gpio_set_drive_capability((gpio_num_t)config_.measurePinR, GPIO_DRIVE_CAP_3);
    gpio_set_drive_capability((gpio_num_t)config_.chargePin, GPIO_DRIVE_CAP_3);
    digitalWrite(config_.chargePin, LOW);
}

gate_reading_t Gate::measure() {
    int32_t totalCount = 0;
    int32_t counterL = 0;
    int32_t counterR = 0;
    bool lTriggered, rTriggered;
    noInterrupts();
    digitalWrite(config_.chargePin, HIGH);
    do {
        lTriggered = digitalRead(config_.measurePinL);
        rTriggered = digitalRead(config_.measurePinR);
        if (!lTriggered) {
            ++counterL;
        };
        if (!rTriggered) {
            ++counterR;
        }
        ++totalCount;
        if (totalCount >= kChargeTimeout_) {
            break;
        }
    } while (!lTriggered || !rTriggered);
    digitalWrite(config_.chargePin, LOW);
    interrupts();
    gate_reading_t output = {0};
    output.timeRawL = counterL;
    output.TimeRawR = counterR;
    output.timeL = counterL + config_.calibration.timeOffsetL;
    output.timeR = counterR + config_.calibration.timeOffsetR;
    output.timeDelta = output.timeL - output.timeR;
    if (config_.invertDirection) {
        output.timeDelta = -output.timeDelta;
    }
    return output;
}

} // namespace bee_counter