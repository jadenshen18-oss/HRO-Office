#include <Arduino_RouterBridge.h>

void set_led_state(bool state);

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    // Start with the LED OFF (HIGH state of the pin, since it is active-low)
    digitalWrite(LED_BUILTIN, HIGH);

    Bridge.begin();
    Bridge.provide("set_led_state", set_led_state);
}

void loop() {}

void set_led_state(bool state) {
    // LOW state means LED is ON (active-low)
    digitalWrite(LED_BUILTIN, state ? LOW : HIGH);
}
