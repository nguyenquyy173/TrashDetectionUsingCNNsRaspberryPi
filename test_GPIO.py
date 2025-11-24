# -*- coding: cp1258 -*-
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

TEST_PIN = 24   # d?i thành chân b?n mu?n test

GPIO.setup(TEST_PIN, GPIO.OUT)

print(f"Testing GPIO {TEST_PIN} ...")

try:
    while True:
        # Set HIGH
        GPIO.output(TEST_PIN, GPIO.HIGH)
        time.sleep(0.1)
        val_high = GPIO.input(TEST_PIN)
        print(f"Set HIGH  -> Read: {val_high}")

        # Set LOW
        GPIO.output(TEST_PIN, GPIO.LOW)
        time.sleep(0.1)
        val_low = GPIO.input(TEST_PIN)
        print(f"Set LOW   -> Read: {val_low}")

        print("--------------------")
        time.sleep(0.5)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Stopped.")
