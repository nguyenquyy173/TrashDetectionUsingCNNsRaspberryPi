# -*- coding: cp1258 -*-
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

# HC-SR04
TRIG1 = 23
ECHO1 = 24

# SRF05
TRIG2 = 5
ECHO2 = 6   # OUT không c?n dùng

GPIO.setup(TRIG1, GPIO.OUT)
GPIO.setup(ECHO1, GPIO.IN)
GPIO.setup(TRIG2, GPIO.OUT)
GPIO.setup(ECHO2, GPIO.IN)

def getDistance(TRIG, ECHO):
    GPIO.output(TRIG, False)
    time.sleep(0.0002)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start = time.time()
    timeout = start + 0.02

    # d?i echo HIGH
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return -1

    timeout = time.time() + 0.02
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return -1

    pulse_duration = pulse_end - pulse_start if pulse_end - pulse_start >= 0 else 0
    distance = pulse_duration * 17150
    return round(distance, 2)


if __name__ == "__main__":
    while True:
        d1 = getDistance(TRIG1, ECHO1)  # HC-SR04
        time.sleep(0.1)
        d2 = getDistance(TRIG2, ECHO2)  # SRF05
        time.sleep(0.1)

        print("HC-SR04:", d1, "cm  |  SRF05:", d2, "cm")
