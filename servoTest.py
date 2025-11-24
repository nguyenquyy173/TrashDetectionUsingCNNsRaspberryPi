# sudo apt-get install pigpio
# sudo systemctl enable pigpiod
# sudo systemctl start pigpiod
import pigpio, time, math

SERVO_PIN_1 = 18
SERVO_PIN_2 = 19

MIN_US = 1000   # 1.0 ms
MAX_US = 2000   # 2.0 ms

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not running")

def angle_to_us(angle):
    angle = max(0, min(180, angle))
    return int(MIN_US + (angle/180.0)*(MAX_US - MIN_US))

def set_angle(pin, angle, hold=True, ramp_ms=400, steps=20):
    start = pi.get_servo_pulsewidth(pin)
    if start < 500 or start > 2500:
        start = angle_to_us(angle)  # l?n d?u
    target = angle_to_us(angle)
    for i in range(1, steps+1):
        t = i/steps
        s = 0.5 - 0.5*math.cos(math.pi*t)
        pw = int(start + (target - start)*s)
        pi.set_servo_pulsewidth(pin, pw)
        time.sleep(max(ramp_ms/1000.0/steps, 0.01))
    if not hold:
        pi.set_servo_pulsewidth(pin, 0)  # 0 = t?t xung (nh?)

# demo
try:
    set_angle(SERVO_PIN_1, 90,  hold=True)
    time.sleep(0.8)
    set_angle(SERVO_PIN_1, 150, hold=True)
    time.sleep(0.8)
    set_angle(SERVO_PIN_2, 30,  hold=True)
    time.sleep(0.8)
finally:
 
    pi.set_servo_pulsewidth(SERVO_PIN_1, 0)
    pi.set_servo_pulsewidth(SERVO_PIN_2, 0)
    pi.stop()
