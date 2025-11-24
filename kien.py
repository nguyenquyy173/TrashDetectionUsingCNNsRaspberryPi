# -*- coding: cp1258 -*-
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

# ===== C?u h́nh servo =====
SERVO_PIN_1 = 18  # Servo tr?c Z
SERVO_PIN_2 = 19  # Servo tr?c Y
PWM_FREQUENCY = 50  # 50Hz (chu k? ~20ms)

GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

servo_pwm18 = GPIO.PWM(SERVO_PIN_1, PWM_FREQUENCY)
servo_pwm19 = GPIO.PWM(SERVO_PIN_2, PWM_FREQUENCY)

servo_pwm18.start(0)  # duty = 0 (t?t xung)
servo_pwm19.start(0)

# ===== Các góc logic dùng trong h? =====
anglez1 = 0
anglez2 = 40
angley1 = 0
angley2 = 130
angle_neutral = 90

# ===== Gi?i h?n duty (có th? ch?nh l?i n?u biên b? ru?n) =====
MAX_DUTY = 12.5
MIN_DUTY = 2.5

# ===== Luu góc hi?n t?i (theo l?nh g?i ra) =====
current_angle_18 = angle_neutral
current_angle_19 = angle_neutral


def angle_to_duty_cycle(angle):
    """Chuy?n góc (0–180) sang duty PWM tuong ?ng."""
    angle = max(0, min(180, angle))
    duty = MIN_DUTY + (angle / 180.0) * (MAX_DUTY - MIN_DUTY)
    return duty


def set_angle_servo(servo_pin, angle):
    """
    Set góc tuy?t d?i cho servo:
    - G?i xung 0.4s d? nó quay t?i v? trí
    - Sau dó dua duty v? 0 d? gi?m rung
    """
    global current_angle_18, current_angle_19

    angle = max(0, min(180, angle))
    duty = angle_to_duty_cycle(angle)
    print(f"-> Servo {servo_pin}: goto {angle} deg (duty={duty:.2f})")

    if servo_pin == 18:
        servo_pwm18.ChangeDutyCycle(duty)
        current_angle_18 = angle
    elif servo_pin == 19:
        servo_pwm19.ChangeDutyCycle(duty)
        current_angle_19 = angle
    else:
        return

    # cho th?i gian d? servo quay
    time.sleep(0.4)

    # t?t xung d? gi?m rung
    if servo_pin == 18:
        servo_pwm18.ChangeDutyCycle(0)
    elif servo_pin == 19:
        servo_pwm19.ChangeDutyCycle(0)

    time.sleep(0.1)


def get_angle_servo(servo_pin):
    """Tr? v? góc hi?n t?i (góc dă ra l?nh) c?a servo."""
    if servo_pin == 18:
        return current_angle_18
    elif servo_pin == 19:
        return current_angle_19
    else:
        return None

def trashCan(x):
    if x == 1:
        set_angle_servo(19, 90)
        time.sleep(1)

        set_angle_servo(18, 70)
        time.sleep(1)

        set_angle_servo(19, 20)
        time.sleep(1)

        set_angle_servo(19, 90)
        time.sleep(1)

        set_angle_servo(18, 120)
        time.sleep(1)

    elif x == 2:
        set_angle_servo(19, 90)
        time.sleep(1)

        set_angle_servo(18, 155)
        time.sleep(1)

        set_angle_servo(19, 20)
        time.sleep(1)

        set_angle_servo(19, 90)
        time.sleep(1)

        set_angle_servo(18, 120)
        time.sleep(1)

    elif x == 3:
        set_angle_servo(19, 90)
        time.sleep(1)

        set_angle_servo(18, 170)
        time.sleep(1)

        set_angle_servo(19, 150)
        time.sleep(1)

        set_angle_servo(19, 90)
        time.sleep(1)

        set_angle_servo(18, 120)
        time.sleep(1)

    else:
        set_angle_servo(19, 90)
        time.sleep(1)

        set_angle_servo(18, 60)
        time.sleep(1)

        set_angle_servo(19, 150)
        time.sleep(1)

        set_angle_servo(19, 90)
        time.sleep(1)

        set_angle_servo(18, 120)
        time.sleep(1)

if __name__ == "__main__":
    """
    try:
        for i in range(8):
            trashCan(1)
            trashCan(3)
            trashCan(2)
            trashCan(4)

            #set_angle_servo(18, 90)
            #time.sleep(1)

    except KeyboardInterrupt:
        pass

    finally:
        servo_pwm18.stop()
        servo_pwm19.stop()
        GPIO.cleanup()
    """
    set_angle_servo(18,120)
    set_angle_servo(19,90)
