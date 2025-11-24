import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
# servo 180
SERVO_PIN_1 = 18 # GPIO pin (BCM numbering)
SERVO_PIN_2 = 19
PWM_FREQUENCY = 50 # Standard for hobby servos (50Hz = 20ms period)

GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

servo_pwm18 = GPIO.PWM(SERVO_PIN_1, PWM_FREQUENCY)
servo_pwm18.start(0) # Start with 0 duty cycle

servo_pwm19 = GPIO.PWM(SERVO_PIN_2, PWM_FREQUENCY)
servo_pwm19.start(0)

anglez1 = 0
anglez2 = 90
angley1 = 0
angley2 = 180
angle_neutral = 90

MAX_DUTY = 12.5
MIN_DUTY = 2.5

def angle_to_duty_cycle(angle):
    angle = max(0, min(180, angle))
    duty = MIN_DUTY + (angle / 180.0) * (MAX_DUTY - MIN_DUTY)
    return duty

def set_angle_servo(servo_pin, angle):
    duty = angle_to_duty_cycle(angle)
    print(f'-> Moving to {angle}')
    
    if servo_pin == 18: servo_pwm18.ChangeDutyCycle(duty)
    elif servo_pin == 19: servo_pwm19.ChangeDutyCycle(duty)
    time.sleep(0.5)
    
def turnLid(tai_che, huu_co, ko_tai_che, pin):
	chute_moved = False
	# truc z
	if tai_che in (1, 2) or huu_co in (1, 0):
		set_angle_servo(18, anglez1)
		chute_moved = True
		print('quay truc z')
	elif ko_tai_che == 1 or pin in (1,3):
		set_angle_servo(18, anglez2)
		chute_moved = True
	
	if chute_moved: 
		time.sleep(1)
		# truc y
		if tai_che in (1,2) or ko_tai_che == 1 :
			set_angle_servo(19, angley1)
			time.sleep(1)
			set_angle_servo(19, angle_neutral)
		elif pin in (1,3) or huu_co in (1, 0):
			set_angle_servo(19, angley2)
			time.sleep(1)
			set_angle_servo(19, angle_neutral)
	# return chute
	
	time.sleep(1)
	if chute_moved: 
		set_angle_servo(18, angle_neutral)
		time.sleep(1)


if __name__ == '__main__':
	set_angle_servo(18, 90)
	set_angle_servo(19, 90)
	print('turn')
	turnLid(-1, 0, -1, -1)
	turnLid(-1, -1, 1, -1)
	turnLid(2, -1, -1, -1)
	turnLid(-1, -1, -1, 3)
