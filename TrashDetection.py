import cv2
import kien
import deployModel
import os
import tensorflow as tf
import screen
import time
import ultraSonic
import shutil
import numpy as np
import RPi.GPIO as GPIO

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

model_path = "/home/pi/DoAn/trash_16_cls_grayscale_finetuned_noLambda.keras"
model = tf.keras.models.load_model(model_path)

img_path = 'test.jpg'
img_current_path = 'current.jpg'
img_pre_path = 'previous.jpg'
CAPTURE_INTERVAL = 2
check_object_change = False
DEFAULT_PIXEL_DIFF_THRESHOLD = 40
DEFAULT_MAJOR_CHANGE_PERCENT = 5

distance = 20

pic_HMI = {'banana':2, 'battery':13, 'bimbim':14, 'coc':7, 'cucumberpeel':4,'egg':5, 'hopxop':12, 'leaves':11, 'lotCoc':8, 'mask':10, 'metal':15, 'orange':3,
				'paper':9, 'pen':6, 'bottle':0, 'bag':1}
				
label_HMI = {'banana':2, 'battery':3, 'bimbim':1, 'coc':1, 'cucumberpeel':2,'egg':2, 'hopxop':1, 'leaves':2, 'lotCoc':1, 'mask':1, 'metal':0, 'orange':2,
				'paper':0, 'pen':1, 'bottle':0, 'bag':1, 'tai_che':0, 'huu_co':2, 'ko_tai_che':1, 'pin':3}
		
def set_up():
	screen.AddressLW(10,4)
	screen.AddressLW(8,2)
	screen.AddressWriteLB()
	kien.set_angle_servo(18, 120)
	kien.set_angle_servo(19, 90)
	
def is_object_changed(
    reference_path: str, 
    current_path: str, 
    change_threshold_percent: float = DEFAULT_MAJOR_CHANGE_PERCENT,
    pixel_diff_threshold: int = DEFAULT_PIXEL_DIFF_THRESHOLD
) -> bool:
	reference_image = cv2.imread(reference_path)
	reference_image = cv2.resize(reference_image, (240, 240), interpolation=cv2.INTER_NEAREST)
	
	current_image = cv2.imread(current_path)
	current_image = cv2.resize(current_image, (240, 240), interpolation=cv2.INTER_NEAREST)
	
	ref_shape = reference_image.shape[:2]
	curr_shape = current_image.shape[:2]
	
	if ref_shape != curr_shape:
		print(f"Error: Image sizes must match. Ref: {ref_shape}, Curr: {curr_shape}")
		return False
	
	total_area = ref_shape[0] * ref_shape[1]
	if len(reference_image.shape) > 2:
		ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
		curr_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
	else: 
		ref_gray = reference_image
		curr_gray = current_image
	
	ref_blur = cv2.GaussianBlur(ref_gray, (3, 3), 0)
	curr_blur = cv2.GaussianBlur(curr_gray, (3, 3), 0)
	
	diff = cv2.absdiff(ref_blur, curr_blur)
	
	_, mask = cv2.threshold(diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
	changed_pixels_count = np.sum(mask == 255)
		
	required_change_count = total_area * (change_threshold_percent / 100.0)
	print(f'Change pixels: {changed_pixels_count}, required_change_count: {required_change_count}')
	if changed_pixels_count > required_change_count: return True
	else: return False
			
def scoreLess50():
	print('Vui long chon loai rac!')
	#screen.AddressLW(6, 2)  # warning
	#time.sleep(1)
	screen.AddressLW(11, 4)  # change window
	# get option
	select = True
	while select:
		if screen.AddressReadLB(3)>0: 
			kien.trashCan(1)
			select = False
		if screen.AddressReadLB(0)>0: 
			kien.trashCan(2)
			select = False
		if screen.AddressReadLB(1)>0: 
			kien.trashCan(3)
			select = False
		if screen.AddressReadLB(2)>0: 
			kien.trashCan(4)
			select = False
		
		ret, frame = cap.read()
		if not ret or frame is None:
			print("Failed to grab frame")
		else:
			cv2.imwrite(img_current_path, frame)
		#picam2.capture_file(img_current_path)
		check = is_object_changed(img_path, img_current_path)
		print(f'is_object_changed: {check}')
		time.sleep(CAPTURE_INTERVAL)
		if check: 
			print('a new object appeared')
			break
	
	# reset toggles
	screen.AddressWriteLB()
			
	screen.AddressLW(7, 2) # notify: thankyou for helping
	time.sleep(10)
	screen.AddressLW(8,2)
	
def scoreGreater50():
	screen.AddressLW(pic_HMI[predicted_name], 0)
	screen.AddressLW(label_HMI[predicted_name], 1)
	
	if label_HMI[predicted_name] == 0: kien.trashCan(1)
	elif label_HMI[predicted_name] == 1: kien.trashCan(2)
	elif label_HMI[predicted_name] == 2: kien.trashCan(3)
	elif label_HMI[predicted_name] == 3: kien.trashCan(4)
	
	screen.AddressLW(5,2) # notify: thank you
	time.sleep(1)

if __name__ == '__main__':
	set_up()
	while True:
		screen.AddressLW(10, 4)
		"""
		distance1 = ultraSonic.getDistance(23,24)
		distance2 = ultraSonic.getDistance(5,6)
		while distance1 <= -2 or distance2 <= -2:
			screen.AddressLW(4, 2)
			print(f'khoang  cach qua nho: {distance1}, {distance2}')
		else: screen.AddressLW(8,2)
		"""
		
		while check_object_change == False:
			ret, frame = cap.read()
			if not ret or frame is None:
				print("Failed to grab frame")
			else:
				cv2.imwrite(img_path, frame)
			
			check_object_change = is_object_changed('previous.jpg', 'test.jpg')
			print(f'{img_path}, {img_pre_path}, {check_object_change}')
			if check_object_change: 
				check_object_change = False
				break
				
		predicted_name, predicted_score = deployModel.predict_single_image(model,img_path)
		#os.remove(img_path)
		
		if predicted_score <0.5:
			scoreLess50()
				
		elif predicted_score >= 0.5:
			scoreGreater50()
			
		#screen.AddressLW(12, 4) # change to main window
		
		#cv2.imshow('Test', frame)
		shutil.copy(img_path, img_pre_path)
		
		time.sleep(CAPTURE_INTERVAL)
		#break
		if cv2.waitKey(1) == ord("q"):
			break
		  
cv2.destroyAllWindows()
