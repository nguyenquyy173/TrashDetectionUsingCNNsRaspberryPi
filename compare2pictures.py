import cv2
import numpy as np
import cv2

DEFAULT_PIXEL_DIFF_THRESHOLD = 15
DEFAULT_MAJOR_CHANGE_PERCENT = 5

def is_object_changed(
    reference_path: str, 
    current_path: str, 
    change_threshold_percent: float = DEFAULT_MAJOR_CHANGE_PERCENT,
    pixel_diff_threshold: int = DEFAULT_PIXEL_DIFF_THRESHOLD
) -> bool:
	reference_image = cv2.imread(reference_path)
	reference_image = cv2.resize(reference_image, (240, 240))
	
	current_image = cv2.imread(current_path)
	current_image = cv2.resize(current_image, (240, 240))
	
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
	
	ref_blur = cv2.GaussianBlur(ref_gray, (9, 9), 0)
	curr_blur = cv2.GaussianBlur(curr_gray, (9, 9), 0)
	
	diff = cv2.absdiff(ref_blur, curr_blur)
	
	_, mask = cv2.threshold(diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
	changed_pixels_count = np.sum(mask == 255)
		
	required_change_count = total_area * (change_threshold_percent / 100.0)
	print(f'Change pixels: {changed_pixels_count}')
	if changed_pixels_count > required_change_count: return True
	else: return False
    
if __name__ == '__main__':
	print('Start running')
	SIZE = 240
	
	result = is_object_changed('test.jpg', 'previous.jpg')
	print(f'RESULT: {result}')
