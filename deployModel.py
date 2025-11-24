# -*- coding: cp1258 -*-
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# ======================
# 1. Thông tin model
# ======================
class_names = [
    'banana', 'battery', 'bimbim', 'coc', 'cucumberpeel', 'egg', 'hopxop',
    'leaves', 'lotCoc', 'mask', 'metal', 'orange', 'paper', 'pen',
    'bottle', 'bag'
]

IMG_SIZE = (240, 240)


# ======================
# 2. Hàm l?y ROI
# ======================
def get_roi(frame):
    h, w = frame.shape[:2]

    # Khung ROI gi?ng nhu b?n d?t trong kien_deploy_model
    roi_x1 = max(0, w//2 - 250)
    roi_y1 = max(0, h//2 - 220)
    roi_x2 = min(w, w//2 + 180)
    roi_y2 = min(h, h//2 + 210)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    if roi is None or roi.size == 0:
        return None

    return roi

def is_black_background(roi, threshold=100, black_ratio=0.90):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < threshold)
    total_pixels = gray.size
    ratio = black_pixels / total_pixels
    return ratio >= black_ratio

# ======================
# 3. Predict tr?c ti?p t? ROI (tuong t? kien_deploy_model)
# ======================
def predict_frame(model, frame):

    roi = get_roi(frame)
    if roi is None:
        raise ValueError("ROI is empty or out of frame range")
    
    if is_black_background(roi):
        return "none", 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    img_array = np.expand_dims(gray, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    idx = np.argmax(predictions, axis=-1)[0]
    score = predictions[0][idx]

    return class_names[idx], score


# ======================
# 4. Predict t? file ?nh (dùng ROI tru?c khi predict)
# ======================
def predict_single_image(model, img_path):

    frame = cv2.imread(img_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {img_path}")

    roi = get_roi(frame)
    if roi is None:
        raise ValueError("ROI out of range in predict_single_image")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    img_array = np.expand_dims(gray, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    idx = np.argmax(predictions, axis=-1)[0]
    score = predictions[0][idx]

    predicted_class_name = class_names[idx]

    print("Predicted class:", predicted_class_name)
    print("Predicted score:", score)

    return predicted_class_name, score


# ======================
# 5. Ch?y test d?c l?p
# ======================
if __name__ == '__main__':

    model_path = "/home/pi/DoAn/trash_14_cls_grayscale_finetuned_noLambda.keras"
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        exit()

    for _ in range(10):
        ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to grab frame")
        exit()

    # Test predict_frame
    cls, score = predict_frame(model, frame)
    print("Predict_frame result:", cls, score)

    # Test save + predict_single_image
    cv2.imwrite("test.jpg", frame)
    cls2, score2 = predict_single_image(model, "test.jpg")
    print("predict_single_image result:", cls2, score2)
