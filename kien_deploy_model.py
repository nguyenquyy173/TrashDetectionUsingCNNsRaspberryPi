# -*- coding: cp1258 -*-
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class_names = [
    'banana', 'battery', 'bimbim', 'coc', 'cucumberpeel', 'egg', 'hopxop',
    'leaves', 'lotCoc', 'mask', 'metal', 'orange', 'paper', 'pen',
    'bottle', 'bag'
]

IMG_SIZE = (240, 240)

model_path = "/home/pi/DoAn/trash_16_cls_grayscale_finetuned_noLambda.keras"
model = tf.keras.models.load_model(model_path)

def predict_frame(model, roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    img_array = np.expand_dims(gray, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    idx = np.argmax(predictions, axis=-1)[0]
    score = predictions[0][idx]

    return class_names[idx], score

def is_black_background(roi, threshold=100, black_ratio=0.97):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < threshold)
    total_pixels = gray.size
    ratio = black_pixels / total_pixels
    return ratio >= black_ratio


if __name__ == "__main__":
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("? ERROR: Cannot open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ============================
        # 1) T?O KHUNG ROI (vùng nh?n d?ng)
        # ============================
        h, w = frame.shape[:2]

        # ví d? t?o khung chính gi?a
        roi_x1 = max(0, w//2 - 250)
        roi_y1 = max(0, h//2 - 220)
        roi_x2 = min(w, w//2 + 180)
        roi_y2 = min(h, h//2 + 210)

        # v? khung
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2),
                      (0, 255, 0), 2)

        # ============================
        # 2) C?T ROI
        # ============================
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if is_black_background(roi):
            predicted_name = "none"
            score = 0.0
        else: predicted_name, score = predict_frame(model, roi)

        # hi?n th? k?t qu? ngay t?i khung ROI
        cv2.putText(frame,
                    f"{predicted_name}: {score:.3f}",
                    (roi_x1, roi_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 2)

        # show h́nh
        cv2.imshow("Trash Classification Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
