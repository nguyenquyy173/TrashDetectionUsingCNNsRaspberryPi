# -*- coding: cp1258 -*-
import cv2

# M? webcam (0 = camera m?c d?nh)
cap = cv2.VideoCapture(0)

# Thi?t l?p d? phân gi?i (tu? ch?n)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("? Không m? du?c webcam!")
    exit()

print("? Webcam dă m?, dang hi?n th? video...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("? Không d?c du?c frame t? webcam!")
        break

    # Hi?n th? h́nh
    cv2.imshow("Webcam Test", frame)

    # Nh?n q d? thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
