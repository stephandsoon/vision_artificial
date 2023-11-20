from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')
results = model.predict(source='0', show=True)
print(results)