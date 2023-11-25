

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
import models
yv8n_model = YOLO(models.yolov8n)
lp_model= YOLO(models.plate)
char_model = YOLO(models.charbox)

# video_path = './Videos/Carros.mp4'
video_path = './BigFiles/minas_parqueadero_SH.mp4'
# video_path = './Videos/paraminas.mp4'

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps,height,width)

def is_image_sharp(image, thresh=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var > thresh

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

n=0
skip = 0
n_plate = 0
while True:
    
    for _ in range(0):
        ret, frame = cap.read()
        cv2.imshow("result", frame)
        time.sleep(1/15)

    ret, frame = cap.read()

    if not ret:
        break
    skip+=1
    
    
    height, width, _ = frame.shape

    new_width = width
    new_height = int(height * 9 / 16)
    frame2 = frame.copy()
    frame = cv2.resize(frame, (0, 0), fx=1/4, fy=1/4)
    frame1 = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
    results = yv8n_model(frame1, imgsz=320, stream=True, verbose=False, conf=0.5)
    
    for result in results:
        for box in result.boxes.cpu().numpy():
            cls = int(box.cls[0])
            conf = box.conf[0]

            vehicle_found = cls in [2,3,5,7] #and conf > 0.65
            labels = {2:'car',3:'motorcycle',5:'bus',7:'truck'}
            if vehicle_found:
                r = 3*(box.xyxy[0].astype(int))
                w = r[2] - r[0]
                h = r[3] - r[1]
                if True:
                    step=1 if cls == 3 else 1
                    if skip%step==0:
                        vehicle_detection = frame[r[1]:r[3], r[0]:r[2]]
                        plate_results = lp_model(vehicle_detection, imgsz=640, stream=True, verbose=False, conf=0.65 )
                        if plate_results:  
                            for result in plate_results:
                                for box in result.boxes.cpu().numpy():
                                    n_plate +=1
                                    rp = 4*(box.xyxy[0].astype(int))
                                    if (rp[2] - rp[0]) < (rp[3] - rp[1]) : continue
                                    plate = vehicle_detection[rp[1]:rp[3], rp[0]:rp[2]]
                                    plate = frame2[4*r[1]:4*r[3], 4*r[0]:4*r[2]][rp[1]:rp[3], rp[0]:rp[2]]
                                    chars_results = char_model(plate, imgsz=224, stream=True, verbose=False)
                                    if chars_results:
                                        for char in chars_results:
                                            for box in char.boxes.cpu().numpy():
                                                rc = box.xyxy[0].astype(int)
                                                cv2.rectangle(plate, rc[:2], rc[2:], (0, 255, 0), 1)
                                        cv2.rectangle(vehicle_detection, rp[:2], rp[2:], (0, 255, 255), 2)
                                        cv2.imshow('placa',plate)
                                    cv2.putText(vehicle_detection, str(box.conf[0]), (int(rp[0]),int(rp[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) , 1)

                if cls == 2 or cls==7 and conf > 0.6:
                    if  w>h and  700< w < 1000 and is_image_sharp((crop:=frame[r[1]:r[3], r[0]:r[2]]),1500):
                        # crop = frame[r[1]:r[3], r[0]:r[2]]
                        n+=1
                        # cv2.imshow('Captura Carro', crop)
                        # cv2.imwrite(f"./crops/Car{str(n)}.jpg", crop)
                    cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)
                elif cls ==3 and conf > 0.4 :
                    # if w > 200 and w>h/3.3 and w < 700 and is_image_sharp((crop:=frame[r[1]:r[3], r[0]:r[2]]),1500):
                    if w > 200 and w>h/3.3 and w < 700 and 300 < r[3] and is_image_sharp((crop:=frame[r[1]:r[3], r[0]:r[2]]),1000):
                            # crop = frame[r[1]:r[3], r[0]:r[2]]
                        n+=1
                        # cv2.imshow('Captura Moto', crop)
                    cv2.rectangle(frame, r[:2], r[2:], (255, 0, 0), 2)
                        # cv2.imwrite(f"./crops/Bike{str(n)}.jpg", crop)
                    
    cv2.imshow("result", frame)
    # frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow("result", frame_resized)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # video_salida.release()
        break
    elif key == ord('p'):
        cv2.waitKey(0)
        
    elif key == ord('x'):
        for i in range(1000):
            ret, frame = cap.read()
            if not ret : break
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

