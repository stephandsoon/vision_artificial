
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
import models
yv8n_model = YOLO(models.yolov8n)
lp_model= YOLO(models.plate)
char_model = YOLO(models.charbox)


def is_image_sharp(image, thresh=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var > thresh


# video_path = './Videos/Carros.mp4'
video_path = './Videos/Motos.mp4'
# video_path = './Videos/paraminas.mp4'

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps,height,width)
# delay = 1 / fps
# delay = 0.05


if not cap.isOpened():
    print("Error al abrir el video")
    exit()

n=0
skip = 0
n_plate = 0
vehicles = [2,3,5,7]
labels = {2:'car',3:'motorcycle',5:'bus',7:'truck'}
stream = False
while True:
    
    # for _ in range(0):
    #     ret, frame = cap.read()
    #     cv2.imshow("result", frame)
    #     time.sleep(1/15)

    ret, frame = cap.read()

    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()

    if not ret:
        break
    skip+=1
    
    
    height, width, _ = frame.shape

    new_width = width
    new_height = int(height * 9 / 16)

    # top = int((height - new_height) / 2)
    # bottom = top + new_height
    # s = 150
    # frame = frame[s + top: s + bottom, :, :]
    framehd = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
    frame640 = cv2.resize(framehd, (0, 0), fx=1/2, fy=1/2)
    frame320 = cv2.resize(frame640, (0, 0), fx=1/2, fy=1/2)
    
    # if not is_image_sharp(frame, 900):
    #     cv2.imshow("result", frame)
    #     time.sleep(1/10)
    #     continue
    
    vehicles_results = yv8n_model(frame320, imgsz=320, stream=stream, verbose=False, conf=0.5, classes=[2,3,5,7])
    # results = yv8n_model(frame1,imgsz=320, verbose=False, conf=0.5)
    # print(results)
    # break
    if not vehicles_results: continue
    # if results:
    # print(vehicles_results)
    if stream:
        vehicles_detected = [result.boxes.cpu().numpy().data.astype(int) for result in vehicles_results][0]
    else:
        vehicles_results[0].plot()
        vehicles_detected = vehicles_results[0].boxes.cpu().data.numpy().astype(int)

        
    # vehicles_detected = detections[np.isin(results[:, -1], vehicles)]
    # else:
    #     continue
    
    for vehicle in vehicles_detected:
        conf, cls = vehicle[-2:]
        r = vehicle[:4] * 2
        # vehicle_frame = frame320[r[1]:r[3], r[0]:r[2]]
        vehicle_frame = frame640[r[1]:r[3], r[0]:r[2]]
        # cv2.imshow('frame320', frame640)
        # w = r[2] - r[0]
        # h = r[3] - r[1]
                # cv2.putText(frame, str(cls), (int(r[2]-w/2),int(r[3]-h/2)),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0) , 2)
                # cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)
                # area = (r[2] - r[0]) * (r[3] - r[1])
                # if area > 300000 and is_image_sharp((crop:=frame[r[1]:r[3], r[0]:r[2]]),1000):
                # if 260 < r[3] < 500 and r[2]>600:
                # if True:
                    # step=1 if cls == 3 else 1
                    # if skip%step==0:
                        # detect2 = model2(frame[r[1]:r[3], r[0]:r[2]])
        lp_results = lp_model(vehicle_frame, imgsz=640, stream=stream, verbose=False, conf=0.5)
        if lp_results:
            if stream:
                lps_detected = [result.boxes.cpu().numpy().data.astype(int) for result in lp_results][0]
            else:
                lps_detected = lp_results[0].boxes.cpu().data.numpy().astype(int)
                
            for lp in lps_detected:
                lp_conf = lp[-2]
                # cv2.imshow("annotated p", lp.plot())
                # cv2.imshow("annotated p", lp_results[0].plot())
                  
                rp = lp[:4] * 6
                if (rp[2] - rp[0]) < (rp[3] - rp[1]) or (rp[2] - rp[0])/(rp[3] - rp[1]) < 1.2 : continue
                print(rp[2] - rp[0],  rp[3] - rp[1])
                print((rp[2] - rp[0])/ (rp[3] - rp[1]))
                                    # n_plate +=1
                                    # plate = plate_detection[rp[1]:rp[3], rp[0]:rp[2]]
                                    # if plate is not None:
                                    # print(type(plate))
                                    # cv2.imwrite(f'./plates_detected/plate{n_plate}.jpg',plate)
                                    # cv2.imshow('placa', plate)
                # lp_frame = vehicle_frame[rp[1]:rp[3], rp[0]:rp[2]]
                ro = r*6
                # lp_frame = frame[rp[1]:rp[3], rp[0]:rp[2]]
                lp_frame = frame[ro[1]:ro[3], ro[0]:ro[2]][rp[1]:rp[3], rp[0]:rp[2]]
                # char_results = char_model(lp_frame, imgsz=224, stream=stream, verbose=False, iou=0.4, max_det=6)
                char_results = char_model(lp_frame, imgsz=320, stream=stream, verbose=False, iou=0.4, max_det=6)
                if char_results:
                    if stream:
                        chars_detected = [result.boxes.cpu().numpy().data.astype(int) for result in char_results][0]
                    else:
                        chars_detected = char_results[0].boxes.cpu().data.numpy().astype(int)
                        
                    for char in chars_detected:
                        char_conf = char[-2]
                        rc = char[:4]
                        cv2.rectangle(lp_frame, rc[:2], rc[2:], (0, 255, 0), 1)
                        
                                    # for char in char_results:
                                    #         cv2.imshow("annotated frame", char.plot())
                                    #         for box in char.boxes.cpu().numpy():
                                    #             rc = box.xyxy[0].astype(int)
                                    #             cv2.rectangle(plate, rc[:2], rc[2:], (0, 255, 0), 1)
                                    #     cv2.rectangle(vehicle_detection, rp[:2], rp[2:], (0, 255, 255), 2)
                                    #     cv2.imshow('placa',plate)
                                    # cv2.putText(vehicle_detection, str(box.conf[0]), (int(rp[0]),int(rp[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) , 1)
                                    # cv2.imshow('Detector de placas', plate_detection)
                        # if detect2.pred[0].tolist():
                        #     cv2.imshow('Detector de placas', np.squeeze(detect2.render()))
                            # cv2.imwrite(f"./crops/detection_{str(n)}.jpg",np.squeeze(detect2.render()))
                            
                cv2.rectangle(vehicle_frame, lp[:4][:2], lp[:4][2:], (0, 255, 255), 1)
                cv2.imshow('License Plate',lp_frame)
                # cv2.waitKey(0)
                
            # if  w>h and  700< w < 1000 and is_image_sharp((crop:=frame[r[1]:r[3], r[0]:r[2]]),1500):
            #     # crop = frame[r[1]:r[3], r[0]:r[2]]
            #     n+=1
                # cv2.imshow('Captura Carro', crop)
                # cv2.imwrite(f"./crops/Car{str(n)}.jpg", crop)
        if cls ==3 :
            cv2.rectangle(frame640, r[:2], r[2:], (255, 0, 0), 2)
        else:
            cv2.rectangle(frame640, r[:2], r[2:], (255, 255, 255), 2)
                    # if w > 200 and w>h/3.3 and w < 700 and is_image_sharp((crop:=frame[r[1]:r[3], r[0]:r[2]]),1500):
                    # if w > 200 and w>h/3.3 and w < 700 and 300 < r[3] and is_image_sharp((crop:=frame[r[1]:r[3], r[0]:r[2]]),1000):
                            # crop = frame[r[1]:r[3], r[0]:r[2]]
                        # n+=1
                        # cv2.imshow('Captura Moto', crop)
                        # cv2.imwrite(f"./crops/Bike{str(n)}.jpg", crop)
                    
    # cv2.rectangle(frame, (600, 260), (width, 500), (0, 0, 255), 2)
    # frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("result", frame640)
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

