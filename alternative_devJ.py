
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
import models

yv8n_model = YOLO(models.yolov8n)
lp_model= YOLO(models.plate2)
char_model = YOLO(models.chars3)



    
def is_image_sharp(image, thresh=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var > thresh

    
stream = False
def detect(model, frame, imgsz, conf=0.5, classes=None, stream=stream, 
           verbose=False, show=False, iou=0.7, max_det=300):

    results = model(frame, imgsz=imgsz, stream=stream, verbose=verbose, 
                    conf=conf, classes=classes, iou=iou, max_det=max_det)
    
    if not results: return []
    
    if stream:
        detected = [result.boxes.cpu().numpy().data.astype(int) for result in results][0]
    else:
        if show: results[0].plot()
        detected = results[0].boxes.cpu().data.numpy().astype(int)
    return detected

def otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return bin_otsu

def otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # adaptive_thresh = cv2.adaptiveThreshold(
    #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     cv2.THRESH_BINARY, 11, 2
    # )
    # return adaptive_thresh
    return bin_otsu


video_path = './Videos/Carros.mp4'
# video_path = './Videos/VID_20231120_161225.mp4'
# video_path = './Videos/VID_20230506_170043.mp4'
# video_path = './Videos/Motos.mp4'
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
    

    ret, frame = cap.read()
    for _ in range(0):
        ret, frame = cap.read()
        frame640 = cv2.resize(framehd, (0, 0), fx=1/6, fy=1/6)
        cv2.imshow("result", frame640)
        time.sleep(1/15)

    if not ret:
        break
    skip+=1
    
    
    height, width, _ = frame.shape

    new_width = width
    new_height = int(height * 9 / 16)

    framehd = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
    # framehd = cv2.resize(frame, (0, 0), fx=1/1.5, fy=1/1.5)
    frame640 = cv2.resize(framehd, (0, 0), fx=1/2, fy=1/2)
    frame320 = cv2.resize(frame640, (0, 0), fx=1/2, fy=1/2)
    
    
    # if not is_image_sharp(frame, 900):
    #     cv2.imshow("result", frame)
    #     time.sleep(1/10)
    #     continue
    
    vehicles_results = yv8n_model(frame320, imgsz=320, stream=stream, verbose=False, conf=0.5, classes=[2,3,5,7])
    
    if not vehicles_results: continue
    
    if stream:
        vehicles_detected = [result.boxes.cpu().numpy().data.astype(int) for result in vehicles_results][0]
    else:
        vehicles_results[0].plot()
        vehicles_detected = vehicles_results[0].boxes.cpu().data.numpy().astype(int)

    # vehicles_detected = detections[np.isin(results[:, -1], vehicles)]
    
    for vehicle in vehicles_detected:
        conf, cls = vehicle[-2:]
        r = vehicle[:4] * 2
        vehicle_frame = frame640[r[1]:r[3], r[0]:r[2]]
    
        lp_results = lp_model(vehicle_frame, imgsz=640, stream=stream, verbose=False, conf=0.5, iou=0.4)
        
        if not lp_results: continue
        
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
            # print(rp[2] - rp[0],  rp[3] - rp[1])
            # print((rp[2] - rp[0])/ (rp[3] - rp[1]))
            ro = r*6
            lp_frame = frame[ro[1]:ro[3], ro[0]:ro[2]][rp[1]:rp[3], rp[0]:rp[2]]
            
            
            
            char_results = char_model(lp_frame, imgsz=224, stream=stream, verbose=False, iou=0.4, max_det=6)
            # char_results = char_model(lp_frame, imgsz=320, stream=stream, verbose=False, iou=0.4, max_det=6)
            if not char_results: continue
            
            # lp_bin = otsu(lp_frame)
            # cv2.imshow('License Plate Bin',lp_bin)
            # ocr_detections = reader.readtext(lp_bin)
            # for ocr_detection in ocr_detections:
            #     bbox, text, score = ocr_detection
            #     text = re.sub(r'[.,/#@ \)\(:"];_\[\]=]', '', text.upper())
            #     if x:=re.match(re.compile(r'^[A-Z]{3}\d{2}[A-Z0-9]{1}$'), text):
            #         text = x[0]
            #         print(text)
                
            # if len(text) ==6:
            #     print(text)
            
            if stream:
                chars_detected = [result.boxes.cpu().numpy().data.astype(int) for result in char_results][0]
            else:
                char_results[0].plot()
                chars_detected = char_results[0].boxes.cpu().data.numpy().astype(int)
                print(chars_detected)
                
                
            for char in chars_detected:
                char_conf = char[-2]
                rc = char[:4]
                cv2.rectangle(lp_frame, rc[:2], rc[2:], (0, 255, 0), 1)
            
            
                
            cv2.rectangle(vehicle_frame, lp[:4][:2], lp[:4][2:], (0, 255, 255), 1)
            cv2.imshow('License Plate',lp_frame)

        if cls ==3 :
            cv2.rectangle(frame640, r[:2], r[2:], (255, 0, 0), 2)
        else:
            cv2.rectangle(frame640, r[:2], r[2:], (255, 255, 255), 2)
                  
    cv2.imshow("result", frame640)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(0)
        
    elif key == ord('x'):
        for i in range(500):
            ret, frame = cap.read()
            frame640 = cv2.resize(framehd, (0, 0), fx=1/6, fy=1/6)
            cv2.imshow("result", frame640)
            time.sleep(1/60)
            if not ret : break
        cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

