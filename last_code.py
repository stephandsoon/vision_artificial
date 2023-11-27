
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
import Models
yv8n_model = YOLO(Models.yolov8n)
lp_model= YOLO(Models.plate)
# lp_model= YOLO(Models.plate)
char_model = YOLO(Models.charbox)
# ynas_model = NAS("yolo_nas_s.pt")
import easyocr
import cv2
import re

reader = easyocr.Reader(['en'], gpu=False)

def is_image_sharp(image, thresh=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var > thresh

def resize(frame):
    framehd = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
    frame640 = cv2.resize(framehd, (0, 0), fx=1/2, fy=1/2)
    frame320 = cv2.resize(frame640, (0, 0), fx=1/2, fy=1/2)
    return framehd, frame640, frame320



def otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return bin_otsu

def adaptative(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive_thresh = cv2.adaptiveThreshold(
    #     # gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    #     cv2.THRESH_BINARY, 97, 10
    # )
    adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,199,20)
    return adaptive_thresh

def ocr(img):
    img_h, img_w = img.shape[:2]
    plate_area = img_h*img_w
    
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_input = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,199,20)
    cv2.imshow('ocr_view', img_input)
    contours, hierarchy = cv2.findContours(img_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    xmi = img_w//2
    ymi = img_h//2
    hma = 0
    wma = 0
    
    for contour in contours:
        (x,y,w1,h1) = cv2.boundingRect(contour)
        box_area = w1 * h1
        if box_area/plate_area>0.05 and h1/w1>1.85: 
            save = img.copy() ###
            cv2.rectangle(img, (x,y), (x+w1,y+h1), (255, 0, 0), 2)
            cv2.imshow('detections', img)
            # box_area = cv2.contourArea(contour)
            # print('area:', box_area, 'img_area', plate_area,
            #   'rel', box_area/plate_area,
            #   'h/w', h1/w1)
            if x < xmi: xmi = x
            if y < ymi: ymi  = y
            if h1 + y > hma: hma = h1 + y
            if w1 + x > wma: wma = w1 + x
            img = save
            # cv2.waitKey(0)
        # if cv2.contourArea(contour) >a:
    snipped = img_gray[ymi : hma, xmi :wma ]
    try:
        ocr_detections = reader.readtext(snipped)
    except:
        return ''
    
    for ocr_detection in ocr_detections:
        bbox, text, score = ocr_detection
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        # print(text)
        if x:=re.search(r'\w{3}\d{2}[\w\d]', text, flags=0):
            print(x[0])
            cv2.imshow('to_ocr', snipped)
            return(x[0])
    # cv2.imshow('to_ocr', snipped)
    return ''
#  cv2.adaptiveThreshold(plate,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 97, 10)

# video_path = './Videos/Carros.mp4'
# video_path = './Videos/VID_20231120_161225.mp4'
# video_path = './Videos/VID_20230506_170043.mp4'
# video_path = './Videos/Motos.mp4'
# video_path = './Videos/paraminas.mp4'
# video_path = './Videos/evaluationSH.mp4'
# video_path = './Videos/evaluation1.mp4'
# video_path = './Videos/evaluation2.mp4'
video_path = './Videos/volador.mp4'

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps,height,width)
delay = 1 / fps
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
labels_lp= {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 
            15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 
            20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 
            25: 'P', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 
            30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}
scale = width/720

car_plates = dict()
bus_plates = dict()
bike_plates = dict()
truck_plates = dict()

while True:
    

    for _ in range(5):
        ret, frame = cap.read()
        if not ret: break
        frame640 = cv2.resize(frame, (0, 0), fx=1/(2*scale), fy=1/(2*scale))
        cv2.imshow("frame", frame360)
        time.sleep(delay)

    ret, frame = cap.read()
    if not ret: break
    skip+=1
    
    
    # height, width, _ = frame.shape

    new_width = width
    new_height = int(height * 9 / 16)
    
    frame720 = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)
    frame360 = cv2.resize(frame720, (0, 0), fx=1/2, fy=1/2)
    frame180 = cv2.resize(frame360, (0, 0), fx=1/2, fy=1/2)
    
    
    # if not is_image_sharp(frame360, 900):
    #     print('skiping frame')
    #     cv2.imshow("frame", frame360)
    #     time.sleep(delay)
    #     continue
    
    vehicles_results = yv8n_model.track(frame180, imgsz=320, stream=stream, verbose=False, conf=0.5, classes=[2,3,5,7])
    
    if stream:
        vehicles_detected = [result.boxes.cpu().numpy().data.astype(int) for result in vehicles_results][0]
    else:
        # cv2.imshow('V',vehicles_results[0].plot())
        vehicles_detected = vehicles_results[0].boxes.cpu().data.numpy().astype(int)

    # vehicles_detected = detections[np.isin(results[:, -1], vehicles)]
    if vehicles_detected.size == 0:
        car_plates = dict()
        bike_plates = dict()
    
    for vehicle in vehicles_detected:
        v_id, conf, cls = vehicle[-3:]
        r = vehicle[:4] * 2
        vehicle_frame = frame360[r[1]:r[3], r[0]:r[2]]
        cv2.imshow("annotated v", vehicles_results[0].plot())
    
        lp_results = lp_model(vehicle_frame, imgsz=640, stream=stream, verbose=False, conf=0.5, iou=0, max_det=1)
        
        if stream:
            lps_detected = [result.boxes.cpu().numpy().data.astype(int) for result in lp_results][0]
        else:
            lps_detected = lp_results[0].boxes.cpu().data.numpy().astype(int)
            
            
        for lp in lps_detected:
            lp_conf = lp[-2]
            # cv2.imshow("annotated p", lp.plot())
            # cv2.imshow("annotated p", lp_results[0].plot())
            rp = lp[:4] * int(2*scale)
            ro = r * int(2*scale)
            if (rp[2] - rp[0])/(rp[3] - rp[1]) < 1.5 : continue
            # print('lp_ratio', (rp[2] - rp[0])/(rp[3] - rp[1]))
            # print(rp[2] - rp[0],  rp[3] - rp[1])
            # print((rp[2] - rp[0])/ (rp[3] - rp[1]))
            lp_frame = frame[ro[1]:ro[3], ro[0]:ro[2]][rp[1]:rp[3], rp[0]:rp[2]]
            

            lp_text = ocr(lp_frame)
            if lp_text !='': print(lp_text)

            
            # cv2.rectangle(vehicle_frame, lp[:4][:2]*2, lp[:4][2:]*2, (0, 255, 255), 1)
            cv2.rectangle(vehicle_frame, lp[:4][:2]*2, lp[:4][2:]*2, (0, 255, 255), 1)
            cv2.imshow('License Plate',lp_frame)

        if cls ==3 :
            cv2.rectangle(frame360, r[:2], r[2:], (255, 0, 0), 2)
        else:
            cv2.rectangle(frame360, r[:2], r[2:], (255, 255, 255), 2)
                  
    cv2.imshow("frame", frame360)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(0)
        
    elif key == ord('x'):
        for i in range(500):
            ret, frame = cap.read()
            # frame640 = cv2.resize(framehd, (0, 0), fx=1/6, fy=1/6)
            cv2.imshow("fame", frame360)
            time.sleep(delay/10)
            if not ret : break
        cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()