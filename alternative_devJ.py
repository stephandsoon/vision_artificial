# Library import
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
import models
from functions import *


# Define and open externally pretrained models
yv8n_model = YOLO(models.yolov8n)
lp_model= YOLO(models.plate2)
char_model = YOLO(models.chars3)


# Define the path of the video to be analized, load it and print video properties (width, height, frames per second). Raise error if opening is not possible.
video_path = './BigFiles/minas_parqueadero_SH.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print("Video opened. Video properties: Frames per second =", fps, "| Video size (height x width):", height, "x", width)
if not cap.isOpened():
    print("Error while opening video.")
    exit()
#Defining we are working with a video currently (no stream)
stream = False

# Determine the vehicle classes that are to be recognized
labels = {2:'car',3:'motorcycle',5:'bus',7:'truck'}


# Create image analization loop until the end of the video
while True:

    # Read the loaded video until the end of the video
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Scale frame down to a lower resolution to achieve less necessary computational power and therefore faster vehicle recognition     
    new_width = width
    new_height = int(height * 9 / 16)
    framehd = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
    frame640 = cv2.resize(framehd, (0, 0), fx=1/2, fy=1/2)
    frame320 = cv2.resize(frame640, (0, 0), fx=1/2, fy=1/2)
        
    # Vehicle detection with loaded YOLOv8n model using the frame with reduced resolution for faster processing | parameter vid_stride defines framerate: every n frames 1 frame is being processed by YOLOv8n
    vehicles_results = yv8n_model(frame320, imgsz=320, stream=stream, verbose=False, conf=0.5, classes=[2,3,5,7], vid_stride=2) 
    
    # If no vehicle detection just continue with next iteration
    if not vehicles_results: continue
    
    # Extracting the bounding box coordinates from the results of the vehicle detection model and converting them to a NumPy array of integers
    if stream:
        vehicles_detected = [result.boxes.cpu().numpy().data.astype(int) for result in vehicles_results][0]
    else:
        vehicles_results[0].plot()
        vehicles_detected = vehicles_results[0].boxes.cpu().data.numpy().astype(int)

    # Now pass the coordinate of the bounding boxes of the recognized vehicle to the license plate detecion model.    
    for vehicle in vehicles_detected:

        # Recovering the 640 pixels image quality for the license plate detection 
        conf, cls = vehicle[-2:]
        r = vehicle[:4] * 2
        vehicle_frame = frame640[r[1]:r[3], r[0]:r[2]]
    
        # Detection of the license plates using the externally pretrained license plate recognition model
        lp_results = lp_model(vehicle_frame, imgsz=640, stream=stream, verbose=False, conf=0.5, iou=0.4)
        
        # If no license plated detected just continue with next iteration
        if not lp_results: continue
        
        # Extracting the bounding box coordinates from the results of the vehicle detection model and converting them to a NumPy array of integers
        if stream:
            lps_detected = [result.boxes.cpu().numpy().data.astype(int) for result in lp_results][0]
        else:
            lps_detected = lp_results[0].boxes.cpu().data.numpy().astype(int)
            
        # Now pass the coordinate of the bounding boxes of the recognized license plate to the character identification model and iterate through them
        for lp in lps_detected:
            
            # Extract from the original frame the detection area and re-establish the original video quality
            lp_conf = lp[-2]
            rp = lp[:4] * 6
            # Pre-selection: sort out all the images where the width-height ratio doesn´t fit the expected license plate ratio
            if (rp[2] - rp[0])/(rp[3] - rp[1]) < 1.2 : continue
            ro = r*6
            lp_frame = frame[ro[1]:ro[3], ro[0]:ro[2]][rp[1]:rp[3], rp[0]:rp[2]]
            
            # Identify the characters on the license plate by the character identification model
            char_results = char_model(lp_frame, imgsz=224, stream=stream, verbose=False, iou=0.4, max_det=6)
            
            # If no characters detected and identified just continue with next iteration
            if not char_results: continue

            # Extracting the bounding box coordinates from the results of the character identification model and converting them to a NumPy array of integers
            if stream:
                chars_detected = [result.boxes.cpu().numpy().data.astype(int) for result in char_results][0]
            else:
                char_results[0].plot()
                chars_detected = char_results[0].boxes.cpu().data.numpy().astype(int)
                print(chars_detected)
                
            
            ###------------Visualization of the detected vehicles, license plates and characters-------------###
            # Visualize all the recognized characters
            for char in chars_detected:
                char_conf = char[-2]
                rc = char[:4]
                cv2.rectangle(lp_frame, rc[:2], rc[2:], (0, 255, 0), 1)
            # Visualize all the recognized license plates 
            cv2.rectangle(vehicle_frame, lp[:4][:2], lp[:4][2:], (0, 255, 255), 1)
            # Use for those recognitions an additional window showing the license plate frame
            cv2.imshow('License Plate',lp_frame)

        # Visualize the vehicle detection in the 640 pixels frame (detection of motorcycles in blue, any other vehicle detection in white)
        if cls ==3 :
            cv2.rectangle(frame640, r[:2], r[2:], (255, 0, 0), 2)
        else:
            cv2.rectangle(frame640, r[:2], r[2:], (255, 255, 255), 2)

    # Show the 640 pixels frame while executing the detection              
    cv2.imshow("result", frame640)

    # Define options to close the windows and object recognition with the key "q" and to pause it with the key "p"
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

# Clean finish: Release the captured frame and close windows
cap.release()
cv2.destroyAllWindows()

