# Library import
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
import models
from functions import *
import re
from datetime import datetime


# Define and open externally pretrained models
yv8n_model = YOLO(models.yolov8n)
lp_model= YOLO(models.plate2)
char_model = YOLO(models.chars3)


# Define the path of the video to be analized, load it and print video properties (width, height, frames per second). Raise error if opening is not possible.
video_path = './BigFiles/minas_parqueadero.mp4'
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
labels_lp= {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

# Create an empty list where all the recognized character identifications are going to be stored
identified_characters = []


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
            
            # Extract from the original frame the license plate detection area and re-establish the original video quality
            lp_conf = lp[-2]
            rp = lp[:4] * 6
            # Pre-selection: sort out all the images where the width-height ratio doesn´t fit the expected license plate ratio
            if (rp[2] - rp[0])/(rp[3] - rp[1]) < 1.2 : continue
            ro = r*6
            lp_frame = frame[ro[1]:ro[3], ro[0]:ro[2]][rp[1]:rp[3], rp[0]:rp[2]]
            

            # Image pre-processing for better character recognition
            # Define where to save the license plate images after preprocessing
            subfolder_path = "fotos"
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_lp = f"{subfolder_path}/license_plate_{current_time}.png"
            cv2.imwrite(filename_lp, lp_frame)
            #Preprocessing way no.1 for yellow license plates
            lp_frame_preprocessed_1_step_1 = preprocessing_1_segmentation(lp_frame)
            lp_frame_preprocessed_2_step_1 = preprocessing_2_segmentation(lp_frame)
            if lp_frame_preprocessed_2_step_1[:,:,2].mean() < 35:
                lp_frame_preprocessed_1_step_2 = preprocessing_1_color_correction(lp_frame_preprocessed_1_step_1)
                preprocessed_img_3channels = cv2.cvtColor(lp_frame_preprocessed_1_step_2, cv2.COLOR_GRAY2RGB)
                #Save preprocessed license plate image 1
                filename_ppi_1 = f"{subfolder_path}/lp_ppi_1_{current_time}.png"
                cv2.imwrite(filename_ppi_1, lp_frame_preprocessed_1_step_2)
            else:
                #Preprocessing way no.2 for white license plates and converting it into a 3 channel img
                lp_frame_preprocessed_2_step_2 = preprocessing_2_color_correction(lp_frame_preprocessed_2_step_1)
                preprocessed_img_3channels = cv2.cvtColor(lp_frame_preprocessed_1_step_2, cv2.COLOR_GRAY2RGB)
                #Save preprocessed license plate image 2
                filename_ppi_2 = f"{subfolder_path}/lp_ppi_2_{current_time}.png"
                cv2.imwrite(filename_ppi_2, lp_frame_preprocessed_2_step_2)
                # print("Damaged license plate found.")
                
            # Show the preprocessed image
            cv2.imshow('Preprocessed license plate',preprocessed_img_3channels)

            # Identify the characters on the license plate by the character identification model
            char_results = char_model(preprocessed_img_3channels, imgsz=224, stream=stream, verbose=False, iou=0.8, max_det=6, conf=0.2)
    
            # If no characters detected and identified just continue with next iteration
            if not char_results: continue

            # Extracting the bounding box coordinates from the results of the character identification model and converting them to a NumPy array of integers
            if stream:
                chars_detected = [result.boxes.cpu().numpy().data.astype(int) for result in char_results][0]
            else:
                char_results[0].plot()
                chars_detected = char_results[0].boxes.cpu().data.numpy().astype(int)
            
            # Extract the detected character prediction results if there are 6 characters identified
            lp_text = ''
            chars_detected_ordenados = sorted(chars_detected, key=lambda char: char[0])
            for char in chars_detected_ordenados:
                # print(char)
                char_conf, char_cls = char[-2:]
                rc = char[:4]
                # Visualize the identified characters in a green box
                cv2.rectangle(lp_frame, rc[:2], rc[2:], (0, 255, 0), 1)
                lp_text+=labels_lp[char_cls]
            if len(lp_text)==6:
                print(lp_text)
                # Append the recognized characters to the list identified_characters to save them later on
                identified_characters.append(lp_text)
            if x:=re.match(re.compile(r'^[A-Z]{3}\d{2}[A-Z0-9]{1}$'), lp_text):
                text = x[0]
                print(text)
                                
            
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

# Clean finish: Release the captured frame and close windows
cap.release()
cv2.destroyAllWindows()

# Save the recognized license plates list to a txt-file
# Open the file in write mode
with open("recognitions.txt", 'w') as file:
    # Write each string from the list to the file
    for item in identified_characters:
        file.write(item + '\n')



