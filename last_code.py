# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
import Models
import easyocr
import cv2
import re

# Configure YOLO models and EasyOCR
yv8n_model = YOLO(Models.yolov8n)
lp_model = YOLO(Models.plate)
char_model = YOLO(Models.charbox)
reader = easyocr.Reader(['en'], gpu=False)

# Function to check image sharpness
def is_image_sharp(image, thresh=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var > thresh

# Function to resize an image
def resize(frame):
    framehd = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
    frame640 = cv2.resize(framehd, (0, 0), fx=1/2, fy=1/2)
    frame320 = cv2.resize(frame640, (0, 0), fx=1/2, fy=1/2)
    return framehd, frame640, frame320

# Function to perform OCR on an image
def ocr(img):
    img_h, img_w = img.shape[:2]
    plate_area = img_h * img_w
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_input = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 20)
    cv2.imshow('ocr_view', img_input)
    
    contours, hierarchy = cv2.findContours(img_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    xmi = img_w // 2
    ymi = img_h // 2
    hma = 0
    wma = 0
    
    for contour in contours:
        (x, y, w1, h1) = cv2.boundingRect(contour)
        box_area = w1 * h1
        if box_area / plate_area > 0.05 and h1 / w1 > 1.85:
            save = img.copy()  # Make a copy of the original image
            cv2.rectangle(img, (x, y), (x + w1, y + h1), (255, 0, 0), 2)  # Draw rectangle around the area of interest
            cv2.imshow('detections', img)
            
            if x < xmi: xmi = x
            if y < ymi: ymi = y
            if h1 + y > hma: hma = h1 + y
            if w1 + x > wma: wma = w1 + x
            img = save
            
    # Crop the region of interest
    snipped = img_gray[ymi:hma, xmi:wma]
    
    try:
        # Perform OCR on the cropped region
        ocr_detections = reader.readtext(snipped)
    except:
        return ''
    
    for ocr_detection in ocr_detections:
        bbox, text, score = ocr_detection
        text = re.sub(r'[^A-Z0-9]', '', text.upper())  # Remove unwanted characters
        if x := re.search(r'\w{3}\d{2}[\w\d]', text, flags=0):
            print(x[0])
            cv2.imshow('to_ocr', snipped)
            return x[0]
    
    return ''

# Video capture setup
video_path = './Videos/volador.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps, height, width)
delay = 1 / fps

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening the video")
    exit()

# Initialize variables
n = 0
skip = 0
n_plate = 0
vehicles = [2, 3, 5, 7]
stream = False
labels_lp = {0: '0', 1: '1', 2: '2', 3: '3', ...
