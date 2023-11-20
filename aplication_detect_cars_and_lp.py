from ultralytics import YOLO
import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')
lp_model = YOLO('plate_model.pt')

while True:
    
    ret, frame = cap.read()
    
    if not ret: break
    
    results = model(frame, imgsz=320, stream=True, verbose=False)
    for result in results:
         for box in result.boxes.cpu().numpy():
            # print(box)
            cls = int(box.cls[0])
            conf = box.conf[0]

            vehicle_found = cls in [2,3,5,7] and conf > 0.65 #  2: 'car',  3: 'motorcycle',  5: 'bus', 7: 'truck',
            labels = {2: 'car',  3: 'motorcycle',  5: 'bus', 7: 'truck'}

            if vehicle_found:
                # Define box coordinates
                r = box.xyxy[0].astype(int) # -> r[0]=x1; r[1]=y1; r[2]=x2; r[3]=y2;
                w = r[2] - r[0]
                h = r[3] - r[1]
                # Draw a rectangle around that found object if it´s fulfilles the requirements (vehicle with certainty >0.65)
                cv2.rectangle(frame, (r[0],r[1]), (r[2],r[3]), (255, 255, 255), 1) #r[:2], r[2:], (255, 255, 255), 2)
                # Extract the region inside the rectangle
                vehicle_frame = frame[r[1]:r[3], r[0]:r[2]]
                # Save the extracted region to a PNG file with the current timestamp
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                subfolder_path = "fotos"
                filename = f"{subfolder_path}/vehicle_{current_time}.png"
                cv2.imwrite(filename, vehicle_frame)

                #From here on: pass the vehicle_frame to that placa model.
                lp_results = lp_model(vehicle_frame, imgsz=320, stream=True, verbose=False)
                for lp_result in lp_results:
                    for lp_box in lp_result.boxes.cpu().numpy():
                        lp_cls = int(lp_box.cls[0])
                        lp_conf = lp_box.conf[0]

                        lp_found = lp_cls in [0] and conf > 0.65 #  # Classes: 0: vehicle registration plate, 1: vehicle
                        labels = {0: 'vehicle registration plate',  1: 'vehicle'}

                        if vehicle_found:
                            # Define box coordinates
                            r = lp_box.xyxy[0].astype(int) # -> r[0]=x1; r[1]=y1; r[2]=x2; r[3]=y2;
                            w = r[2] - r[0]
                            h = r[3] - r[1]
                            # Draw a rectangle around that found object if it´s fulfilles the requirements (vehicle with certainty >0.65)
                            cv2.rectangle(vehicle_frame, (r[0],r[1]), (r[2],r[3]), (255, 0, 0), 1) #r[:2], r[2:], (255, 255, 255), 2)
                            # Extract the region inside the rectangle
                            lp_frame = vehicle_frame[r[1]:r[3]+1, r[0]:r[2]+1]
                            # Save the extracted region to a PNG file with the current timestamp
                            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                            subfolder_path = "fotos"
                            filename = f"{subfolder_path}/license_plate_{current_time}.png"
                            cv2.imwrite(filename, lp_frame)

                
    
    cv2.imshow('video', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()