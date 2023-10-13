import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


# for video

cap = cv2.VideoCapture('../Videos/bangladesh_vehicles1.mp4')



model = YOLO('../Yolo-Weights/bd_vehicle_final_yolov8n.pt')



classNames = ['bus', 'rickshaw', 'motorbike', 'car', 'three wheelers (CNG)',
              'pickup', 'minivan', 'suv', 'van', 'taxi', 'truck', 'bicycle', 'policecar',
              'ambulance', 'human hauler', 'wheelbarrow', 'minibus', 'auto rickshaw',
              'army vehicle', 'scooter', 'garbagevan']




# tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [170,400,700,400]
car_ids = []
while True:
    success, img = cap.read()





    results = model(img,stream=True)

    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:


            # showing bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1


            # showing confidence values
            conf = math.ceil((box.conf[0] * 100)) / 100



            # showing class name
            cls = int(box.cls[0])  # class id of coco dataset
            currentClass = classNames[cls]



            if  conf > 0.7:
                cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1)), scale=2, thickness=2,
                                   offset=1)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=5,rt=2)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    vehicle_type_count = {i: 0 for i in range(len(classNames))}  # Initialize count for each vehicle type


    resultsTracker = tracker.update(detections)
    # cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2,id = int(x1), int(y1), int(x2), int(y2),int(id)
        print(result)
        w, h = x2 - x1, y2 - y1
        # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2,colorR=(255,0,0))
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
        #                    scale=2, thickness=3, offset=10)

        center_x,center_y = x1 + (w // 2) , y1 + (h // 2)


        if car_ids.count(id) == 0:
            car_ids.append(id)
            # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

            # Count the vehicle type
            if id < len(classNames):  # Check if the ID is within the range of classNames
                vehicle_type_count[id] += 1

    # Calculate the total number of detected vehicles
    total_vehicles = len(car_ids)



    # Display the total count of detected vehicles on top-left
    total_count_text = f"Total Vehicles: {total_vehicles}"
    cv2.putText(img, total_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    cv2.imshow('Image',img)
    cv2.waitKey(1)