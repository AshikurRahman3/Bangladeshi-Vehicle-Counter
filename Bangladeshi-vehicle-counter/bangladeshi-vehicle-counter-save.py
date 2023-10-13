import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# for video
cap = cv2.VideoCapture('../Videos/bangladesh_vehicles1.mp4')

# Retrieve the original video's FPS
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object with the same FPS
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
out = cv2.VideoWriter('bd-vehicle-counter-video1-large-0.9-1.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

model = YOLO('../Yolo-Weights/bd_vehicle_final_yolov8l.pt')

classNames = ['bus', 'rickshaw', 'motorbike', 'car', 'three wheelers (CNG)',
              'pickup', 'minivan', 'suv', 'van', 'taxi', 'truck', 'bicycle', 'policecar',
              'ambulance', 'human hauler', 'wheelbarrow', 'minibus', 'auto rickshaw',
              'army vehicle', 'scooter', 'garbagevan']

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [170, 400, 700, 400]
car_ids = []

while True:
    success, img = cap.read()

    if not success:
        break

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.9:
                cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1)), scale=2, thickness=2,
                                   offset=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    vehicle_type_count = {i: 0 for i in range(len(classNames))}

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        center_x, center_y = x1 + (w // 2), y1 + (h // 2)

        if id not in car_ids:
            car_ids.append(id)

            if id < len(classNames):
                vehicle_type_count[id] += 1

    total_vehicles = len(car_ids)
    total_count_text = f"Total Vehicles: {total_vehicles}"
    cv2.putText(img, total_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(img)  # Write the frame to the video

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
