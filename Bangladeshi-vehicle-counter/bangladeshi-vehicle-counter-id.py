import numpy as np
from ultralytics import YOLO
import cv2
import math
from collections import OrderedDict

# for video
cap = cv2.VideoCapture('../Videos/bangladesh_vehicles1.mp4')

model = YOLO('../Yolo-Weights/bd_vehicle_final_yolov8n.pt')

classNames = ['bus', 'rickshaw', 'motorbike', 'car', 'three wheelers (CNG)',
              'pickup', 'minivan', 'suv', 'van', 'taxi', 'truck', 'bicycle', 'policecar',
              'ambulance', 'human hauler', 'wheelbarrow', 'minibus', 'auto rickshaw',
              'army vehicle', 'scooter', 'garbagevan']

# Custom object tracking using dictionaries to store object IDs
object_dict = OrderedDict()  # Use OrderedDict to maintain order
next_object_id = 1  # Initialize the next available object ID

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

            if conf > 0.8:
                # Draw bounding box with better styling
                color = (255, 0, 255)  # Purple color
                thickness = 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                # Add class label to the bounding box
                label = ""
                matching_obj_id = None
                matching_obj_distance = None

                for obj_id, obj_data in object_dict.items():
                    prev_x1, prev_y1, prev_x2, prev_y2, _, _ = obj_data
                    prev_centroid_x, prev_centroid_y = (prev_x1 + prev_x2) / 2, (prev_y1 + prev_y2) / 2

                    # Calculate Euclidean distance from the current centroid to the previous centroid
                    distance = np.linalg.norm([x1 + (x2 - x1) / 2 - prev_centroid_x, y1 + (y2 - y1) / 2 - prev_centroid_y])

                    if distance < 20:  # If the object is close to an existing one, use its ID
                        matching_obj_id = obj_id
                        matching_obj_distance = distance
                        break

                if matching_obj_id is not None:
                    # Update the position of the existing object
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    object_dict[matching_obj_id] = [x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2]
                    label = f'ID: {matching_obj_id} ({currentClass})'
                else:
                    # Create a new object entry in the dictionary
                    object_dict[next_object_id] = [x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2]
                    label = f'ID: {next_object_id} ({currentClass})'
                    next_object_id += 1

                # Draw the label on the bounding box
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                label_x = x1 + 5
                label_y = y1 - 5
                cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Count the total number of detected vehicles
    total_vehicles = len(object_dict)

    # Display the total count of detected vehicles on top-left
    total_count_text = f"Total Vehicles: {total_vehicles}"
    cv2.putText(img, total_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
