from ultralytics import YOLO
import cv2
import cvzone
import math

# for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# for video
cap = cv2.VideoCapture('../Videos/bangladesh_vehicles2.mp4')


# model = YOLO('best.pt')
model = YOLO('../Yolo-Weights/bangladesh_vehicles_yolov8l.pt')

classNames = ['bus', 'rickshaw', 'motorbike', 'car', 'three wheelers (CNG)', 'pickup', 'minivan', 'suv', 'van', 'taxi', 'truck', 'bicycle', 'policecar', 'ambulance', 'human hauler', 'wheelbarrow', 'minibus', 'auto rickshaw', 'army vehicle', 'scooter', 'garbagevan']
while True:
    success, img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # alternative way
            # x1,y1,x2,y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            # showing bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h))

            # showing confidence values
            conf = math.ceil((box.conf[0] * 100)) / 100

            # as showing class name also, dont need below code
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

            # showing class name
            cls = int(box.cls[0])  # class id of coco dataset
            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=1,thickness=2)
            cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                               offset=1)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
    cv2.imshow('Image',img)
    cv2.waitKey(0)