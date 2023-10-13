import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


# for video
cap = cv2.VideoCapture('../Videos/cars.mp4')


model = YOLO('../Yolo-Weights/yolov8l.pt')





classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('mask3.png')



# tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [170,400,700,400]
car_ids = []

if (cap.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))


size = (frame_width, frame_height)
mask = cv2.resize(mask, (frame_width, frame_height))

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
out = cv2.VideoWriter('video_output_car_counter_l.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, size)

while True:
    success, img = cap.read()

    # Check if the frame was read successfully
    if not success:
        break

    # Make sure the mask image has the same dimensions as the frame
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread('car_graphics.png',cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(0,0))

    results = model(imgRegion,stream=True)

    detections = np.empty((0,5))
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


            # showing confidence values
            conf = math.ceil((box.conf[0] * 100)) / 100

            # as showing class name also, dont need below code
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

            # showing class name
            cls = int(box.cls[0])  # class id of coco dataset
            currentClass = classNames[cls]


            if currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus'\
                or currentClass == 'motorbike' and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1)), scale=2, thickness=2,
                                   offset=1)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=5,rt=2)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2,id = int(x1), int(y1), int(x2), int(y2),int(id)

        w, h = x2 - x1, y2 - y1
        # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2,colorR=(255,0,0))
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # cvzone.putTextRect(img, f' {classNames[int(id)]}', (max(0, x1), max(35, y1)),
        #                    scale=2, thickness=3, offset=10)

        center_x,center_y = x1 + (w // 2) , y1 + (h // 2)
        cv2.circle(img,(center_x,center_y),5,(255,0,255),cv2.FILLED)

        if limits[0]  < center_x < limits[2] and (limits[1] - 15) < center_y < (limits[1] + 15 ):
            if car_ids.count(id) == 0:
                car_ids.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)


    # cvzone.putTextRect(img, f'Cars: {len(car_ids)}',(50,50))
    cv2.putText(img,str(len(car_ids)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    # cv2.imshow('Image', img)
    # # cv2.imshow('ImageRegion',imgRegion)
    # cv2.waitKey(0)

    # Write the frame with annotations to the video file
    out.write(img)

# When everything done, release
# the video capture and video
# write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
#
print("The video was successfully saved")