# saving video instead of direct real time showing

from ultralytics import YOLO
import cv2
import cvzone
import math

# for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

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
              ]  # Your list of class names here

# Get video details (frames per second, frame width, frame height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_file_path = 'runs/detect/predict/output_video_cars_l.mp4'  # Specify the path to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # ... Rest of the code for displaying bounding boxes and class names
            # showing bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            # showing confidence values
            conf = math.ceil((box.conf[0] * 100)) / 100

            # as showing class name also, dont need below code
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

            # showing class name
            cls = int(box.cls[0])  # class id of coco dataset
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=1,thickness=2)

    # Write the frame with annotations to the video file
    out.write(img)

cap.release()
out.release()
cv2.destroyAllWindows()
