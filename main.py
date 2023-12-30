import cv2
from ultralytics import YOLO
import numpy as np
import torch
from src.logger import logging
from src.exception import CustomException


model = YOLO('runs/detect/train9/weights/best.pt')

path = "path to video"
cap = cv2.VideoCapture(path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame,device='mps')
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(),dtype='int')
    classes = np.array(result.boxes.cls.cpu(),dtype='int')
    clas = ['without_mask','mask_weared_incorrect','with_mask']
    for bbox, cls in zip(bboxes, classes):
        (x,y,x2,y2) = bbox
        cv2.rectangle(frame,(x,y),(x2,y2), (0,0,225), 2)
        cv2.putText(frame, str(clas[cls]), (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
   

    cv2.imshow("img",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


