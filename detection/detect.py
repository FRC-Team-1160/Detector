from ultralytics import YOLO
import cv2
from imageai.Detection import ObjectDetection
import torch
import time
import numpy as np

# model = ObjectDetection()
# model.setModelTypeAsYOLOv3()
# model.setModelPath(r"../runs/detect/train/weights/best.pt")
# model.loadModel()
model = YOLO("../runs/detect/train2/weights/best.pt")

stream = cv2.VideoCapture(1)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

def detect(CAM_ID):
    result = model.predict(source=CAM_ID, show=True)

# DETECTING EVERY FRAME USING OPENCV 

# while True:    
#     ret, img = stream.read()   
#     # img = cv2.imread(img)
#     # print(type(img))
#     # img = np.array(img)
#     # result = model.predict(source=img)
#     result = model(source=img, return_outputs=True, show=True)

#     cv2.imshow("", result)     
    
#     if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
#         break

# stream.release()
# cv2.destroyAllWindows()

