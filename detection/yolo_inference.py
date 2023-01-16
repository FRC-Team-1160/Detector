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
model = YOLO("../runs/detect/train/weights/best.pt")

stream = cv2.VideoCapture(0)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# DETECTING EVERY FRAME USING OPENCV 

# while True:    
#     ret, img = stream.read()   
#     annotated_image, preds = model.detectObjectsFromImage(input_image=img,
#                     input_type="array",
#                       output_type="array",
#                       display_percentage_probability=False,
#                       display_object_name=True)

#     cv2.imshow("", annotated_image)     
    
#     if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
#         break

# stream.release()
# cv2.destroyAllWindows()

result = model.predict(source="0", show=True)