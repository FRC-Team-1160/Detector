from ultralytics import YOLO
import cv2
from imageai.Detection import ObjectDetection
import torch
import time
import numpy as np
from PIL import Image
from matplotlib import cm

# model = ObjectDetection()
# model.setModelTypeAsYOLOv3()
# model.setModelPath(r"../runs/detect/train/weights/best.pt")
# model.loadModel()
model = YOLO("../runs/detect/train2/weights/best.pt")

stream = cv2.VideoCapture(1)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

def detect(CAM_ID):
    while True:    
        ret, img = stream.read()   
        # img = cv2.imread(img)
        # print(type(img))
        # result = model.predict(source=img)
        # img = cv2.imread('/Users/brianchen/Desktop/Detector/Training/ChargedUp23-1/test/images/cone-0affa018-9080-11ed-a834-709cd1141cab_jpg.rf.0a113aa1bd9f8f5f630a777989b573a1.jpg')
        result = model.predict(source=img, show=True)
        print(getCentroid(result=result))
        # cv2.imshow("", result)
        if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
            break
    stream.release()
    cv2.destroyAllWindows()

def getCoor(result):
    obj = result[0]
    coor = obj.boxes.xyxy
    return coor

def getCentroid(result):
    coor = getCoor(result=result)
    if len(coor) == 0:
        return ('(0, 0)')
    else:
        x1 = coor[0][0]
        y1 = coor[0][1]
        x2 = coor[0][2]
        y2 = coor[0][3]
        centrX = x1 + (x2-x1)/2
        centrY = y1 + (y2-y1)/2
        return (f"({centrX}, {centrY})")


# DETECTING EVERY FRAME USING OPENCV 

# while True:    
#     ret, img = stream.read()   
#     # img = cv2.imread(img)
#     # print(type(img))
#     # result = model.predict(source=img)
#     result = model.predict(source=img, show=True)

#     for r in result:
#         boxes = r.boxes # Boxes object for bbox outputs
#         masks = r.masks # Masks object for segmenation masks outputs
#         probs = r.probs # Class probabilities for classification outputs

#     # convert to numpy arr
#     result = result.to("cpu")
#     result = result.numpy()

#     # result = np.array(result)
#     print(result)
#     # cv2.imshow("", result)
#     if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
#         break

# stream.release()
# cv2.destroyAllWindows()

