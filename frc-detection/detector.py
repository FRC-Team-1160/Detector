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
#model = YOLO("../runs/detect/train2/weights/best.pt")

#the detector object
class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.result = None
        self.status = False
        self.label_mapping = {0: "cone", 1: "cube"}
        self.exeTime = 0;
    
    #CONFIG METHODS

    #DETECTION METHODS

    #runs the detector
    def detect(self, img):

        self.status = True
        # stream = cv2.VideoCapture(CAM_ID)
        # stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.streamWidth)
        # stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.streamHeight)
        # while True:
            # ret, img = stream.read()   
            # img = cv2.imread(img)
            # print(type(img))
            # result = model.predict(source=img)
            # img = cv2.imread('/Users/brianchen/Desktop/Detector/Training/ChargedUp23-1/test/images/cone-0affa018-9080-11ed-a834-709cd1141cab_jpg.rf.0a113aa1bd9f8f5f630a777989b573a1.jpg')
        startTime = round(time.time()*1000)
        self.result = self.model.predict(source=img, show=True)
        endTime = round(time.time()*1000)
        self.exeTime = endTime - startTime
            #print(getCentroid(result=result))
            # cv2.imshow("", result)
        #     if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27) or (not self.status):
        #         break
        # stream.release()
        # cv2.destroyAllWindows()
    
    #stops the detector
    def stop(self):
        self.status = False

    #INFORMATION METHODS

    #returns true if the detector is running and false if it is not
    def isActive(self):
        return self.status

    #returns the coordinates of detected objects
    def getCoor(self):
        if not self.status:
            raise Exception("Can only retrieve information if detection is active")
        obj = self.result[0]
        coor = obj.boxes.xyxy

        # Getting object type
        if len(obj.boxes.cls) == 0:
            obj_type = None
        else:
            obj_type = self.label_mapping[int((obj.boxes.cls)[0])]
        return coor, obj_type

    #returns the centroid of the first detected object
    def getCentroid(self):
        if not self.status:
            raise Exception("Can only retrieve information if detection is active")
        coor, obj_type = self.getCoor()
        if len(coor) == 0:
            return [[0,0, None]]
        elif len(coor) == 1:
            x1 = coor[0][0]
            y1 = coor[0][1]
            x2 = coor[0][2]
            y2 = coor[0][3]
            centrX = int(x1 + (x2-x1)/2)
            centrY = int(y1 + (y2-y1)/2)
            return [[centrX, centrY, obj_type]]
        else:
            result = []
            for i in range(0, len(coor)):
                x1 = coor[i][0]
                y1 = coor[i][1]
                x2 = coor[i][2]
                y2 = coor[i][3]
                centrX = int(x1 + (x2-x1)/2)
                centrY = int(y1 + (y2-y1)/2)
                obj_type = self.label_mapping[int((self.result[0].boxes.cls)[i])]
                result.append([centrX, centrY, obj_type])
            return result

    def getExeTime(self):
        return self.exeTime


    
class Frame:
    def __init__(self, CAM_ID) -> None:
        # check if CAM_ID is an integer
        if not isinstance(CAM_ID, int): 
            raise Exception("CAM_ID must be an integer")

        self.stream = cv2.VideoCapture(CAM_ID)
        self.streamWidth = 640
        self.streamHeight = 640

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.streamWidth)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.streamHeight)
    
    def getFrame(self):
        ret, frame = self.stream.read()
        return frame

    # CONFIG METHODS
    
    #sets the size of the stream window. default window is 640x640
    def setWindow(self, width=640, height=640):
        if self.status:
            raise Exception("Cannot change the window while detection is active")
        self.streamWidth = width
        self.streamHeight = height
    
    

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
