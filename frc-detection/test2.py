from detector import Detector
from detector import Frame
import cv2

# TEST FOR YOLO INFERENCE
detector = Detector("../runs/detect/train2/weights/best.pt")
frame = Frame(1)

while True:
    img = frame.getFrame()
    detector.detect(img)
    coor = detector.getCentroid()
    if(len(coor) == 1):
        print(coor[0][2], ": (",coor[0][0], ", ", coor[0][1], ")", end="    ")
    else:
        for i in range(len(coor)):
            print(coor[i][2], ": (",coor[i][0], ", ", coor[i][1], ")", end="   ")
    print(detector.getExeTime())