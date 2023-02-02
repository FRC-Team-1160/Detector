#!/usr/bin/env python3

from detector import Detector
from detector import Frame
import cv2
from speedster import load_model
import sys

# TEST FOR YOLO INFERENCE
detector = Detector("/Users/brianchen/Desktop/Detector/runs/detect/train5/weights/best.pt", gpu=False)
# detector = Detector("/Users/brianchen/Desktop/Detector/runs/detect/train2/weights/model_scripted.pt", gpu=False)
# model = load_model("/Users/brianchen/Desktop/Detector/runs/detect/train2/weights")
# detector = Detector(model_url=model)
frame = Frame(1, verticalFOV=47.5 )

while True:
	UP = '\033[1A'
	CLEAR = '\x1b[2K'
	img = frame.getFrame()
	# import pdb; pdb.set_trace()
	detector.detect(img)
	stream_height = frame.getStreamHeight()
	coor = detector.getCentroid()
	dis = detector.getDistance(frame=frame)
	if(len(coor) == 1):
		print(coor[0][2], ": (",coor[0][0], ", ", coor[0][1], ")", "    distance: ", dis[0],"	",detector.getExeTime(), end='\r')

		print(UP, end=CLEAR)
	else:
		UP = '\033[1A'
		for i in range(len(coor)):
			print(coor[i][2], ": (",coor[i][0], ", ", coor[i][1], ")", "    distance: ", dis[i], end="	")
		print(detector.getExeTime(), end='\r')
		print(UP, end=CLEAR)

	if cv2.waitKey(20) & 0xFF == ord('q'): # press 'd'to stop
		break
		