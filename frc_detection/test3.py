#!/usr/bin/env python3

from detector import Detector
from detector import Frame
import cv2
import sys
from networktables import NetworkTables
import time

# TEST FOR YOLO INFERENCE
detector = Detector("/home/titanium1160/Detector/runs/detect/train7/weights/best.pt", gpu=True)
# detector = Detector("/Users/brianchen/Desktop/Detector/runs/detect/train2/weights/model_scripted.pt", gpu=False)
# model = load_model("/Users/brianchen/Desktop/Detector/runs/detect/train2/weights")
# detector = Detector(model_url=model)
frame = Frame(0, verticalFOV=47.5 )
NetworkTables.initialize(server="10.11.60.2")
table = NetworkTables.getTable("vision")

while True:
	UP = '\033[1A'
	CLEAR = '\x1b[2K'
	img = frame.getFrame()
	# import pdb; pdb.set_trace()
	img1 = detector.detect(img)
	#stream_height = frame.getStreamHeight()
	coor = detector.getCentroid()
	ofst, dis = detector.getDistance(frame=frame)
	table.putNumber("offset", ofst)
	table.putNumber("distance", dis)
	table.putNumber("execTime", detector.getExeTime())
 
	#frames to table
	#encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
	#_, jpg_frame = cv2.imencode('.jpg', img1, encode_param)

    # convert jpg frame to bytearray and send it to network table
	#jpg_bytes = bytearray(jpg_frame)
	#table.putValue('frame', jpg_bytes)
 
	if(len(coor) == 1):
		print(coor[0][2], ": (",ofst, ", ", dis, ")", detector.getExeTime())
		print(coor[0][2], ": (",table.getNumber("offset", -1), ", ", table.getNumber("distance", -1), ")")
		#print("Number of Connections ", len(NetworkTables.getConnections()))
	if cv2.waitKey(20) & 0xFF == ord('q'): # press 'd'to stop
		break
		
