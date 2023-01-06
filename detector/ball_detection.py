# load config
import json
with open('detector/roboflow_config_webcam.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

from roboflow import Roboflow
import cv2 as cv
import base64
import numpy as np
import requests



upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])

def infer(img):
    # Get the current image from the webcam
    # ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    # Parse result image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

