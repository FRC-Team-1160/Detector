# Detector
Python Package that detects objects on the field. 
## Install Anaconda (REQUIRED!)
Pytorch requires a conda environment to work, follow the installation guide [here](https://docs.anaconda.com/anaconda/install/)

Follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to create a conda environment

MAKE SURE TO ACTIVATE THE ENVIRONMENT!
## Install required packages
Make sure to install pip package manager [here](https://pip.pypa.io/en/stable/installation/).
```
pip3 install requirements.txt
```
## Usage
### Roboflow API Inference
Uses Roboflow's API to make predictions with trained model (trained on Roboflow)
```
import detect_API as detect
import cv2

result = detect.infer(img) # returns a np.ndarray
cv2.imshow(result)
```
### YOlOv8 Inference 
Using the model trained by yolov8 to make predictions 
```
import detect
import cv2

detect.detect('0') # Put the camera ID in single quotation mark (char)
```
### Using a different model
Change the path of line 12 in detect.py to the path of the model

## Training the model **(Google Colab or Nvidia GPU only!)**
1. Open Training/train_yolov8_object_detection.ipynb
2. Run each cell
3. BEFORE Custom Training, go to ChargedUp23 folder and change the path of train and val to the folder located in the ChargedUp23 folder
4. Suggest starting with 50 epochs, and change depends on the training error to prevent from overfitting
5. **SKIP** the cells under "deploy model on Roboflow"
6. The weight file `best.pt` can be found in the runs/train#/weights folder
