import cv2
import imutils
import numpy as np
from cv2 import *
import os 

cam = cv2.VideoCapture(0) 
result, image = cam.read() 

filename = "test.jpg"

if result: 
    cv2.imwrite("camoutput.png", image) 

