# -*- coding: UTF-8 -*-
'''
Created on 2014/3/16

@author: rogers
'''
from utils import Util
from utils import Constants
from spilt import SpiltUseCV2
from spilt import SpiltUtil
import numpy as np
import cv2
import os 

img = cv2.imread(Constants.IMG_DIR_TENCENT_TRAIN + "AANV_39b_58b_82b.jpg")
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)
for k in kp:
    print(k)

img=cv2.drawKeypoints(gray,kp)
cv2.imwrite(Constants.IMG_DIR_TENCENT_TRAIN + 'sift_keypoints.jpg',img)