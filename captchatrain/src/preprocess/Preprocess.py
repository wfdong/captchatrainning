# -*- coding: UTF-8 -*-
'''
Created on 2014/2/17

@author: dada
'''
import cv2
from utils import Constants
from utils import Util
import numpy as np
#from PIL import Image

#腐蚀图像
def erode(img):
    #OpenCV定义的结构元素  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
    #腐蚀图像  
    eroded = cv2.erode(img,kernel)    
    return eroded

#膨胀图像
def dilate(img):
    #OpenCV定义的结构元素  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
    #膨胀图像  
    dilated = cv2.dilate(img, kernel)  
    return dilated

'''
消除一个字母的内部填充，因为腾讯的验证码有的可能会空心，但有的会内部全填充为黑色，这时就要对填充后的做处理了
处理的方式也很简单，就是
1.腐蚀，可以用稍微大一点的腐蚀矩阵
2.原图-腐蚀后的图，就OK了
'''
def erasePaddingInOneChar(img):
    #OpenCV定义的结构元素  
    #MORPH_CROSS， MORPH_RECT， MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
    #腐蚀图像  
    eroded = cv2.erode(~img, kernel) 
    height, width = img.shape
    #print(height * width - np.count_nonzero(eroded))
    #if()
    cv2.imshow("Image2", ~eroded)
    Util.printCV2Image(~eroded)
    
    return ~(~img - eroded)

if __name__ == "__main__":
    
    i = 0
    #oriImg = orderJpgs(jpgs, 38, 67, 98)
    #if we want show a img, must add this
    cv2.waitKey (0)