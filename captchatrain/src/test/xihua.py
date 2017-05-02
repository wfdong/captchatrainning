# -*- coding: UTF-8 -*-
'''
Created on 2014/3/9

@author: rogers
'''
from utils import Constants
from utils import Util
import numpy as np
import cv2
from skimage import morphology
import os 

def xihua():
    for lists in os.listdir(Constants.IMG_DIR_TENCENT_TRAIN): 
        #print(lists)
        path = os.path.join(Constants.IMG_DIR_TENCENT_TRAIN, lists) 
        if not os.path.isdir(path): 
            print(path)
            oriImg = cv2.imread(path, 0)
            img = Util.binaryzation(oriImg)
            img = Util.erasePaddingInOneChar(img)
            #cv2.imshow("ori", img)
            #height, width = img.shape
            img = ~Util.skeleton(~img)
           
            cv2.imwrite(path, img)
            #ret.append(readOneImg(path))
    pass

'''
找图像细化后的分支点，这些点的特征是这样的
0 0 0    0 0 0   0 1 0
1 1 1 或     1 1 1 或   0 1 1
0 1 0    1 0 0   1 0 0
基本特点是有4个1，中间的1是原点，其他的都作为一个分支，这样就代表一共有三个分支了
'''
def findXihuaPoint(img):
    pass
#膨胀图像
def dilate(img):
    #OpenCV定义的结构元素  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
    #膨胀图像  
    dilated = cv2.dilate(img, kernel)  
    return dilated

if __name__ == "__main__":
    #xihua()
    oriImg = cv2.imread(Constants.IMG_DIR_TENCENT_TRAIN + "AANV_39b_58b_82b.jpg", 0)
    img = Util.binaryzation(oriImg)
    #img = dilate(~img)
    img = Util.erasePaddingInOneChar(img)
    #img = Util.binaryzation(oriImg)
    #img = ~Util.skeleton(~img)
           
    cv2.imwrite(Constants.IMG_DIR_TENCENT_TRAIN + "1.jpg", img)
    
    '''img = cv2.imread(Constants.IMG_DIR_TENCENT_PUZZLE + "mbha.jpg")
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)[1]
    im[im == 255] = 1
    im = morphology.skeletonize(im)
    im = im.astype(np.uint8)
    im[im == 1] = 255
   
    cv2.imshow("Image1", im)
    cv2.imshow("Image2", img)
    im[im == 255] = 1
    Util.printCV2Image(im)
    cv2.waitKey (0)'''
