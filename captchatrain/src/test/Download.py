# -*- coding: UTF-8 -*-
#from preprocess import Process
import urllib
import os 
from utils import Constants
from utils import Util
import numpy as np
import cv2

def download():
    url = r"https://ssl.captcha.qq.com/getimage?aid=522205405&r=0.616197673836723&uin=3231232321@qq.com" 
    for i in range(0, 200):
        print("Download:" + str(i))
        path = r"../../img/download/" + str(i) + ".jpg"  
        data = urllib.urlopen(url).read()
        f = open(path, 'wb') 
        f.write(data)  
        f.close() 
    
def convertToBinary():
    dir_path = Constants.IMG_DIR_TENCENT_TRAIN + "other/"
    for lists in os.listdir(dir_path): 
        #print(lists)
        path = os.path.join(dir_path, lists) 
        if not os.path.isdir(path): 
            oriImg = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            img = Util.binaryzation(oriImg)
            cv2.imwrite(path, img)
def testRotate():
    oriImg = cv2.imread(Constants.IMG_DIR_TENCENT_PUZZLE + "mbha.jpg", 0)
    img = Util.binaryzation(oriImg)
    img = img[:,0:42]
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
    print(M)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow('img',dst)
    cv2.waitKey(0)
    

    
if __name__ == "__main__":
    #testRotate()
    #convertToBinary()
    download()
    #process = Process()
    #Process.readDir("../../tencent/test", Process.convertToGray)
    
    print("---Finished---")
