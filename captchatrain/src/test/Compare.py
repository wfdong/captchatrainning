# -*- coding: UTF-8 -*-
'''
Created on 2014/3/15

@author: rogers
'''
import cv2
import numpy as np
from utils import Constants
from utils import Util
from trainNeuro import ReadImg
from trainNeuro import TrainUseSVM
from spilt import SpiltUseCV2
from spilt import SpiltUseBackground
from spilt import SpiltUtil
import Main

fileName = Constants.IMG_DIR_TENCENT_TRAIN + "SPPV_32_63_93.jpg"

def spiltInTrain():
    ret = ReadImg.readOneImg(fileName)
    return ret

def spiltInReal():
    train_new = False
    svm = Main.getSVM(train_new)
    Main.readInput(svm, fileName)

if __name__ == "__main__":
    spiltInTrain()
    spiltInReal()
