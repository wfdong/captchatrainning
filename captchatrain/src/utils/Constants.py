# -*- coding: UTF-8 -*-
'''
Created on 2014年2月23日

@author: rogers
'''


IMG_DIR = "../../img/"

IMG_WIDTH = 130

IMG_HEIGHT = 53

SEARCH_FROM_TOP = 't'

SEARCH_FROM_BOTTOM = 'b'

DATA_DIR = "../../data/"

SKL_DATA_FILE_NAME = DATA_DIR + "skllearnData.pkl"

IMG_DIR_TENCENT_TRAIN = IMG_DIR + "train/tencent/"
IMG_DIR_TENCENT_NEW_TRAIN = IMG_DIR + "train/newTrain/"
IMG_DIR_TENCENT_TEST = IMG_DIR + "test/tencent/"
IMG_DIR_TENCENT_PUZZLE = IMG_DIR + "puzzle/tencent/"

IMG_DIR_TENCENT_SPLIT = IMG_DIR + "train/split/"
IMG_DIR_TENCENT_SPLIT2 = IMG_DIR + "train/split2/"
IMG_DIR_TENCENT_SPLIT_ALL = IMG_DIR + "train/splitAll/"
IMG_DIR_TENCENT_SPLIT_ALL_SMALL = IMG_DIR + "train/small/"
IMG_DIR_TENCENT_WHOLE = IMG_DIR + "train/whole/"

IMG_DIR_TENCENT_TEST_SPLIT = IMG_DIR + "test/split/"
IMG_DIR_TENCENT_TEST_SPLIT_ALL = IMG_DIR + "test/splitAll/"
IMG_DIR_TENCENT_TEST_SPLIT_ALL_SMALL = IMG_DIR + "test/splitAllSmall/"
IMG_DIR_TENCENT_TEST_WHOLE = IMG_DIR + "test/whole/"

IMG_DIR_TENCENT_REG = IMG_DIR + "reg/"
IMG_DIR_TENCENT_REG_CORRECT = IMG_DIR + "reg/correct/"
IMG_DIR_TENCENT_REG_WRONG = IMG_DIR + "reg/wrong/"
IMG_DIR_TENCENT_REG_WRONG_OLD = IMG_DIR + "reg/wrong-old/"

IMG_DIR_TENCENT_REG_UPDATE = IMG_DIR + "reg/update/"

#就截取一个正方形好了，这样在旋转时说不定简单些
ONE_CHAR_WIDTH = 50

#高度是53，这里就取整数50了
ONE_CHAR_HEIGHT = ONE_CHAR_WIDTH

#进行机器学习时的图像大小，要从ONE_CHAR_WIDTH变到TRAIN_CHAR_WIDTH的
#中间的缩小和旋转感觉应该是不可避免的
TRAIN_CHAR_WIDTH = 20

TRAIN_CHAR_HEIGHT = TRAIN_CHAR_WIDTH


#黑色
BLACK = 0
#白色
WHITE = 255

#切分成小区域内的宽度。最好可以被ONE_CHAR_WIDTH整除的
REGION_WIDTH = 5
#height,最好和width一样
REGION_HEIGHT = 5

'''
相反的颜色
因为numpy在创建新数组时
'''
INVERT_COLOUR = 1

LEFT_ANGLE = -5

RIGHT_ANGLE = 5

SPLITLINE_1_RANGE = (23, 40)

SPLITLINE_2_RANGE = (43, 67)

SPLITLINE_3_RANGE = (70, 93)

SPILTLINE_RANGES = [(23, 46), (43, 75), (70, 93), (128, 129)]

SEARCH_ALL_POSSIBLE_RANGE = 5

