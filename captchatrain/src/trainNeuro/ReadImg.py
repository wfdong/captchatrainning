# -*- coding: UTF-8 -*-
'''
Created on 2014/2/23

@author: rogers
'''
from utils import Util
from utils import Constants
from spilt import SpiltUseCV2
from spilt import SpiltUtil
import numpy as np
import cv2
import os 



def readAllImg(dir_path):
    ret = []
    labels = []
    for lists in os.listdir(dir_path): 
        #print(lists)
        path = os.path.join(dir_path, lists) 
        if not os.path.isdir(path): 
            labels.extend(getLables(path))
            ret.append(readOneImg(path))
    #target = readOneImg(dir_path + "YbUB_39_68_85.jpg")
    return ret, getLablesAfterWrap(ret, labels)

def getLablesAfterWrap(ret, labels):
    new_labels = []
    wrapCount = len(ret[0]) / 4
    for label in labels:
        for i in range(wrapCount):
            new_labels.append(label)
    return new_labels
        

def getLables(path):
    img_name = path[path.rindex('/') + 1:]
    labels = img_name.split('_')[0]
    return [(ord)(labels[0])-ord('A'), (ord)(labels[1])-ord('A'), (ord)(labels[2])-ord('A'), (ord)(labels[3])-ord('A')]

def getSpiltPoint(img_name):
    img_name = img_name[:img_name.index('.')]
    points = img_name.split('_')
    #print(points)
    spilts = []
    searchDirection = []
    print(img_name)
    if len(points[1]) == 3:
        spilts.append(int(points[1][0:2]))
        searchDirection.append(points[1][2])
    else:
        spilts.append(int(points[1][0:2]))
        searchDirection.append(Constants.SEARCH_FROM_TOP)
        
    if len(points[2]) == 3:
        spilts.append(int(points[2][0:2]))
        searchDirection.append(points[2][2])
    else:
        spilts.append(int(points[2][0:2]))
        searchDirection.append(Constants.SEARCH_FROM_TOP)
    if len(points[3]) == 3:
        spilts.append(int(points[3][0:2]))
        searchDirection.append(points[3][2])
    else:
        spilts.append(int(points[3][0:2]))
        searchDirection.append(Constants.SEARCH_FROM_TOP)
    
    return spilts, searchDirection


#这里返回的仍然是图像的二维矩阵，如果要处理就到后面的函数去处理
'''
这个函数会读取一个图像，然后把这个图像拆分成四部分，依据其文件名定义的分割点
然后再把切分成的四部分转换成train所需的20*20大小的图片
'''
def readOneImg(path):
    oriImg = cv2.imread(path, 0)
    img = Util.binaryzation(oriImg)
    img = Util.erasePaddingInOneChar(img)
    height, width = img.shape
    #获取指定的三个分割点, 再根据分割点做分割线
    points, directions = getSpiltPoint(path[path.rindex('/'):])
    
    path1 = Util.fillPath(SpiltUtil.spiltOne(img, points[0], directions[0], 0), height)
    path2 = Util.fillPath(SpiltUtil.spiltOne(img, points[1], directions[1], 1), height)
    path3 = Util.fillPath(SpiltUtil.spiltOne(img, points[2], directions[2], 2), height)
    path0 = Util.getLeftBorder(height)
    path4 = Util.getRightBoder(width, height)
    ret = []
    
    ret.extend(Util.conventToTrainCharFromAllWrap(getOneChar(img, path0, path1)))
    ret.extend(Util.conventToTrainCharFromAllWrap(getOneChar(img, path1, path2)))
    ret.extend(Util.conventToTrainCharFromAllWrap(getOneChar(img, path2, path3)))
    ret.extend(Util.conventToTrainCharFromAllWrap(getOneChar(img, path3, path4)))
    return ret
    
'''
返回分割线path1和path2之间的字符
返回的字符要进行两项处理
1.Normalization， width = Constants.ONE_CHAR_WIDTH, height = Constants.ONE_CHAR_HEIGHT
2.对于小字符来说，尽量往左上角靠拢，即左边界要有一个黑点，上边界也要有一个黑点
'''
def getOneChar(img, path1, path2):
    print("path1:" + str(path1))
    print("path2:" + str(path2))
    target = np.zeros((Constants.ONE_CHAR_HEIGHT, Constants.ONE_CHAR_WIDTH), dtype=np.int8)
    #print(target.shape)
    #Util.printCV2Image(target)
    targetX = 0
    targetY = 0
    
    mostLeftX = path1[0][0]
    for p in path1:
        if p[0] < mostLeftX:
            mostLeftX = p[0]
    print("mostLeftX=" + str(mostLeftX))
    #以纵坐标作为外层循环
    for y in range(Constants.ONE_CHAR_HEIGHT):
        targetX = 0
        #遍历两个path之间的x坐标
        for x in range(mostLeftX, mostLeftX + min((path2[y][0] - mostLeftX + 1), Constants.ONE_CHAR_WIDTH)):
            if(img.item(y, x) == 0 and x >= path1[y][0]):
                target[y, targetX] = Constants.INVERT_COLOUR
            targetX += 1
    #Util.printCV2Image(target)
    return moveToTopLeft(target)

'''
把取出来的图像移到最左上角去，以不损失像素为基础
先做X和Y轴的投影，其实就是计算每个坐标上1的个数，之后判断连续两个有1的地方就是起点
'''
def moveToTopLeft(img):
    ret = []
    height, width = img.shape
    moveY = 0  
    moveX = 0
    xp = Util.get_X_Projection(img)
    yp = Util.get_Y_Projection(img)
    for i in range(len(xp) - 1):
        if(xp[i] > 0 and xp[i + 1] > 0):
            moveY = i
            break
    for i in range(len(yp) - 1):
        if(yp[i] > 0 and yp[i + 1] > 0):
            moveX = i
            break  
    #print(moveX)
    #print(moveY)
    target = np.zeros((height, width), dtype=np.int8)        
    for y in range(moveY, height):
        for x in range(moveX, width):
            target[y - moveY, x - moveX] = img.item(y, x)
    #Util.printCV2Image(target) 
    '''
             在这里进行wrap变换应该是最佳的
    '''
    target = Util.copyOneImg(target, (height, width))
    return Util.wrap(target)
            
            
#膨胀图像
def dilate(img, element_width):
    #OpenCV定义的结构元素  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(element_width, element_width))  
    #膨胀图像  
    dilated = cv2.dilate(~img, kernel)  
    return ~dilated

if __name__ == "__main__":
    
    fileName = Constants.IMG_DIR_TENCENT_TRAIN + "BSmP_39_54_81.jpg"
    readOneImg(fileName)
    oriImg = cv2.imread(fileName, 0)
    img = Util.binaryzation(oriImg)
    #img = Util.erasePaddingInOneChar(img)
    img = dilate(img, 7)
    img = ~Util.skeleton(img)
    cv2.imwrite(Constants.IMG_DIR_TENCENT_TRAIN + "ske2.jpg", img)
    #cv2.imshow("ske1",img)
    '''path = Constants.IMG_DIR_TENCENT_TRAIN + "VmnZ_31_59_84.jpg"
    points, directions = getSpiltPoint(path[path.rindex('/'):])
    print(points)
    print(directions)
    print(directions[0] == Constants.SEARCH_FROM_TOP)'''
    cv2.waitKey(0)