# -*- coding: UTF-8 -*-
'''
Created on 2014/3/10

@author: rogers
'''
import cv2
#from model import Model
from utils import Util
from utils import Constants
from trainNeuro import ReadImg
import numpy as np

#设定字符宽度
charWidth = 3

'''
以(start, height - 1)为起点，从底向上做一次搜索
start就是起点的x坐标了，y坐标默认为height - 1
'''
def searchOnce_old(img, path, startX, startY):
    height, width = img.shape
    '''
    1.从起点开始往下搜索，直到遇到黑点
    0是黑点，255是白点
    '''
    #print("startX:" + str(startX) + ", startY:" + str(startY))
    blackY = 0
    for j in range(startY, 0, -1):
        if img.item(j, startX) == 0:
            blackY = j
            break
    #print("blackY:" + str(blackY))
    #这里要从第一个黑点开始递归搜索了
    searchRecursion_old(img, path, startX, blackY)
    
def searchRecursion_old(img, path, startX, startY):
    height, width = img.shape
    #print("startX:" + str(startX) + ", startY:" + str(startY))
    #检查是否搜索到尽头了
    if(startY <= 0):
        return 
    '''
    2.从黑点忘左、右分别按水平方向搜索，直到穿过白点再次遇到黑点
            默认就从start-2, start+2开始搜吧，这样就相当于碰到白点了，直接遇到黑点就停止了
    '''
    leftX = 0
    rightX = width - 1
    #向左搜索，记录第一次碰到黑点的距离leftX
    for i in range(startX - 1, 0, -1):
        if img.item(startY, i) == 0:
            leftX = i
            break
    #向右搜索，记录第一次碰到黑点的距离rightX
    for i in range(0, width - startX - 1):
        #print("x:" + str(i + startX + 1) + "y:" + str(startY) + "pixel:" + str(img.getpixel(((i + startX + 1), startY))))
        if img.item(startY, i + startX + 1) == 0:
            rightX = i + startX + 1
            break
    #if rightX == 0:
    #    rightX = width - 1
    #print("leftX:" + str(leftX) + ", rightX:" + str(rightX))
    '''
    3.保存左和右搜索的距离，进行判断，有三种情况
        3.1 两边的距离均大于字符宽度，则上移，继续搜索下一个点
        3.2 两边都小于或等于字符宽度，说明此时字符有重叠，暂时先取两边距离的中点，加入路径，继续搜索下一个点
        3.3 左边的距离较小，暂时作为正确的分割点，加入路径，下移
        3.4 右边的距离较小，把右边的加入路径
    '''
    leftDistance = startX - leftX
    rightDistance = rightX - startX
    #print("leftDistance:" + str(leftDistance) + ", rightDistance:" + str(rightDistance))
    #3.1 继续搜索下一个点
    if leftDistance > charWidth and rightDistance > charWidth:
        searchRecursion_old(img, path, startX, startY - 1)
    #3.2 
    elif leftDistance <= charWidth and rightDistance <= charWidth:
        newX = int((leftX + rightX) / 2 )
        path.append((newX, startY))
        searchRecursion_old(img, path, newX, startY - 1)
    elif leftDistance <= charWidth and rightDistance >= charWidth:
        path.append((leftX, startY))
        searchRecursion_old(img, path, leftX, startY - 1)
    elif leftDistance >= charWidth and rightDistance <= charWidth:
        path.append((rightX, startY))
        searchRecursion_old(img, path, rightX, startY - 1)

'''
startX是起点
'''
def spiltOne(img, startX):
    #for j in range(startX, endX):
    height, width = img.shape
    path = []
    path.append((startX, height - 1))
    searchOnce_old(img, path, startX, height - 1)
    #print(path)
    #for i in range(len(path.path) - 1):
    #    cv2.line(img, path.paths[i].toTuple(), path.paths[i - 1].toTuple(), 0)
    #cv2.imshow("Image", img) 
    #if we want show a img, must add this
    #cv2.waitKey (0)
    return path

def printAllPath(allPaths):
    for i in range(2):
       # for path in allPaths.all_path[i]:
       print(allPaths[i])
       
allPaths={}
paths=[]
'''
得到所以分割线集合,img是二值化后的分割线集合
'''
def getAllPaths(img):
    #四个字符，三条分割线
    startPoint = {0:[20,21], 1:[45,46], 2:[68,69]}
    height, width = img.shape
    for i in range(2):
        paths = []
        for x in startPoint[i]:
            paths.append(Util.fillPath(spiltOne(img, x), height))
        allPaths[i] = paths
    #printAllPath(allPaths)  
    return allPaths



'''
之前写的是从上往下搜索的，鉴于很多字符在上面粘连的太紧密
反而从下面搜索效果貌似好一些，所以这个版本就写从下面搜索的
处理细化后的二值图像
'''
def searchFromBottom():
    pass

'''
找图像细化后的分支点，这些点的特征是这样的
0 0 0    0 0 0   0 1 0
1 1 1 或     1 1 1 或   0 1 1
0 1 0    1 0 0   1 0 0
基本特点是有4个1，中间的1是原点，其他的都作为一个分支，这样就代表一共有三个分支了
'''
def findBranchPoint(img):
    height, width = img.shape
    point = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            #y, x是中心点的坐标
            sum = img.item(y - 1, x - 1) + img.item(y - 1,x) + img.item(y - 1,x + 1)
            sum += img.item(y, x - 1) + img.item(y,x) + img.item(y,x + 1)
            sum += img.item(y + 1, x - 1) + img.item(y + 1,x) + img.item(y + 1, x + 1)
            if sum >= 4:
                point.append((x, y))
    return point

#膨胀图像
def dilate(img, element_width):
    #OpenCV定义的结构元素  
    #element_width = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(element_width, element_width))  
    #膨胀图像  
    dilated = cv2.dilate(~img, kernel)  
    return ~dilated

if __name__ == "__main__":
    #这个图像B的下方分割点是88，如果能正确把B分出来就OK
    oriImg = cv2.imread(Constants.IMG_DIR_TENCENT_TRAIN + "UPUS_34b_45b_77t.jpg", 0)
    img = Util.binaryzation(oriImg)
    cv2.imshow("ori", img)
    height, width = img.shape
    #img = dilate(img, 6)
    
    img = ~Util.skeleton(~img)
    cv2.imshow("ske", img)
    #img[img == 0] = 1
    #img[img == 255] = 0
    '''Util.printCV2Image(img)
    print(findBranchPoint(img))
    #我们只处理细化后的，这样不会收到多余字符的干扰
    
    cv2.imwrite(Constants.IMG_DIR_TENCENT_TRAIN + "ske.jpg", img)'''
    '''print(spiltOne(img, 83))
    path = Util.fillPath(spiltOne(img, 83), height)
    path4 = Util.getRightBoder(width, height)
    img = ReadImg.getOneChar(img, path, path4)
    print(type(img))
    img = img.astype(np.int8)
    #img[img == 1] = 128
    Util.printCV2Image(img)
    #print(path)
    cv2.imshow("ret", img)'''
    #getAllPaths(img)
    cv2.waitKey(0)

    