# -*- coding: UTF-8 -*-
'''
Created on 2014/2/14

@author: rogers
'''
import cv2
#from model import Model
from utils import Util

#设定字符宽度
charWidth = 3

'''
以(start, 0)为起点，做一次搜索
start就是起点的x坐标了，y坐标默认为0
'''
def searchOnce_old(img, path, startX, startY):
    height, width = img.shape
    #print("width:" + str(width) + ", height:" + str(height))
    '''
    1.从起点开始往下搜索，直到遇到黑点
    0是黑点，255是白点
    '''
    #print("startX:" + str(startX) + ", startY:" + str(startY))
    blackY = 0
    for j in range(startY, height):
        if img.item(j, startX) == 0:
            blackY = j
            break
    #print("blackY:" + str(blackY))
    searchRecursion_old(img, path, startX, blackY)
    
def searchRecursion_old(img, path, startX, startY):
    height, width = img.shape
    #print("startX:" + str(startX) + ", startY:" + str(startY))
    #检查是否搜索到尽头了
    if(startY >= height):
        return 
    '''
    2.从黑点忘左、右分别搜索，直到穿过白点再次遇到黑点
            默认就从start-2, start+2开始搜吧，这样就相当于碰到白点了，直接遇到黑点就停止了
    '''
    leftX = 0
    rightX = width - 1
    #向左搜索，记录第一次碰到黑点的距离leftX
    for i in range(startX - 1):
        if img.item(startY, startX - i) == 0:
            leftX = startX - i
            break
    #向右搜索，记录第一次碰到黑点的距离rightX'
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
        3.1 两边的距离均大于字符宽度，则下移，继续搜索下一个点
        3.2 两边都小于或等于字符宽度，说明此时字符有重叠，暂时先取两边距离的中点，加入路径，继续搜索下一个点
        3.3 只有其中一个距离小于或等于字符宽度，暂时作为正确的分割点，加入路径，下移
    '''
    leftDistance = startX - leftX
    rightDistance = rightX - startX
    #print("leftDistance:" + str(leftDistance) + ", rightDistance:" + str(rightDistance))
    #3.1 继续搜索下一个点
    if leftDistance > charWidth and rightDistance > charWidth:
        searchRecursion_old(img, path, startX, startY + 1)
    #3.2 
    elif leftDistance <= charWidth and rightDistance <= charWidth:
        newX = int((leftX + rightX) / 2 )
        path.append((newX, startY))
        searchRecursion_old(img, path, newX, startY + 1)
    else:
        path.append((startX, startY))
        searchRecursion_old(img, path, startX, startY + 1)


'''
以startX-endX为起点进行一次分割
'''
def spiltOne_old(img, startX):
    #for j in range(startX, endX):
    path = []
    path.append((startX, 0))
    img = ~Util.skeleton(img)
    searchOnce_old(img, path, startX, 0)
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
            paths.append(Util.fillPath(spiltOne_old(img, x), height))
        allPaths[i] = paths
    #printAllPath(allPaths)  
    return allPaths

if __name__ == "__main__":
    #oriImg = cv2.imread("1.jpg", 0)
    #img = binaryzation(oriImg)
    #getAllPaths(img)
    pass

    