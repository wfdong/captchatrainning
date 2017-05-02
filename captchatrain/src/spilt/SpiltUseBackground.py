# -*- coding: UTF-8 -*-
'''
Created on 2014/3/12

@author: rogers
'''
import cv2
#from model import Model
from utils import Util
from utils import Constants
from trainNeuro import ReadImg
import numpy as np
import SpiltUtil
import os 

charWidth = 3

#膨胀图像
def dilate(img, element_width):
    #OpenCV定义的结构元素  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(element_width, element_width))  
    #膨胀图像  
    dilated = cv2.dilate(~img, kernel)  
    return ~dilated

'''
对根据corner point选出的paths进行filter，即过滤掉一些不合适的，依据有
1、长度  >= 4
2.斜率  > 1/2
3.path的最后一个点 < 10或  > 40
'''
def filterPathsAndPoints(points, paths):
    new_path = []
    new_points = []
    index = 0
    if len(points) < 5:
        return points, paths
    for path in paths:
        if(len(path) < 4):
            continue
        if(abs(path[0][1] - path[len(path) - 1][1]) < 3):
           continue
        if(path[len(path) - 1][1] <= 40 and path[len(path) - 1][1] >= 11):
            continue
        new_path.append(path)
        new_points.append(points[index])
        index += 1
    return new_points, new_path
'''
判断是否是好的corner points
判断的依据就是在每个SPLITLINE_1_RANGE钟是否有点存在
'''
def goodCornerPoints(points):
    n1 = 0
    n2 = 0
    n3 = 0
    for p in points:
        if p[0] >= Constants.SPLITLINE_1_RANGE[0] and p[0] <= Constants.SPLITLINE_1_RANGE[1]:
            n1 += 1
        if p[0] >= Constants.SPLITLINE_2_RANGE[0] and p[0] <= Constants.SPLITLINE_2_RANGE[1]:
            n2 += 1
        if p[0] >= Constants.SPLITLINE_3_RANGE[0] and p[0] <= Constants.SPLITLINE_3_RANGE[1]:
            n3 += 1
    return (n1 > 0 and n2 > 0 and n3 > 0)
'''
获取细化后的corner point(端点)
端点是这个点的8邻域中只有一个是1的点
本函数返回的corner points就是找好了的，后面的程序就根据这些points做分割线好了
'''
def getCornerPointsAndPaths(oriImg):
    points = []
    paths = []
    #先膨胀，element width最大从6开始
    for element_width in range(10, 1, -1):
        #print("element_width:" + str(element_width))
        img = dilate(oriImg, element_width)
        #cv2.imshow("ske", img)
        #注意skeleton返回的默认黑色为0，白色255，所以要处理
        img = ~Util.skeleton(img)
        #img[img == 0] = 1
        #img[img == 255] = 0
        #Util.printCV2Image(img)
        points = getOneCornerPoint(img)
        #print(points)
        #cv2.imwrite(Constants.IMG_DIR_TENCENT_TRAIN + "ske.jpg", img)
        #找到4个端点了，即认为可以了，可以退出循环了
        if goodCornerPoints(points):
            print("Best element_width:" + str(element_width))
            points.sort(key=lambda x: x[0])
            paths = getPathsFromCornerPoints(img, points)
            points, paths = filterPathsAndPoints(points, paths)
            paths = addEndPointToPaths(paths)
            paths = eraseDuplicateInPaths(paths)
            
            break
    return points, paths

'''
为得到的path加上终点，是为了方便fillPath的工作
'''
def addEndPointToPaths(paths):
    for path in paths:
        #说明是从大到小排列的
        if path[0][1] > path[len(path) - 1][1]:
            if(path[len(path) - 1][1] != 0):
                path.append((path[len(path) - 1][0], 0))
        else:
            if(path[len(path) - 1][1] != Constants.IMG_HEIGHT - 1):
                path.append((path[len(path) - 1][0], Constants.IMG_HEIGHT - 1))        
    return paths

'''
消除path中有重复Y值得项
'''
def eraseDuplicateInPaths(paths):
    for path in paths:
        y = path[0][1]
        length = len(path)
        index = 1
        for i in range(1, length):
            if index == len(path):
                break
            #print(path)
            #print(index)
            if path[index][1] == y:
                del path[index]
                #index -= 1
            else:
                y =  path[index][1]
                index += 1
    return paths
'''
这个函数只获取一次转换后的一个corner point罢了
因为背景细化后的图像我们不继续使用的，所以没必要保留
'''
def getOneCornerPoint(img):
    height, width = img.shape
    points = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            #y, x是中心点的坐标
            if img.item(y, x) == 0:
                sum = img.item(y - 1, x - 1) + img.item(y - 1,x) + img.item(y - 1,x + 1)
                sum += img.item(y, x - 1) + img.item(y,x + 1)
                sum += img.item(y + 1, x - 1) + img.item(y + 1,x) + img.item(y + 1, x + 1)
                if sum == 255 * 7:
                    points.append((x, y))
    return points            
    
def getOnePathFromCornerPoint(img, point, path):
    height, width = img.shape
    x = point[0]
    y = point[1]
    BLACK = 0
    #Util.printCV2Image(img)
    #print(point)
    #print(path)
    #这是边界的终止条件
    if (x - 1 < 0) or (x + 1 >= width) or (y - 1 < 0) or (y + 1 >= height):
        return path
    #终止条件，当两个点纵坐标一样时，查找就停止了
    if len(path) > 2 and path[len(path) - 2][1] == point[1]:
        return path
    
    if img.item(y - 1, x - 1) == BLACK and not ((x - 1, y - 1) in path):
        path.append((x - 1, y - 1))
        return getOnePathFromCornerPoint(img, (x - 1, y - 1), path)
    
    elif img.item(y - 1, x) == BLACK and not ((x, y - 1) in path):
        path.append((x, y - 1))
        return getOnePathFromCornerPoint(img, (x, y - 1), path)
    elif img.item(y - 1, x + 1) == BLACK and not ((x + 1, y - 1) in path):
        path.append((x + 1, y - 1))
        return getOnePathFromCornerPoint(img, (x + 1, y - 1), path)  
    elif img.item(y, x - 1) == BLACK and not ((x - 1, y) in path):
        path.append((x - 1, y))
        return getOnePathFromCornerPoint(img, (x - 1, y), path)  
    elif img.item(y, x + 1) == BLACK and not ((x + 1, y) in path):
        path.append((x + 1, y))
        return getOnePathFromCornerPoint(img, (x + 1, y), path)  
    elif img.item(y + 1, x - 1) == BLACK and not ((x - 1, y + 1) in path):
        path.append((x - 1, y + 1))
        return getOnePathFromCornerPoint(img, (x - 1, y + 1), path)  
    elif img.item(y + 1, x) == BLACK and not ((x, y + 1) in path):
        path.append((x, y + 1))
        return getOnePathFromCornerPoint(img, (x, y + 1), path)  
    elif img.item(y + 1, x + 1) == BLACK and not ((x + 1, y + 1) in path):
        path.append((x + 1, y + 1))
        return getOnePathFromCornerPoint(img, (x + 1, y + 1), path)      
    return path        
'''
遍历所有的corner points，找到每个corner points附属的分割线
'''
def getPathsFromCornerPoints(img, points):
    height, width = img.shape
    paths = []
    for p in points:
        path = []
        path.append(p)
        path = getOnePathFromCornerPoint(img, p, path)
        #再把终点加上
        if len(path) > 2:
            if path[0][1] >= path[len(path) - 1][1]:
                #需要将顶端点为终点
                path.append((path[len(path) - 1][0], 0))
            else:
                path.append((path[len(path) - 1][0], height - 1))
        #print(path)
        paths.append(path)
    return paths

'''
由背景细化得到的分割线构造出完整的分割线
要判断分割线的特性从而决定是从上分割还是从下分割了
'''
def getAllFullSpiltPath(img, paths):
    new_paths = []
    for corner_path in paths:
        if len(corner_path) > 2:
            path = []
            if corner_path[0][1] >= corner_path[len(corner_path) - 1][1]:
                SpiltUtil.searchRecursionFromTop(img, path, corner_path[0][0], corner_path[0][1])
            else:
                SpiltUtil.searchRecursionFromBottom(img, path, corner_path[0][0], corner_path[0][1])
            #print("path:" + str(path))
            #print("corner_path:" + str(corner_path))
            path.extend(corner_path)
            path.sort(key=lambda x: x[1])
            new_paths.append(path)
    return new_paths

'''
根据分割出来的path和point分割字符
'''
'''def getSpiltChars(img, points, paths, fileName):
    for i in len(points):
        if p[0] >= Constants.SPLITLINE_1_RANGE[0] and p[0] <= Constants.SPLITLINE_1_RANGE[1]:
            n1 += 1
        if p[0] >= Constants.SPLITLINE_2_RANGE[0] and p[0] <= Constants.SPLITLINE_2_RANGE[1]:
            n2 += 1
        if p[0] >= Constants.SPLITLINE_3_RANGE[0] and p[0] <= Constants.SPLITLINE_3_RANGE[1]:
            n3 += 1'''

def testAllPoints():
    ret = []
    labels = []
    dir_path = Constants.IMG_DIR_TENCENT_TRAIN
    for lists in os.listdir(dir_path): 
        #print(lists)
        path = os.path.join(dir_path, lists) 
        if not os.path.isdir(path): 
            oriImg = cv2.imread(path, 0)
            img = Util.binaryzation(oriImg)
            points, paths = getCornerPointsAndPaths(img)
            print("----------------RESULT----------------")
            print(path)
            print(points)
            print(len(paths))
    #target = readOneImg(dir_path + "YbUB_39_68_85.jpg")
    return ret, labels
if __name__ == "__main__":
    '''testAllPoints()
    #这个图像B的下方分割点是88，如果能正确把B分出来就OK
    fileName = Constants.IMG_DIR_TENCENT_TRAIN + "VKHM_34_50_77.jpg"
    oriImg = cv2.imread(fileName, 0)
    img = Util.binaryzation(oriImg)
    points, paths = getCornerPointsAndPaths(img)
    #all_paths = getAllFullSpiltPath(paths)
    print("----------------RESULT----------------")
    print(points)'''
    #path = [[(36, 6), (36, 6),(36, 6),(36, 6),(36, 6),(37, 5), (37, 4), (38, 3),(38, 3),(38, 3), (38, 2), (37, 1), (36, 0), (36, 0)]]
    #print(eraseDuplicateInPaths(path))
    #print(all_paths)
    #getSpiltChars(img, points, all_paths, fileName)
    #getAllPaths(img)
    cv2.waitKey(0)
