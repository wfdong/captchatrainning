# -*- coding: UTF-8 -*-
'''
Created on 2014/3/2
程序的main函数
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

'''
这里的target是个四维数组，现在的是43 * 4 * 20 * 20
43 是train文件夹下图片的个数
4是每张图片有4个字符
20 * 20 是每个字符的尺寸
43 和 4 这两个是list， 20 * 20那是numpy array
'''
def rotateAllTarget(target, labels):
    new_target = []
    new_labels = []
    print(labels)
    for i in range(len(target)):
        img = []
        #img_labels = []
        for j in range(len(target[i])):
            for angle in range(Constants.LEFT_ANGLE, Constants.RIGHT_ANGLE):
                newChar = Util.rotate(target[i][j], angle)
                img.append(newChar)
                new_labels.append(labels[i * 4 + j])
        new_target.append(img)
        #new_labels.append(img_labels)
    print(len(new_target[0]))
    print(new_labels)
    return new_target, new_labels

def getSVM(train_new):
    if train_new:
        target, labels = ReadImg.readAllImg(Constants.IMG_DIR_TENCENT_TRAIN)
        #消除有些字母内部的填充
        TrainUseSVM.erasePaddingInAllChar(target)
        new_target, new_labels = rotateAllTarget(target, labels)
        svm = TrainUseSVM.trainSVMInMyWay(new_target, new_labels)
        return svm
    else:
        svm = cv2.SVM()
        svm.load('svm_data.dat')
        return svm
    


def recogniseOne(svm, img, path0, paths, charIndex):
    print("---recogniseOne---")
    height, width = img.shape
    start = Constants.SPILTLINE_RANGES[charIndex][0]
    end = Constants.SPILTLINE_RANGES[charIndex][1]
    char1 = []
    ret = []
    choosedPaths = []
    for path in paths:
        if path[0][0] >= start and path[0][0] <= end:
            #这里要再进行一次搜索的，因为之前的path是不完全的
            if path[len(path) - 1][1] == 0:
                SpiltUtil.searchRecursionFromTop(img, path, path[0][0], path[0][1], charIndex)
            else:
                SpiltUtil.searchRecursionFromBottom(img, path, path[0][0], path[0][1], charIndex)
            path.sort(key=lambda y: y[1])
            path1 = Util.fillPath(path, height)
            oneChar = ReadImg.getOneChar(img, path0, path1)
            print("NonZero in oneChar:" + str(np.count_nonzero(oneChar)))
            #如果这个分割出来的字符中黑色像素点太少，那直接不要了，因为肯定是错误的
            if np.count_nonzero(oneChar) < 30:
                continue
            c1 = Util.conventToTrainChar(oneChar)
            choosedPaths.append(path1)
            #Util.printCV2Image(c1)
            for angle in range(Constants.LEFT_ANGLE, Constants.RIGHT_ANGLE):
                c2 = Util.rotate(c1, angle)
                #Util.printCV2Image(c2)
                #cv2.imshow('img' + str(angle), c2)
                char1.append(c2)
    #最后一个字符，特殊处理
    if end == width - 1:
        path1 = Util.getRightBoder(width, height)
        choosedPaths.append(path1)
        c1 = Util.conventToTrainChar(ReadImg.getOneChar(img, path0, path1))
        for angle in range(Constants.LEFT_ANGLE, Constants.RIGHT_ANGLE):
            c2 = Util.rotate(c1, angle)
            #Util.printCV2Image(c2)
            #cv2.imshow('img' + str(angle), c2)
            char1.append(c2)
    
    '''for p1 in range(start, end):
        path1 = Util.fillPath(SpiltUseCV2.spiltOne(img, p1), height)
        c1 = Util.conventToTrainChar(ReadImg.getOneChar(img, path0, path1))
        for angle in range(Constants.LEFT_ANGLE, Constants.RIGHT_ANGLE):
            c2 = Util.rotate(c1, angle)
            #Util.printCV2Image(c2)
            #cv2.imshow('img' + str(angle), c2)
            char1.append(c2)'''
    #要训练的是(height, width, 20, 20)这样的四维数组
    ret.append(char1)
    result = predictInMyWay(svm, ret)
    #print("result:" + str(result))
    maxValue = getMaxNumChar(result)
    bestSplitPath = findBestSplitPath(result, maxValue, choosedPaths)
    print("bses Path:" + str(bestSplitPath))
    return bestSplitPath, maxValue
    
'''
激动人心的时刻到了！
这个函数就是处理待验证图片的主入口啦
puzzle = img
这里要做一些前期处理的
'''
def readInput(svm, puzzle):
    oriImg = cv2.imread(puzzle, 0)
    img = Util.binaryzation(oriImg)
    img = Util.erasePaddingInOneChar(img)
    points, paths = SpiltUseBackground.getCornerPointsAndPaths(img)
    print("-------RECOGNISE--------")
    print(points)
    print("paths:")
    for p in paths:
        print(p)
    height, width = img.shape
    code = []
    #p1 = [23:39]
    path0 = Util.getLeftBorder(height)
    good_path, result = recogniseOne(svm, img, path0, paths, 0)
    print("First split point:" + str(good_path[0][0]) + " result:" + chr(result + ord('A')))
    code.append(chr(result + ord('A')))
    
    #path0 = Util.fillPath(SpiltUseCV2.spiltOne(img, p1), height)
    good_path, result = recogniseOne(svm, img, good_path, paths, 1)
    print("Second split point:" + str(good_path[0][0]) + " result:" + chr(result + ord('A')))
    code.append(chr(result + ord('A')))
    
    #path0 = Util.fillPath(SpiltUseCV2.spiltOne(img, p2), height)
    good_path, result = recogniseOne(svm, img, good_path, paths, 2)
    print("Third split point:" + str(good_path[0][0]) + " result:" + chr(result + ord('A')))
    code.append(chr(result + ord('A')))
    
    #path0 = Util.fillPath(SpiltUseCV2.spiltOne(img, p3), height)
    good_path, result = recogniseOne(svm, img, good_path, paths, 3)
    print("Fourth split point:" + str(good_path[0][0]) + " result:" + chr(result + ord('A')))
    code.append(chr(result + ord('A')))
    
    print("Result:" + str(code))
    
'''
此函数找最优的分割点
result是预测的结果集
maxValue是result出现次数最多的那个值
start是分割线范围的起点
end是分割线范围的终点
'''
def findBestSplitPath(result, maxValue, choosedPaths):
    #print(result)
    print("maxValue:" + str(maxValue))
    angleNum = Constants.RIGHT_ANGLE - Constants.LEFT_ANGLE
    #保存每个分割点里等于maxValue的个数
    pointScore = []
    onePoint = 0
    onePointScore = 0
    print("result len = " + str(len(result)) + ". Count of paths:" + str(len(choosedPaths)))
    for ret in result:
        if int(ret[0]) == maxValue:
            onePointScore = onePointScore + 1
        onePoint = onePoint + 1
        if onePoint == angleNum:
            pointScore.append(onePointScore)
            onePoint = 0
            onePointScore = 0
            
    #print(sorted(pointScore, reverse=True))
    print(pointScore)  
    #print(pointScore.index(max(pointScore)) + start + 1) 
    
    return (choosedPaths[pointScore.index(max(pointScore))])     

    
'''
此函数找最优的分割点
result是预测的结果集
maxValue是result出现次数最多的那个值
start是分割线范围的起点
end是分割线范围的终点
'''
def findBestSplitPoint_Old(result, maxValue, start, end):
    #print(result)
    print("maxValue:" + str(maxValue))
    angleNum = Constants.RIGHT_ANGLE - Constants.LEFT_ANGLE
    #print(angleNum)
    #保存每个分割点里等于maxValue的个数
    pointScore = []
    onePoint = 0
    onePointScore = 0
    print("result len = " + str(len(result)) + ". Start:" + str(start) + ", end=" + str(end))
    for ret in result:
        if int(ret[0]) == maxValue:
            onePointScore = onePointScore + 1
        onePoint = onePoint + 1
        if onePoint == angleNum:
            pointScore.append(onePointScore)
            onePoint = 0
            onePointScore = 0
            
    #print(sorted(pointScore, reverse=True))
    print(pointScore)  
    #print(pointScore.index(max(pointScore)) + start + 1) 
    
    return (pointScore.index(max(pointScore)) + start + 1) 

def predict(svm, input):
    deskewed = [map(TrainUseSVM.deskew,row) for row in input]
    hogdata = [map(TrainUseSVM.hog,row) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1,TrainUseSVM.bin_n*4)
    result = svm.predict_all(testData)
    #print(result)
    '''for c in result:
        print chr(c + ord('A')), '''
    return result

def predictInMyWay(svm, input):
    testData = np.float32(input).reshape(-1,400)
    result = svm.predict_all(testData)
    #print(result)
    '''for c in result:
        print chr(c + ord('A')), '''
    return result

'''
本函数的目的是找出在array这个list出现次数最多的值
'''
def getMaxNumChar(result):
    l = list(result.reshape(-1,))
    code_num = {}
    for code_item in set(l):
        code_num[chr(int(code_item) + ord('A'))]=l.count(code_item)
    #sorted_code = sorted( code_num.iteritems(), key=operator.itemgetter(1),reverse=True)
    #print(code_num)
    print(sorted(code_num.items(), lambda x, y: cmp(x[1], y[1]), reverse=True))
    count = 0
    maxValue = 0
    for (item,num) in code_num.items():
        if num > count:
            count = num
            maxValue = item
    #print(maxValue)
    #lists = [item for item,num in code_num ]
    #print(count)
    return int(ord(maxValue) - ord('A'))

if __name__ == "__main__":
    print("start main...")
    train_new = False
    svm = getSVM(train_new)
    #以后会传来自rest call中的data
    readInput(svm, Constants.IMG_DIR_TENCENT_TRAIN + "SPPV_32_63_93.jpg")
    cv2.waitKey(0)