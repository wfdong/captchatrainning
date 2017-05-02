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
from utils import FileUtil
from trainNeuro import ReadImg
from trainNeuro import TrainUseSVMInSklLearn
from spilt import SpiltUseCV2
from spilt import SpiltUseBackground
from spilt import SpiltUtil
import os

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
        TrainUseSVMInSklLearn.erasePaddingInAllChar(target)
        new_target, new_labels = rotateAllTarget(target, labels)
        svm = TrainUseSVMInSklLearn.getSVC(new_target, new_labels, True)
        return svm
    else:
        #classifier = svm.SVC()
        svm = TrainUseSVMInSklLearn.getSVC(None, None, False)
        return svm
    



    
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
    print("First split point:" + str(good_path[0][0]) + " result:" + chr(svm.classes_[result] + ord('A')))
    code.append(chr(svm.classes_[result] + ord('A')))
    
    #path0 = Util.fillPath(SpiltUseCV2.spiltOne(img, p1), height)
    good_path, result = recogniseOne(svm, img, good_path, paths, 1)
    print("Second split point:" + str(good_path[0][0]) + " result:" + chr(svm.classes_[result] + ord('A')))
    code.append(chr(svm.classes_[result] + ord('A')))
    
    #path0 = Util.fillPath(SpiltUseCV2.spiltOne(img, p2), height)
    good_path, result = recogniseOne(svm, img, good_path, paths, 2)
    print("Third split point:" + str(good_path[0][0]) + " result:" + chr(svm.classes_[result] + ord('A')))
    code.append(chr(svm.classes_[result] + ord('A')))
    
    #path0 = Util.fillPath(SpiltUseCV2.spiltOne(img, p3), height)
    good_path, result = recogniseOne(svm, img, good_path, paths, 3)
    print("Fourth split point:" + str(good_path[0][0]) + " result:" + chr(svm.classes_[result] + ord('A')))
    code.append(chr(svm.classes_[result] + ord('A')))
    
    print("Result:" + str(code))
    return code
    
'''
此函数找最优的分割点
result是预测的结果集
maxValue是result出现次数最多的那个值
start是分割线范围的起点
end是分割线范围的终点
'''
def findBestSplitPath(result, maxValue, choosedPaths):
    #print(result)
    #print("maxValue:" + str(maxValue))
    angleNum = Constants.RIGHT_ANGLE - Constants.LEFT_ANGLE
    #保存每个分割点里等于maxValue的个数
    pointScore = []
    onePoint = 0
    onePointScore = 0
    #print("result len = " + str(len(result)) + ". Count of paths:" + str(len(choosedPaths)))
    for ret in result:
        if ret == maxValue:
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


def predictInMyWay(svm, input):
    ret = []
    deskewed = [map(TrainUseSVMInSklLearn.deskew,row) for row in input]
    hogdata = [map(TrainUseSVMInSklLearn.hog,row) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1, 64)
    #result = svm.predict(testData)
    result2 = svm.predict_proba(testData)
    #print(result2)
    labels = svm.classes_
    #print(labels)
    for c in result2:
        l = list(c)
        ret.append(l.index(max(l)))
    return ret

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
            #因为在getOneChar里加了wrapAffine，导致了getOneChar是返回的一个字符数组
            #但我们识别字符时现在还不想作wrapaffine，所以就默认去第一个原始字符了
            oneChar = oneChar[0]
            print("NonZero in oneChar:" + str(np.count_nonzero(oneChar)))
            #如果这个分割出来的字符中黑色像素点太少，那直接不要了，因为肯定是错误的
            if np.count_nonzero(oneChar) < 40:
                continue
            c1 = Util.conventToTrainChar(oneChar)
            choosedPaths.append(path1)
            char1.append(c1)
            searchAllPossible(img, path0, path1, char1, choosedPaths, charIndex)
    
    #即在上面的paths中并没有找到合适的分割线
    if len(char1) == 0:
        '''
                            因为上面并没有合适的分割线，所以这里手动选取两个合适的分割点
        1.path0.x + 20
        2.path0.x + 2 *　Constants.SEARCH_ALL_POSSIBLE_RANGE
        '''
        startX1 = path0[0][0] + 20
        path1 = Util.fillPath(SpiltUtil.spiltOne(img, startX1, Constants.SEARCH_FROM_TOP, charIndex), height)
        searchAllPossible(img, path0, path1, char1, choosedPaths, charIndex)
        startX1 = startX1 + (2 * Constants.SEARCH_ALL_POSSIBLE_RANGE)
        path1 = Util.fillPath(SpiltUtil.spiltOne(img, startX1, Constants.SEARCH_FROM_TOP, charIndex), height)
        searchAllPossible(img, path0, path1, char1, choosedPaths, charIndex)
                
    #最后一个字符，特殊处理
    if end == width - 1:
        path1 = Util.getRightBoder(width, height)
        choosedPaths.append(path1)
        #因为在getOneChar里加了wrapAffine，导致了getOneChar是返回的一个字符数组
        #但我们识别字符时现在还不想作wrapaffine，所以就默认去第一个原始字符了
        c1 = Util.conventToTrainChar(ReadImg.getOneChar(img, path0, path1)[0])
        searchAllPossible(img, path0, path1, char1, choosedPaths, charIndex)
    
    #要训练的是(height, width, 20, 20)这样的四维数组
    ret.append(char1)
    indexs, probs = predictAndCalc(svm, ret)
    print("indexs:" + str(indexs))
    print("probs:" + str(probs))
    maxIndex, spiltIndex = getMaxNumChar_New(indexs, probs)
    print("maxIndex:" + str(maxIndex))
    print("spiltIndex:" + str(spiltIndex))
    bestSplitPath = findBestSplitPath_New(spiltIndex, choosedPaths)
    print("best Path:" + str(bestSplitPath))
    return bestSplitPath, maxIndex

def searchAllPossible(img, path0, path1, char1, choosedPaths, charIndex):
    height, width = img.shape
    #在这里想加上对这个开始的分割点左右5个点分别进行搜索，以尽量获得一个最大概率的
    #新加的左右五个点search，先把旋转去掉了
    #从上搜索
    searchRange = Constants.SEARCH_ALL_POSSIBLE_RANGE
    for start in range(max(path1[0][0] - searchRange, 0), path1[0][0]):
        newPath = Util.fillPath(SpiltUtil.spiltOne(img, start, Constants.SEARCH_FROM_TOP, charIndex), height)
        char1.append(Util.conventToTrainChar(ReadImg.getOneChar(img, path0, newPath)[0]))
        choosedPaths.append(newPath)
    #从上搜索
    for start in range(path1[0][0] + 1, min(path1[0][0] + searchRange, height - 1)):
        newPath = Util.fillPath(SpiltUtil.spiltOne(img, start, Constants.SEARCH_FROM_TOP, charIndex), height)
        char1.append(Util.conventToTrainChar(ReadImg.getOneChar(img, path0, newPath)[0]))
        choosedPaths.append(newPath)
    #从下搜索
    for start in range(max(path1[height - 1][0] - searchRange, 0), path1[height - 1][0]):
        newPath = Util.fillPath(SpiltUtil.spiltOne(img, start, Constants.SEARCH_FROM_BOTTOM, charIndex), height)
        char1.append(Util.conventToTrainChar(ReadImg.getOneChar(img, path0, newPath)[0]))
        choosedPaths.append(newPath)
                
    #从下搜索
    for start in range(path1[height - 1][0] + 1, min(path1[height - 1][0] + searchRange, height - 1)):
        newPath = Util.fillPath(SpiltUtil.spiltOne(img, start, Constants.SEARCH_FROM_BOTTOM, charIndex), height)
        char1.append(Util.conventToTrainChar(ReadImg.getOneChar(img, path0, newPath)[0]))
        choosedPaths.append(newPath)

'''
此函数找最优的分割点
result是预测的结果集
maxValue是result出现次数最多的那个值
start是分割线范围的起点
end是分割线范围的终点
'''
def findBestSplitPath_New(spiltIndex, choosedPaths):
    #FIXME 以后要改的，现在只是按最简单的方式实现下
    angle_num = Constants.RIGHT_ANGLE - Constants.LEFT_ANGLE
    return choosedPaths[spiltIndex]

def predictAndCalc(svm, input):
    #这里存储每个max值对应的index
    indexs = []
    #这里存储每个max值对应的概率
    probs = []
    deskewed = [map(TrainUseSVMInSklLearn.deskew,row) for row in input]
    hogdata = [map(TrainUseSVMInSklLearn.hog,row) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1, 64)
    #result = svm.predict(testData)
    result = svm.predict_proba(testData)
    #print(result)
    labels = svm.classes_
    #print(labels)
    for c in result:
        l = list(c)
        indexs.append(l.index(max(l)))
        probs.append(max(l))
                     
    return indexs, probs

'''
本函数的目的是找出在array这个list出现次数最多的值
'''
def getMaxNumChar(result):
    #l = list(result.reshape(-1,))
    code_num = {}
    for code_item in set(result):
        code_num[chr(int(code_item) + ord('A'))]=result.count(code_item)
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

'''
根据概率和indexs出现的次数确定预测的结果
indexs是labels的index
'''
def getMaxNumChar_New(indexs, probs):
    #保存index,对应的prob集合
    ret = {}
    #记录每个概率出现的次数了，为了取均值时使用
    prob_times = {}
    for i in range(len(indexs)):
        if not ret.has_key(indexs[i]):
            ret[indexs[i]] = []
        ret[indexs[i]].append(probs[i])
    #计算概率和的均值
    average = {}
    for (item,num) in ret.items():
        list = sorted(ret[item], reverse=True)
        average[item] = 0
        #如果不够三个元素，就只取最大的好了
        if len(list) < 3:
            average[item] = max(list)
        else:
            #多于3个元素，取前面最大的三分之一算均值
            for i in range(len(list) / 3):
                average[item] = average[item] + list[i]
            print("Total average:" + str(average[item]))
            print("len list　:" + str((len(list) / 3)))
            average[item] = average[item] / (len(list) / 3)

    print("index&prob:" + str(average))
    count = 0
    maxIndex = 0
    #找到均值中最大的概率
    for (item,num) in average.items():
        if num > count:
            count = num
            maxIndex = item
    maxProb = max(ret[maxIndex])
    print("maxProb:" + str(maxProb))
    spiltIndex = probs.index(maxProb)
    return maxIndex, spiltIndex

def verifyAll(svm):
    dir_path = Constants.IMG_DIR_TENCENT_TRAIN
    ret = []
    for lists in os.listdir(dir_path): 
        #print(lists)
        path = os.path.join(dir_path, lists) 
        if not os.path.isdir(path): 
            print("---Now---" + str(path))
            code = readInput(svm, path)
            ret.append((path, code))
    for r in ret:
        print("[Path:%s, code:%s]" % r)

if __name__ == "__main__":
    print("start main...")
    train_new = True
    
    svm = getSVM(train_new)

    #verifyAll(svm)
    #以后会传来自rest call中的data
    #readInput(svm, Constants.IMG_DIR_TENCENT_TRAIN + "BaUK_41b_68b_88t.jpg")
    #print("labels:")
    #print(svm.classes_)
    cv2.waitKey(0)