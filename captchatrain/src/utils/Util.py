# -*- coding: UTF-8 -*-
'''
Created on 2014/2/15

@author: rogers
'''
import cv2
import Constants
from skimage import morphology
import numpy as np
import os

'''
读取指定目录下的图片文件，并对每个图片调用funx进行处理
'''
def readAllImg(dir_path, func):
    for lists in os.listdir(dir_path): 
        path = os.path.join(dir_path, lists) 
        if not os.path.isdir(path): 
            func(path)
    

#二值化图像
def binaryzation(img):
    # Otsu's thresholding
    level, retImg = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return retImg

def skeleton(img):
    img[img == 255] = 1
    img = morphology.skeletonize(img)
    img = img.astype(np.uint8)
    img[img == 1] = 255
   
    #cv2.imshow("Image1", im)
    #cv2.imshow("Image2", img)
    #img[img == 255] = 1
    return img
    #Util.printCV2Image(im)
    #cv2.waitKey (0)


def printCV2Image(img):
    height, width = img.shape
    for i in range(height):
        print("")
        for j in range(width):
            print img.item(i,j),
    print(img.shape)
    
def printListImage(img):
    for i in range(len(img)):
        print("")
        for j in range(len(img[i])):
            print img[i,j],
    
'''
填充path，保证path中所有的纵坐标都是连续的
就是连接path的每两个相邻点
先计算斜率K，再通过斜率填充X，Y
'''
def fillPath(path, height):
    print("[FillPath->OriPath]:" + str(path))
    #如果列表是从下到上的点（即从下方开始搜索的，先反转）
    #统一按从上到下搜索来处理
    if path[0][1] == height - 1:
        #需要把最后缺失的X，0加上
        path.append((path[len(path) - 1][0], 0))
        path.reverse()
    #print("[Inital] " + str(path))  
    if path[0][1] != 0:
        path.insert(0,(path[0][0], 0))
    path_index = 0
    for i in range(height):
        if(path_index == len(path) - 1):
            break
        (x1,y1) = path[path_index]
        (x2,y2) = path[path_index + 1]
        if y1 == y2:
            K = 1
        else:
            K = float((x2 - x1))/(y2 - y1)
        #print("K = " + str(K))
        #baseX = path[path_index][0]
        for Y in range(y1 + 1, y2):
            new_X = (Y - y1) * K + x1
            path_index += 1
            path.insert(path_index, (int(new_X), Y))
        path_index += 1
        #print(path)
    for i in range(len(path), height):
        path.append((path[len(path) - 1][0], i))
    return path

'''
获取img在X轴上的投影
返回的是一维数组
'''
def get_X_Projection(img):
    height, width = img.shape
    xp = []
    for y in range(height):
        p = 0
        for x in range(width):
            #因为白色像素为0，黑色为1，所以直接相加好了
            p += img.item(y, x)
        xp.append(p)
    return xp

'''
获取img在Y轴上的投影
返回的是一维数组
'''
def get_Y_Projection(img):
    height, width = img.shape
    yp = []
    for x in range(width):
        p = 0
        for y in range(height):
            #因为白色像素为0，黑色为1，所以直接相加好了
            p += img.item(y, x)
        yp.append(p)
    return yp
    

def get_X_ListInPath(path):
    x = []
    for point in path:
        x.append(point[0])
    return x

def get_Y_ListInPath(path):
    y = []
    for point in path:
        y.append(point[1])
    return y

#产生左边界，字符分割时用
def getLeftBorder(height):
    path = []
    for j in range(height):
        path.append((0, j))
    return fillPath(path, height)

#产生右边界，字符分割时用
def getRightBoder(width, height):
    path = []
    for j in range(height):
        path.append((width - 1, j))
    return fillPath(path, height)

'''
对一个图像进行resize
img是图像的array
oriSize是图像原来的大小，以tuple的形式传入，(height, width)，最好是正方形
newSize是图像的新大小，以tuple的形式传入，最好是正方形
返回resize后的图像
'''
def resizeImg(img, newSize):
    #print("oriSize:" + str(oriSize))
    #print("newSize:" + str(newSize))
    #return cv2.resize(img, newSize)
    return cv2.resize(img, newSize, interpolation=cv2.INTER_AREA)
    
'''
强行把source图像的所以元素赋值到dst图像中
'''
def copyOneImg(source, oriSize):
    #这句话以后肯定要改了，现在是因为有ndarray type的问题，导致不能resize，只能先这样
    file_img = binaryzation(cv2.imread(Constants.IMG_DIR_TENCENT_TRAIN + "YbUB_44b_72b_88b.jpg", 0))
    imgChop = file_img[0:oriSize[1], 0:oriSize[0]]
    height, width = source.shape
    for j in range(height):
        for i in range(width):
            imgChop[j, i] = source.item(j, i)
    return imgChop

def conventToTrainCharFromAllWrap(allWrapChars):
    ret = []
    for oneChar in allWrapChars:
        ret.append(conventToTrainChar(oneChar))
    return ret
'''
将一个字母由ONE_CHAR_WIDTH组成的正方形转换成TRAIN_CHAR_WIDTH组成的正方形
有的可能要缩放的，旋转就先不要了
'''
def conventToTrainChar(oneChar):
    ret = []
    height, width = oneChar.shape
    #寻找字符的右边界和下边界，以确定字符的大小，判断是否要缩小，只缩小就行了，不用扩大 
    rightBorder = 0
    bottom = 0
    find = False
    for y in range(height - 1, -1, -1):
        for x in range(width):
            if(oneChar.item(y, x) == Constants.INVERT_COLOUR):
                bottom = y
                find = True
                break
        if(find == True):
            break
    find = False
    for x in range(width - 1, -1, -1):
        for y in range(height):
            if(oneChar.item(y, x) == Constants.INVERT_COLOUR):
                rightBorder = x
                find = True
                break
        if(find == True):
            break    
    #print("rightBorder:" + str(rightBorder))
    #print("bottom:" + str(bottom))
    maxOne = max(bottom, rightBorder)
    
    #只截取边界内的点，为了是正方形，所以取一个最大值了
    charCrop = oneChar[0:maxOne, 0:maxOne] 
    #charCrop = copyOneImg(charCrop, (maxOne, maxOne))
    #print("-----Before resize-----")
    #printCV2Image(charCrop)
    newShape = (Constants.TRAIN_CHAR_HEIGHT, Constants.TRAIN_CHAR_WIDTH)

    newChar = resizeImg(charCrop, newShape)
    print("-----After resize-----")
    printCV2Image(newChar)
    '''print("------Before Wrap------")
    for wImg in wrap(crop):
        printCV2Image(wImg)
        print("-----Before resize-----")
        printCV2Image(wImg)
        newShape = (Constants.TRAIN_CHAR_HEIGHT, Constants.TRAIN_CHAR_WIDTH)
        newChar = resizeImg(wImg, newShape)
        ret.append(newChar)
        print("-----After resize-----")
        printCV2Image(newChar)
        #print(newChar.shape)'''
    return newChar

'''
消除一个字母的内部填充，因为腾讯的验证码有的可能会空心，但有的会内部全填充为黑色，这时就要对填充后的做处理了
处理的方式也很简单，就是
1.腐蚀，可以用稍微大一点的腐蚀矩阵
2.原图-腐蚀后的图，就OK了
'''
def erasePaddingInOneChar(img):
    #OpenCV定义的结构元素  
    #MORPH_CROSS， MORPH_RECT， MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
    #腐蚀图像  
    eroded = cv2.erode(~img, kernel) 
    return ~(~eroded - img)

'''
旋转图片，angle是要旋转的角度，应该是从-20 : 20
角度如果是正的，是向右转，是负的，就像左转
'''
def rotate(img, angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
    #cv2.imshow('img',dst)
    #cv2.waitKey(0)
    
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
                newChar = rotate(target[i][j], angle)
                #print("after rotate:" + str(newChar))
                img.append(newChar)
                new_labels.append(labels[i * 4 + j])
                #对每个选择的字符进行仿射变换
                '''for wImg in wrap(newChar):
                    img.append(wImg)
                    new_labels.append(labels[i * 4 + j])'''
                
        new_target.append(img)
        #new_labels.append(img_labels)
    #print(len(new_target[0]))
    #print(new_labels)
    return new_target, new_labels

'''
对图像进行仿射变换
'''
def wrap(img):
    '''img = cv2.imread(Constants.IMG_DIR_TENCENT_TEST + "H.jpg", 0)
    img = Util.binaryzation(img)'''
    ret = []
    rows,cols = img.shape

    pts1 = np.float32([[0,0],[0,20],[20,0]])
    #这样变换会横向变窄
    #pts2 = np.float32([[0,0],[0,20],[18,0]])
    #这样会纵向变扁
    #pts2 = np.float32([[0,0],[0,18],[20,0]])
    #这样横向变窄，且有一定的变形
    #pts2 = np.float32([[0,0],[0,20],[18,2]])
    #这样纵向变窄，且有一定的变形
    #pts2 = np.float32([[0,0],[2,18],[20,0]])
    #pts2All = [np.float32([[0,0],[0,20],[20,0]]), np.float32([[0,0],[0,20],[18,0]]), np.float32([[0,0],[0,18],[20,0]]), 
    #           np.float32([[0,0],[0,20],[18,2]]), np.float32([[0,0],[2,18],[20,0]])]
    pts2All = [np.float32([[0,0],[0,20],[20,0]]), np.float32([[0,0],[0,20],[18,2]]), np.float32([[0,0],[2,18],[20,0]])]
    #pts2All = [np.float32([[0,0],[0,20],[20,0]])]
    #pts2 = np.float32([[0,0],[20,20],[20,0]])
    for pts2 in pts2All:
        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(img,M,(cols,rows))
        #print("-------WARP ONCE-------")
        #printCV2Image(dst)
        ret.append(dst)
    
    return ret
    '''dst = Util.binaryzation(dst)
    cv2.imshow("2", img)
    cv2.imshow("1", ~dst)
    Util.printCV2Image(img)
    Util.printCV2Image(~dst)
    cv2.waitKey(0)'''
    #plt.subplot(121),plt.imshow(img),plt.title('Input')
    #plt.subplot(122),plt.imshow(dst),plt.title('Output')
    #plt.show()
    