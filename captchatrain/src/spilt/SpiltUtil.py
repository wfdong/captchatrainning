# -*- coding: UTF-8 -*-
'''
Created on 2014/3/10

@author: rogers
'''
import cv2
from utils import Util
from utils import Constants
#设定字符宽度
charWidth = 3

'''
startX是起点
direction表明搜索的方向，从上开始搜，还是从下开始搜
charIndex表示这是搜索的第几个字符，因为要用char region限定分割线范围的
'''
def spiltOne(img, startX, direction, charIndex):
    #for j in range(startX, endX):
    height, width = img.shape
    path = []
    img = Util.erasePaddingInOneChar(img)
    img = ~Util.skeleton(~img)
    #Util.printCV2Image(img)
    #cv2.imshow("1", img)
    
    if direction == Constants.SEARCH_FROM_TOP:
        print("direction: t")
        path.append((startX, 0))
        searchOnceFromTop(img, path, startX, 0, charIndex)
    else:
        print("direction: b")
        path.append((startX, height - 1))
        searchOnceFromBottom(img, path, startX, height - 1, charIndex)
    #print(path)
    #for i in range(len(path.path) - 1):
    #    cv2.line(img, path.paths[i].toTuple(), path.paths[i - 1].toTuple(), 0)
    #cv2.imshow("Image", img) 
    #if we want show a img, must add this
    #cv2.waitKey (0)
    return path

'''
以(start, height - 1)为起点，从底向上做一次搜索
start就是起点的x坐标了，y坐标默认为height - 1
这个是从白点开始搜索的函数
'''
def searchOnceFromBottom(img, path, startX, startY, charIndex):
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
    searchRecursionFromBottom(img, path, startX, blackY, charIndex)
    
'''
这个是从黑点开始递归搜索的函数
'''
def searchRecursionFromBottom(img, path, startX, startY, charIndex):
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
        searchRecursionFromBottom(img, path, startX, startY - 1, charIndex)
    #3.2 
    elif leftDistance <= charWidth and rightDistance <= charWidth:
        newX = int((leftX + rightX) / 2 )
        newX = limitX(newX, charIndex)
        path.append((newX, startY))
        searchRecursionFromBottom(img, path, newX, startY - 1, charIndex)
    elif leftDistance <= charWidth and rightDistance >= charWidth:
        leftX = limitX(leftX, charIndex)
        path.append((leftX, startY))
        searchRecursionFromBottom(img, path, leftX, startY - 1, charIndex)
    elif leftDistance >= charWidth and rightDistance <= charWidth:
        rightX = limitX(rightX, charIndex)
        path.append((rightX, startY))
        searchRecursionFromBottom(img, path, rightX, startY - 1, charIndex)

def searchOnceFromTop(img, path, startX, startY, charIndex):
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
    print("blackY:" + str(blackY))
    searchRecursionFromTop(img, path, startX, blackY, charIndex)

def searchRecursionFromTop(img, path, startX, startY, charIndex):
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
    for i in range(startX - 1, 0, -1):
        if img.item(startY, i) == 0:
            leftX = i
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
        searchRecursionFromTop(img, path, startX, startY + 1, charIndex)
    #3.2 
    elif leftDistance <= charWidth and rightDistance <= charWidth:
        newX = int((leftX + rightX) / 2 )
        newX = limitX(newX, charIndex)
        path.append((newX, startY))
        searchRecursionFromTop(img, path, newX, startY + 1, charIndex)
    elif leftDistance <= charWidth and rightDistance >= charWidth:
        leftX = limitX(leftX, charIndex)
        path.append((leftX, startY))
        searchRecursionFromTop(img, path, leftX, startY + 1, charIndex)
    elif leftDistance >= charWidth and rightDistance <= charWidth:
        rightX = limitX(rightX, charIndex)
        path.append((rightX, startY))
        searchRecursionFromTop(img, path, rightX, startY + 1, charIndex)

'''
对分割线的X点做限定
使得X点要在charIndex指定的范围之内
'''
def limitX(x, charIndex):
    if x > Constants.SPILTLINE_RANGES[charIndex][1]:
        x = Constants.SPILTLINE_RANGES[charIndex][1] - 1
    elif x < Constants.SPILTLINE_RANGES[charIndex][0]:
        x = Constants.SPILTLINE_RANGES[charIndex][0] + 1
    return x
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
    oriImg = cv2.imread(Constants.IMG_DIR_TENCENT_TRAIN + "ANWm_33_58_82.jpg", 0)
    img = Util.binaryzation(oriImg)
    #cv2.imshow("ori", img)
    height, width = img.shape
    img = dilate(img, 6)
    cv2.imshow("ske", img)
    img = ~Util.skeleton(img)
    #img[img == 0] = 1
    #img[img == 255] = 0
    Util.printCV2Image(img)
    #我们只处理细化后的，这样不会收到多余字符的干扰
    #getAllPaths(img)
    cv2.waitKey(0)

    