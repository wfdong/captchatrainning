# -*- coding: UTF-8 -*-
from PIL import Image
import os  
import cv2
from utils import Constants
from utils import Util
import numpy as np

def readDir(rootDir, func): 
    for lists in os.listdir(rootDir): 
        path = rootDir + "/" + lists
        print(path)
        if(func):
            func(path)
            
        if os.path.isdir(path): 
            readDir(path, func) 

def convertToGray(imgPath):
    img = Image.open(imgPath)
    #width, height = img.size
    #print(str(width) + "-" + str(height))
    new_img = img.convert('L')
    '''for i in range(width):
            print("")
            for j in range(height):
            print(str(new_img.getpixel((i,j))) + " ",end='')'''
    new_img.save(imgPath)

def spilt(imgPath):
    img = Image.open(imgPath)
    width, height = img.size
    print(str(width) + "-" + str(height))
    for i in range(width):
        #print("")
        sum = 0
        for j in range(height):
            sum += img.getpixel((i,j))
            #print(str(img.getpixel((i,j))) + " ",end='')
        #print(str(sum) + " ",end='')
    pass

'''
将一整张图片切分成带有4个字母的图片并保存
'''
def split2Single(imgPath):
    print(imgPath)
    img = cv2.imread(imgPath, 0)
    #img = Util.binaryzation(img)
    i1 = (10, 60)
    i2 = (30, 80)
    i3 = (50, 100)
    i4 = (70, 120)
    imgs = [i1, i2, i3, i4]
    img_name = imgPath[imgPath.rindex('/') + 1:]
    if img_name.find('_') > -1 :
        labels = img_name.split('_')[0]
    else:
        labels = img_name[:img_name.index('.')]
    for i in range(4):
        crop_img = img[1:51, imgs[i][0]:imgs[i][1]]
        splitChar = "_"
        if ((ord)(labels[i])-ord('a')) >= 0:
             splitChar = "__"
        fileName = Constants.IMG_DIR_TENCENT_SPLIT + labels[i] + splitChar + str(indexDict[(ord)(labels[i])-ord('A')]) + ".jpg"
        if TEST:
            fileName = Constants.IMG_DIR_TENCENT_TEST_SPLIT + labels[i] + splitChar + str(indexDict[(ord)(labels[i])-ord('A')]) + ".jpg"
        #print(fileName)
        cv2.imwrite(fileName, crop_img)
        indexDict[(ord)(labels[i])-ord('A')] += 1
        
def split2SingleNew(imgPath):
    split2SingleWithSplit(imgPath, (10, 60), (30, 80), (50, 100), (70, 120))
    split2SingleWithSplit(imgPath, (0, 50), (25, 75), (55, 105), (75, 125))
    split2SingleWithSplit(imgPath, (5, 55), (35, 85), (60, 110), (65, 115))

def split2SingleWithSplit(imgPath, i1, i2, i3, i4):
    print(imgPath)
    img = cv2.imread(imgPath, 0)
    #img = Util.binaryzation(img)
    '''i1 = (10, 60)
    i2 = (30, 80)
    i3 = (50, 100)
    i4 = (70, 120)'''
    imgs = [i1, i2, i3, i4]
    img_name = imgPath[imgPath.rindex('/') + 1:]
    if img_name.find('_') > -1 :
        labels = img_name.split('_')[0]
    else:
        labels = img_name[:img_name.index('.')]
    for i in range(4):
        crop_img = img[1:51, imgs[i][0]:imgs[i][1]]
        splitChar = "_"
        if ((ord)(labels[i])-ord('a')) >= 0:
             splitChar = "__"
        fileName = Constants.IMG_DIR_TENCENT_SPLIT + labels[i] + splitChar + str(indexDict[(ord)(labels[i])-ord('A')]) + ".jpg"
        if TEST:
            fileName = Constants.IMG_DIR_TENCENT_TEST_SPLIT + labels[i] + splitChar + str(indexDict[(ord)(labels[i])-ord('A')]) + ".jpg"
        #print(fileName)
        cv2.imwrite(fileName, crop_img)
        indexDict[(ord)(labels[i])-ord('A')] += 1
    
    '''return [(ord)(labels[0])-ord('A'), (ord)(labels[1])-ord('A'), (ord)(labels[2])-ord('A'), (ord)(labels[3])-ord('A')]
    
    #对图像进行裁剪，第一个参数是纵坐标范围，第二个参数是横坐标范围
    crop_img = img[1:51, 0:50] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    cv2.imwrite("a.jpg", crop_img)'''

def test(imgPath):
    img = Image.open(imgPath)
    width, height = img.size
    print(str(width) + "-" + str(height))
    for i in range(width):
        #print("")
        sum = 0
        '''for j in range(height):
            if(img.getpixel(i,j) < 150):
                img.setpixel(i,j) = 
            #sum += img.getpixel((i,j))
            #print(str(img.getpixel((i,j))) + " ",end='')
        print(str(sum) + " ",end='')'''
    img.save("../../tencent/test/000.jpg")
    
def wrapAffine():
    img = cv2.imread(Constants.IMG_DIR_TENCENT_TEST + "H.jpg", 0)
    img = Util.binaryzation(img)
    rows,cols = img.shape

    pts1 = np.float32([[0,0],[0,20],[20,0]])
    #这样变换会横向变窄
    #pts2 = np.float32([[0,0],[0,20],[18,0]])
    #这样会纵向变扁
    #pts2 = np.float32([[0,0],[0,18],[20,0]])
    #这样横向变窄，且有一定的变形
    #pts2 = np.float32([[0,0],[0,20],[18,2]])
    #这样纵向变窄，且有一定的变形
    pts2 = np.float32([[0,0],[2,18],[20,0]])
    pts2All = [np.float32([[0,0],[0,20],[18,0]]), np.float32([[0,0],[0,18],[20,0]]), 
               np.float32([[0,0],[0,20],[18,2]]), np.float32([[0,0],[2,18],[20,0]])]
    #pts2 = np.float32([[0,0],[20,20],[20,0]])

    M = cv2.getAffineTransform(pts1,pts2All[0])
    print(M)

    dst = cv2.warpAffine(~img,M,(cols,rows))
    dst = Util.binaryzation(dst)
    cv2.imshow("2", img)
    cv2.imshow("1", ~dst)
    Util.printCV2Image(img)
    Util.printCV2Image(~dst)
    cv2.waitKey(0)
    #plt.subplot(121),plt.imshow(img),plt.title('Input')
    #plt.subplot(122),plt.imshow(dst),plt.title('Output')
    #plt.show()
    
def wrapAffine2(imgPath):
    img = cv2.imread(imgPath, 0)
    rows,cols = img.shape
    img_name = imgPath[imgPath.rindex('/') + 1:]
    label = img_name[:img_name.index('_')]
    print("label:" + label)
    pts1 = np.float32([[0,0],[0,20],[20,0]])
    '''
    41个变换，保存
    '''
    '''pts2All = [np.float32([[0,0],[0,20],[18,0]]), np.float32([[0,0],[0,18],[20,0]]), 
               np.float32([[0,0],[0,20],[18,2]]), np.float32([[0,0],[2,18],[20,0]]),
               np.float32([[0,0],[0,20],[18,1]]), np.float32([[0,0],[1,18],[20,0]]),
               np.float32([[0,0],[0,20],[17,1]]), np.float32([[0,0],[1,17],[20,0]]),
               np.float32([[0,0],[0,20],[16,1]]), np.float32([[0,0],[1,16],[20,0]]),
               np.float32([[0,0],[0,20],[15,1]]), np.float32([[0,0],[1,15],[20,0]]),
               np.float32([[0,0],[0,20],[14,1]]), np.float32([[0,0],[1,14],[20,0]]),
               np.float32([[0,0],[0,20],[13,1]]), np.float32([[0,0],[1,13],[20,0]]),
               np.float32([[0,0],[0,20],[12,1]]), np.float32([[0,0],[1,12],[20,0]]),
               np.float32([[0,0],[0,20],[11,1]]), np.float32([[0,0],[1,11],[20,0]]),
               np.float32([[0,0],[0,20],[10,1]]), np.float32([[0,0],[1,10],[20,0]]),
               np.float32([[0,1],[0,20],[20,0]]), np.float32([[0,2],[0,20],[20,0]]),
               np.float32([[0,3],[0,20],[20,0]]), np.float32([[0,4],[0,20],[20,0]]),
               np.float32([[0,5],[0,20],[20,0]]), np.float32([[0,6],[0,20],[20,0]]),
               np.float32([[0,7],[0,20],[20,0]]), np.float32([[0,8],[0,20],[20,0]]),
               np.float32([[0,9],[0,20],[20,0]]), np.float32([[0,10],[0,20],[20,0]]),
               np.float32([[1,0],[0,20],[20,0]]), np.float32([[2,0],[0,20],[20,0]]),
               np.float32([[3,0],[0,20],[20,0]]), np.float32([[4,0],[0,20],[20,0]]),
               np.float32([[5,0],[0,20],[20,0]]), np.float32([[6,0],[0,20],[20,0]]),
               np.float32([[7,0],[0,20],[20,0]]), np.float32([[8,0],[0,20],[20,0]]),
               np.float32([[9,0],[0,20],[20,0]]), np.float32([[10,0],[0,20],[20,0]])]'''
    '''pts2All = [np.float32([[0,0],[0,20],[18,0]]), np.float32([[0,0],[0,20],[18,2]]), 
               np.float32([[0,0],[2,18],[20,0]]), np.float32([[0,0],[1,18],[20,0]]),
               np.float32([[0,0],[0,20],[17,1]]), np.float32([[0,0],[1,17],[20,0]]),
               np.float32([[0,0],[1,16],[20,0]]), np.float32([[0,0],[0,20],[15,1]]), 
               np.float32([[0,0],[0,20],[14,1]]), np.float32([[0,0],[1,14],[20,0]]),
               np.float32([[0,0],[0,20],[13,1]]), np.float32([[0,0],[1,12],[20,0]]),
               np.float32([[0,0],[0,20],[10,1]]), np.float32([[0,0],[1,10],[20,0]]), 
               np.float32([[0,2],[0,20],[20,0]]), np.float32([[0,3],[0,20],[20,0]]), 
               np.float32([[0,4],[0,20],[20,0]]), np.float32([[0,5],[0,20],[20,0]]), 
               np.float32([[0,6],[0,20],[20,0]]), np.float32([[0,9],[0,20],[20,0]]), 
               np.float32([[2,0],[0,20],[20,0]]), np.float32([[4,0],[0,20],[20,0]]), 
               np.float32([[6,0],[0,20],[20,0]]), np.float32([[8,0],[0,20],[20,0]])]'''
    pts2All = [np.float32([[0,0],[2,18],[20,0]])]
    for i in range(len(pts2All)):
        M = cv2.getAffineTransform(pts1,pts2All[i])
        dst = cv2.warpAffine(~img,M,(cols,rows))
        fileName = Constants.IMG_DIR_TENCENT_SPLIT_ALL + img_name[:img_name.index(".jpg")] + "_" + str(indexDict[(ord)(label)-ord('A')]) + ".jpg"
        if TEST:
            fileName = Constants.IMG_DIR_TENCENT_TEST_SPLIT_ALL + img_name[:img_name.index(".jpg")] + "_" + str(indexDict[(ord)(label)-ord('A')]) + ".jpg"
        cv2.imwrite(fileName, ~dst)
        indexDict[(ord)(label)-ord('A')] += 1

    
    cv2.waitKey(0)
    
def wrapAffine3():
    img = cv2.imread(Constants.IMG_DIR_TENCENT_SPLIT_ALL + "A_0_0.jpg", 0)
    rows,cols = img.shape

    pts1 = np.float32([[0,0],[0,20],[20,0]])
    pts2 = np.float32([[0,0],[0,15],[20,0]])

    M = cv2.getAffineTransform(pts1,pts2)
    print(M)

    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("2", img)
    cv2.imshow("1", dst)
    cv2.waitKey(0)

indexDict = {}

def initDict():
    for i in range(60):
        indexDict[i] = 0

def resizeAllImg(imgPath):
    img = cv2.imread(imgPath, 0)
    #img = Util.erasePaddingInOneChar(img)
    img = Util.resizeImg(img,(28,28))
    img_name = imgPath[imgPath.rindex('/') + 1:]
    fileName = Constants.IMG_DIR_TENCENT_SPLIT_ALL_SMALL + img_name
    if TEST:
        fileName = Constants.IMG_DIR_TENCENT_TEST_SPLIT_ALL_SMALL + img_name
    cv2.imwrite(fileName, img)  
    
TEST = False

if __name__ == "__main__":
    #wrapAffine3()
    initDict()
    #process train data
    Util.readAllImg(Constants.IMG_DIR_TENCENT_NEW_TRAIN, split2SingleNew)
    #initDict()
    #Util.readAllImg(Constants.IMG_DIR_TENCENT_SPLIT, wrapAffine2)
    #Util.readAllImg(Constants.IMG_DIR_TENCENT_SPLIT_ALL, resizeAllImg)
    
    TEST = True
    initDict()
    #process test data
    Util.readAllImg(Constants.IMG_DIR_TENCENT_TEST, split2SingleNew)
    initDict()
    #Util.readAllImg(Constants.IMG_DIR_TENCENT_TEST_SPLIT, wrapAffine2)
    #img = cv2.imread(Constants.IMG_DIR_TENCENT_WHOLE + "a__0.jpg")
    #print(img.shape)
    #Util.readAllImg(Constants.IMG_DIR_TENCENT_TEST_SPLIT_ALL, resizeAllImg)
    #split2Single(Constants.IMG_DIR_TENCENT_TRAIN + "AANV_39b_58b_82b.jpg")
    #wrapAffine()
    #spilt("../../tencent/test/EUZB.jpg")