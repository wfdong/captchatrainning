# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from utils import Constants
from utils import Util
import ReadImg

SZ=20
bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def erasePaddingInAllChar(target):
    for img in target:
        for char in img:
            char = Util.erasePaddingInOneChar(char)
            
def train(target, labels):
    # First half is trainData, remaining is testData
    train_cells = target
    ######     Now training      ########################
    deskewed = [map(deskew,row) for row in train_cells]
    hogdata = [map(hog,row) for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1,64)
    print(trainData.shape)
    #response is array of n * 1, i.e. [[1],[2],[3]...]
    responses = np.float32(labels)[:,np.newaxis]
    svm = cv2.SVM()
    svm.train(trainData,responses, params=svm_params)
    #svm.save('svm_data.dat')
    
    ######     Now testing      ########################
    deskewed = [map(deskew,row) for row in train_cells]
    hogdata = [map(hog,row) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1,bin_n*4)
    result = svm.predict_all(testData)
    #print(len(result))
    #print(len(responses))
    #print(type(result))
    #print(result.size)
    #print(responses)
    
    '''for d in testData:
        svm.predict(d)
        print(svm.get_var_count())'''
    #######   Check Accuracy   ########################
    mask = result==responses
    
    #for c in result:
    #    print(chr(c + ord('A')))
    mask = mask.astype(np.uint8)
    #print(mask)
    correct = np.count_nonzero(mask)
    #print(correct)
    print(correct*100.0/result.size)
    
def trainInMyWay(target, labels):
    # First half is trainData, remaining is testData
    train_cells = target
    ######     Now training      ########################
    deskewed = [map(deskew,row) for row in train_cells]
    hogdata = [map(hog,row) for row in deskewed]
    trainData = np.float32(train_cells).reshape(-1,64)
    print(trainData.shape)
    #response is array of n * 1, i.e. [[1],[2],[3]...]
    responses = np.float32(labels)[:,np.newaxis]
    svm = cv2.SVM()
    svm.train(trainData,responses, params=svm_params)
    #svm.save('svm_data.dat')
    
    ######     Now testing      ########################
    testData = trainData
    result = svm.predict_all(testData)
    
    #######   Check Accuracy   ########################
    mask = result==responses
    
    #for c in result:
    #    print(chr(c + ord('A')))
    mask = mask.astype(np.uint8)
    #print(mask)
    correct = np.count_nonzero(mask)
    #print(correct)
    print(correct*100.0/result.size)
    
'''
这里的target是个四维数组，现在的是43 * 4 * 20 * 20
43 是train文件夹下图片的个数
4是每张图片有4个字符
20 * 20 是每个字符的尺寸
43 和 4 这两个是list， 20 * 20那是numpy array
'''
def trainSVM(target, labels):
    
    print(len(target))
    print(len(target[0]))
    # First half is trainData, remaining is testData
    train_cells = target
    ######     Now training      ########################
    deskewed = [map(deskew,row) for row in train_cells]
    hogdata = [map(hog,row) for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1,64)
    #print trainData.shape
    responses = np.float32(labels)[:,np.newaxis]
    svm = cv2.SVM()
    svm.train(trainData,responses, params=svm_params)
    svm.save('svm_data.dat')
    return svm

'''
这里的target是个四维数组，现在的是43 * 4 * 20 * 20
43 是train文件夹下图片的个数
4是每张图片有4个字符
20 * 20 是每个字符的尺寸
43 和 4 这两个是list， 20 * 20那是numpy array
'''
def trainSVMInMyWay(target, labels):
    
    print(len(target))
    print(len(target[0]))
    # First half is trainData, remaining is testData
    train_cells = target
    ######     Now training      ########################
    trainData = np.float32(train_cells).reshape(-1,400)
    #print trainData.shape
    responses = np.float32(labels)[:,np.newaxis]
    svm = cv2.SVM()
    svm.train(trainData,responses, params=svm_params)
    svm.save('svm_data.dat')
    return svm

if __name__ == "__main__":
    print("start!")
    target, labels = ReadImg.readAllImg(Constants.IMG_DIR_TENCENT_TRAIN)
    #消除有些字母内部的填充
    erasePaddingInAllChar(target)
    target, labels = Util.rotateAllTarget(target, labels)
    trainInMyWay(target, labels)
    cv2.waitKey(0)