# -*- coding: UTF-8 -*-
'''
Created on 2014/6/14

@author: rogers
'''
import cPickle
import time

import numpy, random

import theano
import theano.tensor as T

from convolutional_mlp import LeNetConvPoolLayer  
from utils import Constants

IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50

allImg = []
allLabels = []
allImageLabels = []
allTestImageLabels = []
testAllImgs = []
testAllLabels = []

def saveParams(epoch_counter, params):
    print("Begin save params")
    save_file = open('../data/params-8-' + str(epoch_counter), 'wb')
    print(len(params))
    print(type(params))
    for p in params:
        cPickle.dump(p.get_value(borrow=True), save_file, -1)
    save_file.close()
    print("-----------------------------")
    print("save params done")
    print("-----------------------------")

def loadData(path, trainRange, validRange):
    global allImg, allLabels
    save_file = open(path, 'rb')
    allImageLabels = cPickle.load(save_file)
    allTestImageLabels = cPickle.load(save_file)
    random.shuffle(allImageLabels)
    for item in allImageLabels:
        allImg.append(item[0])
        allLabels.append(item[1])
    imgs = numpy.asarray(allImg)
    labels = numpy.asarray(allLabels)
    print(imgs.shape)
    print(labels.shape)
    for item in allTestImageLabels:
        testAllImgs.append(item[0])
        testAllLabels.append(item[1])
    testImgs = numpy.asarray(testAllImgs)
    testLabels = numpy.asarray(testAllLabels)
    print(testImgs.shape)
    print(testLabels.shape)
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    #trainRange = (0, 75000)
    #validRange = (75000, 80000)
    #testRange = (24000, 25000)
    
    train_set_x, train_set_y = shared_dataset(imgs[trainRange[0]:trainRange[1]], labels[trainRange[0]:trainRange[1]])
    valid_set_x, valid_set_y = shared_dataset(imgs[validRange[0]:validRange[1]], labels[validRange[0]:validRange[1]])
    test_set_x, test_set_y = shared_dataset(testImgs, testLabels)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def readParams(path, count):
    save_file = open(path, 'rb')
    params = []
    for i in range(count):
        w = theano.shared(value=cPickle.load(save_file), borrow=True)
        b = theano.shared(value=cPickle.load(save_file), borrow=True)
        params.append((w,b))
    save_file.close()
    return params

def getTime():
    return time.strftime('%Y-%m-%d-%X',time.localtime(time.time()))

def convLayer0(input2, nkerns=[20, 50]):
    rng = numpy.random.RandomState(23455)
    x = T.matrix('x')   # the data is presented as rasterized images
    layer0_input = x.reshape((1, 1, 50, 50))
    print(type(layer0_input))
    print(layer0_input.ndim)

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(1, 1, IMAGE_WIDTH, IMAGE_HEIGHT),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
    
    f = theano.function([x], layer0.output)
    print(type(input2))
    output = f(input2)
    print(output.shape)
    for k in range(20):
        for i in range(23):
            for j in range(23):
                output[0][k][i][j] = output[0][k][i][j] * 256
    for i in range(20):
        cv2.imwrite(Constants.IMG_DIR_TENCENT_SPLIT + str(i) + "111.jpg", output[0][i])
    for i in range(22):
        for j in range(22):
            print(output[0][0][i][j])

if __name__ == "__main__":
    #img is (50, 50)
    '''img = cv2.imread(Constants.IMG_DIR_TENCENT_SPLIT + "a__0.jpg", 0)
    import random
    a = [(1,2),(3,4),(5,6),(7,8)]
    random.shuffle(a)
    print(a) '''
    '''img = Util.binaryzation(oriImg)
    #img = ~img
    img = numpy.asarray(img, dtype='float64') / 256.
    print(img.shape)
    i2 = img.reshape(1,2500)
    print(i2.shape)
    shared_x = theano.shared(numpy.asarray(i2,dtype=theano.config.floatX), borrow=True)
    #print(input3.shape)
    convLayer0(i2)'''
    pass