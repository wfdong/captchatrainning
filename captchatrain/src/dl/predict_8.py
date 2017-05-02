# -*- coding: UTF-8 -*-
'''
Created on 2014/6/18

@author: rogers
'''

import cPickle
import os

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

#from utils import Util, Constants
import cv2

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = numpy.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        self.params = [self.W, self.b]

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, params, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        self.W = params[0]

        self.b = params[1]

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class LeNetConvPoolLayerNoPooling(object):
    """Pool Layer of a convolutional network """

    def __init__(self, params, input, filter_shape, image_shape):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # initialize weights with random weights
        '''W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)'''
        self.W = params[0]

        self.b = params[1]

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        #pooled_out = downsample.max_pool_2d(input=conv_out,
        #                                    ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, params, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = params[0]
        # initialize the baises b as a vector of n_out 0s
        self.b = params[1]

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

'''
自己的版本，输入输出会有变化
输入50 * 50
layer0:{image_shape=(batch_size, 1, 48, 48),filter_shape=(nkerns[0], 1, 5, 5), output = (batch_size, nkerns[0], 44/2, 44/2)}
layer1:{image_shape=(batch_size, nkerns[0], 22, 22),filter_shape=(nkerns[1], nkerns[0], 5, 5), output = {batch_size, nkerns[1], 18/2, 18/2}}
最后9*9的有点太大了，好想再加一层conv，5*5的核太大
平滑output
HiddenLayer， input是平滑后的大小，output = 500
LogisticRegression, n_in = 500, n_out = 52
'''
def build_lenet5(params, nkerns=[48, 128, 192, 192], batch_size=1):
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (50, 50)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 50, 50))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(params[6], input=layer0_input,
            image_shape=(batch_size, 1, 50, 50),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(params[5], input=layer0.output,
            image_shape=(batch_size, nkerns[0], 23, 23),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(2, 2))
    
    '''layer1_3 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 10, 10),
            filter_shape=(nkerns[2], nkerns[1], 3, 3), poolsize=(2, 2))'''
    
    layer1_3 = LeNetConvPoolLayerNoPooling(params[4], input=layer1.output,
            image_shape=(batch_size, nkerns[1], 10, 10),
            filter_shape=(nkerns[2], nkerns[1], 3, 3))
    
    layer1_4 = LeNetConvPoolLayer(params[3], input=layer1_3.output,
            image_shape=(batch_size, nkerns[2], 8, 8),
            filter_shape=(nkerns[3], nkerns[2], 3, 3), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1_4.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(None, input=layer2_input, n_in=nkerns[3] * 3 * 3,
                         n_out=1920, W=params[2][0] * 0.5, b=params[2][1], activation=ReLU)
    
    layer2_2 = HiddenLayer(None, input=layer2.output, n_in=1920,
                         n_out=1920, W=params[1][0] * 0.5, b=params[1][1], activation=ReLU)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(params[0], input=layer2_2.output, n_in=1920, n_out=58)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    #predict_model = theano.function([x], layer3.y_pred)
    predict_model = theano.function([x], layer3.p_y_given_x)
    
    return predict_model

def predict(predict_model, input):
    ###############
    #    PREDICT  #
    ###############

    #start_time = time.clock()
    
    result = predict_model(input)
    #print(chr(result + ord('A')))
    #end_time = time.clock()
    return chr(result + ord('A'))
    
def validate(predict_model, input, label):
    global num
    result = predict_model(input)
    if result == label:
        print("Success : " + chr(result + ord('A')))
        num = num + 1
    else:
        print("Failed: Correct is " + chr(label + ord('A')) + " but is " + chr(result + ord('A')))
    

def readParams():
    save_file = open('../data/params-8--6', 'rb')
    params = []
    for i in range(7):
        w = theano.shared(value=cPickle.load(save_file), borrow=True)
        b = theano.shared(value=cPickle.load(save_file), borrow=True)
        params.append((w,b))
    save_file.close()
    return params

'''
预测一个image
基本思想是每个字母有三个分割图，分别计算三个分割图的概率，最后把三个概率相加，取综合最大的那个字母
'''
def predictOneImg(predict_model, imgPath):
    img = cv2.imread(imgPath, 0)
    ret = ""
    '''s1 = [(0,50), (2, 52), (5, 55), (7, 57), (10, 60)]
    s2 = [(25, 75), (30, 80), (35, 85)]
    s3 = [(50, 100), (55, 105), (60, 110)]
    #s3 = [(50, 100)]
    s4 = [(65, 115), (70, 120), (75, 125)]'''
    s1 = [(0,50), (5, 55), (10, 60)]
    s2 = [(25, 75), (30, 80), (35, 85)]
    s3 = [(50, 100), (55, 105), (60, 110)]
    #s3 = [(50, 100)]
    s4 = [(65, 115), (70, 120), (75, 125)]
    ranges = [s1, s2, s3, s4]
    splitImgs = []
    for i in range(4):
        prob = []
        for split in ranges[i]:
            crop_img = img[1:51, split[0]:split[1]]
            crop_img = numpy.asarray(crop_img, dtype='float64') / 256.
            crop_img = crop_img.reshape(50 * 50)
            result = predict_model(numpy.asarray([crop_img]))
            #if i == 0:
            #    print(result)
            prob.append(result)
        all = 0
        for p in prob:
            all += p
        #if i == 0:
        #    print(all)
        maxIndex = numpy.argmax(all)
        #print(maxIndex)
        #print(chr(maxIndex + ord('A')))
        ret += chr(maxIndex + ord('A'))
        
    return ret

def getFormattedCropImg(cropImg):
    dst = numpy.ones((50, 50), dtype='int8') * 240
    ori = numpy.asarray(cropImg, dtype='int8')
    width, height = ori.shape
    for i in range(width):
        for j in range(height):
            dst[i][j] = ori[i][j]
    dst = numpy.asarray(dst, dtype='float64') / 256.
    dst = dst.reshape(50 * 50)
    return dst

initStep = 20
step = 10

def predictLeft(oriImg, index, start, retList, oldRet):
    if index == 4:
        retList.append(oldRet)
        return None
    for j in range(3):
        newStart = start + initStep + step * j
        newStart = min(newStart, 130)
        crop_img = oriImg[1:51, start:newStart]
        img = getFormattedCropImg(crop_img)
        #print(img.shape)
        ret = predict_model(numpy.asarray([img]))
        
        copyedOldRet = oldRet[:]
        copyedOldRet.extend(ret)
        predictLeft(oriImg, index + 1, newStart, retList, copyedOldRet)
        #oldRet.extend(ret)
def predictUseVitebi(predict_model, imgPath):
    img = cv2.imread(imgPath, 0)
    retList = []
    oldRet = []
    
    start = 10
    predictLeft(img, 0, 10, retList, oldRet)
    newRet = []
    #for one in retList:
    #    print(len(one))
    for one in retList:
        sum = 0
        ret = ""
        for p in one:
            sum += numpy.max(p)
            ret += chr(numpy.argmax(p) + ord('A'))
        newRet.append((sum, ret))    
    for r in newRet:
        print(r)    
    #print(retList)    
    return None

def validateOneImage(predict_model, imgPath):
    img_name = imgPath[imgPath.rindex('/') + 1:]
    if img_name.find('_') > -1 :
        labels = img_name.split('_')[0]
    else:
        labels = img_name[:img_name.index('.')]
    predict = predictOneImg(predict_model, imgPath)
    accuracy = 0
    for i in range(4):
        if labels[i] == predict[i]:
            accuracy += 25
    return (labels, predict, accuracy)

def getPredictModel():
    params = readParams()
    predict_model = build_lenet5(params)
    return predict_model

def printRet(ret):
    print("Labels : %s, Predict : %s, Accuracy : %d" % (ret[0], ret[1], ret[2]))
        
def countRet(retList):
    accuracy = [0, 0, 0, 0, 0]
    for ret in retList:
        accuracy[ret[2]/25] += 1
    print(accuracy)
    
def predictAllTest(predict_model, dir_path):
    retList = []
    for lists in os.listdir(dir_path): 
        path = os.path.join(dir_path, lists) 
        if not os.path.isdir(path): 
            ret = validateOneImage(predict_model, path)
            printRet(ret)
            retList.append(ret)
    countRet(retList)
    
if __name__ == "__main__":
    params = readParams()
    #save_file = open('../data/db', 'rb')
    #allImageLabels = cPickle.load(save_file)
    predict_model = build_lenet5(params)
    #predictAllTest(predict_model, Constants.IMG_DIR_TENCENT_TEST)
    #ret = validateOneImage(predict_model, Constants.IMG_DIR_TENCENT_TEST + "ETMH.jpg")
    #ret = predictUseVitebi(predict_model, Constants.IMG_DIR_TENCENT_TEST + "AeAN.jpg")
    #printRet(ret)
    #ret = predictOneImg(predict_model, Constants.IMG_DIR_TENCENT_REG_WRONG + "SXWH.jpg")
    #ret = getProb(predict_model, Constants.IMG_DIR_TENCENT_REG_WRONG + "ABHH.jpg")
    #Util.readAllImg(Constants.IMG_DIR_TENCENT_TEST_SPLIT, addOneToTest)
    #for img in allTestImageLabels:
    #    validate(predict_model, numpy.asarray([img[0]]), img[1])
        #predict(predict_model, numpy.asarray([img[0]]))
    #img = numpy.asarray([allImageLabels[0][0]])
    #evaluate_lenet5(params, img)
