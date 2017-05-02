# -*- coding: UTF-8 -*-
'''
Created on 2014/6/18

@author: rogers
'''

import sys
sys.path.append('/root/zhangda/captcha/src')
import cPickle
import os
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from mlp import HiddenLayer
from utils import Util, Constants
import cv2


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
def build_lenet5(params, nkerns=[48, 64, 96], batch_size=1):
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
    layer0 = LeNetConvPoolLayer(params[4], input=layer0_input,
            image_shape=(batch_size, 1, 50, 50),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(params[3], input=layer0.output,
            image_shape=(batch_size, nkerns[0], 23, 23),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(2, 2))

    layer1_3 = LeNetConvPoolLayer(params[2], input=layer1.output,
            image_shape=(batch_size, nkerns[1], 10, 10),
            filter_shape=(nkerns[2], nkerns[1], 3, 3), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1_3.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(None, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=960, W=params[1][0], b=params[1][1], activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(params[0], input=layer2.output, n_in=960, n_out=58)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    predict_model = theano.function([x], layer3.y_pred)
    
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
    

        
ret = []

allTestImageLabels = []
num = 0

def addOneToTest(imgPath):
    global allImg, allLabels
    #read img
    img = cv2.imread(imgPath, 0)
    img = numpy.asarray(img, dtype='float64') / 256.
    newImg = img.reshape(50 * 50)
    #allImg.append(newImg)
    #read labels
    index = imgPath.index('_')
    c = imgPath[index - 1:index]
    label = ord(c) - ord('A')
    l = numpy.asarray(label)
    #allLabels.append(l)
    allTestImageLabels.append((newImg, l))

def readParams():
    save_file = open('../data/params-2', 'rb')
    params = []
    for i in range(5):
        w = theano.shared(value=cPickle.load(save_file), borrow=True)
        b = theano.shared(value=cPickle.load(save_file), borrow=True)
        params.append((w,b))
    save_file.close()
    return params

def predictOneImg(predict_model, imgPath):
    img = cv2.imread(imgPath, 0)
    i1 = (0, 50)
    i2 = (25, 75)
    i3 = (55, 105)
    i4 = (75, 125)
    ranges = [i1, i2, i3, i4]
    splitImgs = []
    for i in range(4):
        crop_img = img[1:51, ranges[i][0]:ranges[i][1]]
        crop_img = numpy.asarray(crop_img, dtype='float64') / 256.
        crop_img = crop_img.reshape(50 * 50)
        splitImgs.append(numpy.asarray([crop_img]))
    ret = ""
    for img in splitImgs:
        c = predict(predict_model, img)
        ret += c
    return ret

def getPredictModel():
    params = readParams()
    predict_model = build_lenet5(params)
    return predict_model
        
if __name__ == "__main__":
    params = readParams()
    #save_file = open('../data/db', 'rb')
    #allImageLabels = cPickle.load(save_file)
    predict_model = build_lenet5(params)
    ret = predictOneImg(predict_model, Constants.IMG_DIR_TENCENT_TEST + "AhHV.jpg")
    #Util.readAllImg(Constants.IMG_DIR_TENCENT_TEST_SPLIT, addOneToTest)
    #for img in allTestImageLabels:
    #    validate(predict_model, numpy.asarray([img[0]]), img[1])
        #predict(predict_model, numpy.asarray([img[0]]))
    print(ret)
    #img = numpy.asarray([allImageLabels[0][0]])
    #evaluate_lenet5(params, img)