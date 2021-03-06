# -*- coding: UTF-8 -*-
'''
Created on 2014/6/13

@author: rogers
'''
"""
用Imagenet的方法，多个卷积层
layer0:(1,50,50) * (48,5,5) - > (48,23,23)
layer1:(1,23,23) * (128,3,3) -> (128,12,12)
layer2:(1,12,12) * (192,3,3) -> (192,5,5)
HiddenLayer:1000
"""

import sys
sys.path.append('/root/zhangda/captcha/src')
import cPickle
import gzip
import os
import sys
import time
import random

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from utils import Util, Constants


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
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
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

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

'''
自己的版本，输入输出会有变化
输入50 * 50
layer0:{image_shape=(batch_size, 1, 48, 48),filter_shape=(nkerns[0], 1, 5, 5), output = (batch_size, nkerns[0], 44/2, 44/2)}
layer1:{image_shape=(batch_size, nkerns[0], 22, 22),filter_shape=(nkerns[1], nkerns[0], 5, 5), output = {batch_size, nkerns[1], 18/2, 18/2}}
最后9*9的有点太大了，好想再加一层conv，5*5的核太大
平滑output
HiddenLayer， input是平滑后的大小，output = 500
LogisticRegression, n_in = 500, n_out = 52
layer0:(1,50,50) * (48,5,5) - > (48,23,23)
layer1:(1,23,23) * (128,3,3) -> (128,12,12)
layer2:(1,12,12) * (192,3,3) -> (192,5,5)
'''
def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    nkerns=[48, 64, 96], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer 这样默认就是第一次20个kernel，第二层50个kernel？
    """

    rng = numpy.random.RandomState(23455)

    '''
             原来的load data
    datasets = load_data(dataset)
    '''
    
    datasets = loadData()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    print(n_train_batches)
    print(n_valid_batches)
    if n_test_batches == 0:
        n_test_batches = 1

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
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 50, 50),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 23, 23),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(2, 2))
    
    layer1_3 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 10, 10),
            filter_shape=(nkerns[2], nkerns[1], 3, 3), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1_3.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[2] * 4 * 4,
                         n_out=1000, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=1000, n_out=58)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1_3.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 1000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    best_params = params

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    saveParams(params)

def saveParams(params):
    save_file = open('../data/params2', 'wb')
    print(len(params))
    print(type(params))
    for p in params:
        cPickle.dump(p.get_value(borrow=True), save_file, -1)
    save_file.close()

def readParams():
    save_file = open('../data/params', 'rb')
    params = []
    for i in range(8):
        #p = theano.shared(0, borrow=True)
        #p.set_value(cPickle.load(save_file), borrow=True)
        p = theano.shared(value=cPickle.load(save_file), borrow=True)
        params.append(p)
        print(p.get_value(borrow=True))
    save_file.close()

allImg = []
allLabels = []
allImageLabels = []
allTestImageLabels = []
testAllImgs = []
testAllLabels = []

def loadData():
    global allImg, allLabels
    save_file = open('../data/db2', 'rb')
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
    trainRange = (0, 60000)
    validRange = (60000, 62400)
    #testRange = (24000, 25000)
    
    train_set_x, train_set_y = shared_dataset(imgs[trainRange[0]:trainRange[1]], labels[trainRange[0]:trainRange[1]])
    valid_set_x, valid_set_y = shared_dataset(imgs[validRange[0]:validRange[1]], labels[validRange[0]:validRange[1]])
    test_set_x, test_set_y = shared_dataset(testImgs, testLabels)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    evaluate_lenet5()
    #readParams()
