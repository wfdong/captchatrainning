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
import os
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
import signal

import Model, DLUtil

G_params = []


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
def evaluate_lenet5(params, learning_rate=0.1, n_epochs=120,
                    nkerns=[48, 128, 192, 192], batch_size=500):
    global G_params
    
    datasets = DLUtil.loadData('../data/db4', (0, 80000), (80000, 82350))

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
    layer0 = Model.LeNetConvPoolLayer(params[5], input=layer0_input,
            image_shape=(batch_size, 1, 50, 50),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = Model.LeNetConvPoolLayer(params[4], input=layer0.output,
            image_shape=(batch_size, nkerns[0], 23, 23),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(2, 2))
    
    layer1_3 = Model.LeNetConvPoolLayerNoPooling(params[3], input=layer1.output,
            image_shape=(batch_size, nkerns[1], 10, 10),
            filter_shape=(nkerns[2], nkerns[1], 3, 3))
    
    layer1_4 = Model.LeNetConvPoolLayer(params[2], input=layer1_3.output,
            image_shape=(batch_size, nkerns[2], 8, 8),
            filter_shape=(nkerns[3], nkerns[2], 3, 3), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1_4.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = Model.HiddenLayer(None, input=layer2_input, n_in=nkerns[3] * 3 * 3,
                         n_out=1920, W=params[1][0], b=params[1][1], activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = Model.LogisticRegression(params[0], input=layer2.output, n_in=1920, n_out=58)

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
    params = layer3.params + layer2.params + layer1_4.params + layer1_3.params + layer1.params + layer0.params
    G_params = params
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
    patience_increase = 2  
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience / 2)

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
    print("Begin save params")
    save_file = open('../data/params' + time.strftime('%Y-%m-%d-%X',time.localtime(time.time())), 'wb')
    print(len(params))
    print(type(params))
    for p in params:
        cPickle.dump(p.get_value(borrow=True), save_file, -1)
    save_file.close()
    print("-----------------------------")
    print("save params done")
    print("-----------------------------")
    
def sigTermHandler(sig, func=None):
    global G_params
    print("SIGTERM received, call save params")
    saveParams(G_params)
    

if __name__ == '__main__':
    #SIGTERM  kill -15 pid
    signal.signal(signal.SIGTERM, sigTermHandler)
    params = DLUtil.readParams('../data/params-5', 6)
    evaluate_lenet5(params)
    
    #readParams()
    print(time.strftime('%Y-%m-%d-%X',time.localtime(time.time())) )
