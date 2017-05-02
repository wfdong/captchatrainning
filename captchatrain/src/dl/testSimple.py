# -*- coding: UTF-8 -*-
'''
Created on 2014/5/24

@author: rogers
'''

from theano.tensor.nnet import conv
import theano.tensor as T
import numpy, theano
from utils import DLUtil
import pylab
from PIL import Image

'''
要输出卷积层的结果的话，每个卷继层一个function就可以了
layer1
layer2分别是一个function，没问题的
'''
def test():
    rng = numpy.random.RandomState(23455)

    # symbol variable
    input = T.tensor4(name = 'input')

    # initial weights
    w_shape = (2,3,9,9) #2 convolutional filters, 3 channels, filter shape: 9*9
    w_bound = numpy.sqrt(3*9*9)
    w = numpy.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound,size = w_shape))
    print(type(w))
    print(w.shape)   
    print(w_bound)   
    b_shape = (2,)
    b = numpy.asarray(rng.uniform(low = -.5, high = .5, size = b_shape))
    
    conv_out = conv.conv2d(input,w)


    #output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
    f = theano.function([input],conv_out)
    
    print(type(b))
    print(b.shape) 
    img1 = Image.open(DLUtil.UI_ORI_IMG)
    width1,height1 = img1.size
    print(img1.size)
    img1 = numpy.asarray(img1, dtype = 'float32')/256. # (height, width, 3)
    print(img1.shape)
    img1_rgb = img1.swapaxes(0,2).swapaxes(1,2).reshape(1,3,height1,width1) #(3,height,width)
    print(img1_rgb.shape)
    filtered_img = f(img1_rgb)
    print(type(filtered_img))
    print(filtered_img.shape)
    for i in range(10):
        print(filtered_img[0][0][0][i])

def test2():
    input = T.vector(name = 'input')
    w = numpy.asarray([2,3,4])
    conv_out = conv.conv2d(input,w)
    print(type(conv_out))
    f = theano.function(input, conv_out)
    
#test2()
def test3():
    url = "http://a/folder/another_folder/file_name"
    if url.find("http://") >= 0:
        url = url[url[9:].find("/") + 9:]
    print(url)
test3()

