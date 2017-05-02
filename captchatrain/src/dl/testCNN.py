# -*- coding: UTF-8 -*-
'''
Created on 2014/5/24

@author: rogers
'''
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:55:26 2014

@author: rachel

Function: convolution option of two pictures with same size (width,height)
input: 3 feature maps (3 channels <RGB> of a picture)
convolution: two 9*9 convolutional filters
"""

from theano.tensor.nnet import conv
import theano.tensor as T
import numpy, theano
from utils import DLUtil


rng = numpy.random.RandomState(23455)

# symbol variable
input = T.tensor4(name = 'input')

# initial weights
w_shape = (2,3,9,9) #2 convolutional filters, 3 channels, filter shape: 9*9
w_bound = numpy.sqrt(3*9*9)
'''

'''
W = theano.shared(numpy.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound,size = w_shape),
                                dtype = input.dtype),name = 'W')
print(type(W))
print(W.shape)   
print(w_bound) 

b_shape = (2,)
b = theano.shared(numpy.asarray(rng.uniform(low = -.5, high = .5, size = b_shape),
                                dtype = input.dtype),name = 'b')
                                
conv_out = conv.conv2d(input,W)

#T.TensorVariable.dimshuffle() can reshape or broadcast (add dimension)
#dimshuffle(self,*pattern)
# >>>b1 = b.dimshuffle('x',0,'x','x')
# >>>b1.shape.eval()
# array([1,2,1,1])
'''
这里用sigmod是有特殊原因的，因为W是uniform过的，都是小数，如果这里不用sigmod的话值就很小了，而且sigmod也能过滤掉一些很大的值吧
'''
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
f = theano.function([input],output)





# demo
import pylab
from PIL import Image
# open random image of dimensions 639x516
img = Image.open(DLUtil.UI_ORI_IMG)
width1,height1 = img.size
img = numpy.asarray(img, dtype='float64') / 256.
print("oriimg : " + str(img.shape))
# put image in 4D tensor of shape (1, 3, height, width)

img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, height1, width1)
print("img : " + str(img_.shape))
filtered_img = f(img_)
print(type(filtered_img))
print(filtered_img.shape)

for i in range(10):
    print(filtered_img[0][0][0][i])

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()