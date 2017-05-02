# -*- coding: UTF-8 -*-
'''
Created on 2014/2/22
识别一个字符，需要进行的操作
for -30 : 30 目标字符在左旋和右旋30度的范围内进行检测，即检测60次
    1.获取目标字符的边框，按边框提取目标字符，然后进行缩放（放大或缩小，因为我们神经网络训练的图像都是缩放到指定大小的）
    2.与已经保存好的匹配图像一一进行比较，选出结果最小的即可（比较的方式就是两个图像相减的结果再开平方，平方不平方貌似也无所谓）
@author: rogers
'''
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img