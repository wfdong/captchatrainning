# -*- coding: UTF-8 -*-
'''
Created on 2014年3月9日

@author: rogers
'''
import numpy as np
import cv2

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

if __name__ == "__main__":
    img = cv2.imread("222.png", 0)
    img = deskew(img)
    cv2.imshow("Image1", img)
    cv2.waitKey (0)
    print("---Finished---")