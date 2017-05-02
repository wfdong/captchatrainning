# -*- coding: UTF-8 -*-
'''
Created on 2014/3/26

@author: rogers
'''
import numpy as np

'''
把target数组写入文件
mode='w'是重写文件，mode='a'是追加文件
'''
def writeTarget(content):
    np.savetxt('../../data/target.txt', content)
    
def writeLabels(content):
    np.savetxt('../../data/labels.txt', content)

def readTarget():
    target = np.loadtxt('../../data/target.txt', dtype=np.float)
    return target

def readLabels():
    labels = np.loadtxt('../../data/labels.txt', dtype=np.int8)
    return labels

if __name__ == "__main__":
    #l = readTarget()
    l = readLabels()
    print(l)
