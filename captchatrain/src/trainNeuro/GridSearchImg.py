# -*- coding: UTF-8 -*-
'''
Created on 2014/3/22

@author: rogers
'''
from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import cv2
import numpy as np
from utils import Constants
from utils import Util
from trainNeuro import ReadImg
from trainNeuro import TrainUseSVM

'''
X是features，大小是这样的，(1797, 64)
y是labels，a.k.a (1797,)
'''
def gridSearch(X, y):
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)
    print("X_train:" + str(X_train.shape))
    print("X_test:" + str(X_test.shape))
    print("y_train:" + str(y_train.shape))
    print("y_test:" + str(y_test.shape))

    # Set the parameters by cross-validation
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3,5e-3,1e-4],
    #                 'C': [1,2,25,100,1000]}]
    Cs = np.logspace(0, 10, 15, base=2)
    gammas = np.logspace(-7, 4, 15, base=2)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gammas,
                     'C': Cs}]

    #scores = ['precision', 'recall']
    scores = ['precision','accuracy']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

def loadBasicDigits():
    digits = datasets.load_digits()
    
    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    print("X:" + str(X.shape))
    print("y:" + str(y.shape))
    return X,y

def getFeatures(target):
    train_cells = target
    deskewed = [map(TrainUseSVM.deskew,row) for row in train_cells]
    hogdata = [map(TrainUseSVM.hog,row) for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1,64)
    return trainData
    
def readAndPreProcessAllImg():
    target, labels = ReadImg.readAllImg(Constants.IMG_DIR_TENCENT_TRAIN)
    #消除有些字母内部的填充
    TrainUseSVM.erasePaddingInAllChar(target)
    new_target, new_labels = Util.rotateAllTarget(target, labels)
    return  getFeatures(new_target), new_labels

def readTarget():
    target = np.loadtxt('target.txt', dtype=np.float)
    return target

def readLabels():
    labels = np.loadtxt('labels.txt', dtype=np.int8)
    return labels
    
if __name__ == "__main__":
    #target, labels = loadBasicDigits()
    target = readTarget()
    labels = readLabels()
    #target, labels = readAndPreProcessAllImg()
    print(target.shape)
    print(labels.shape)
    gridSearch(target, labels)
    pass