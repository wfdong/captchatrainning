# -*- coding: UTF-8 -*-
'''
# Created on 2014/4/10

@author: rogers
'''
import threading, sys
import RegTestWithVC2, Util
from Queue import Queue
from dl import predict_8
import shutil, cv2, numpy, time
from utils import Constants

class RegThread(threading.Thread):
   
    def __init__(self, model, lock):
        threading.Thread.__init__(self)
        self.model = model
        self.lock = lock
    
    def run(self):
        global queue
        while True:
            '''proxy = queue.get()
            print("---------------------------")
            print("Use proxy : " + proxy)
            print("---------------------------")
            '''
            #try:
            reg = RegTestWithVC2.RegQQ()
            try:
                reg.prepareRegister(self.getName(), None, predict, model, lock)
            except :
                print "Unexpected error:", sys.exc_info()
                
            time.sleep(1)


def getProxy():
    proxys = []
    global queue
    for line in open("proxy.txt"):
        proxys.append(line[:len(line) - 1])
        queue.put(line[:len(line) - 1])
    return proxys

def writeToFile(content):
    myfile = open('ret.txt', 'a')            # open for output (creates)
    myfile.write(content)        # write a line of text
    myfile.close()


queue = Queue()
regQueue = Queue()
codes = {}

def moveFileto(sourceDir,  targetDir):#复制指定文件到目录 
    shutil.copy(sourceDir,  targetDir) 
    
def predict(model, imgPath):
    return predict_8.predictOneImg(model, imgPath)



model = None
if __name__ == "__main__":
    #global model
    
    #proxys = getProxy()
    threadNum = 5
    lock = threading.Lock()
    model = predict_8.getPredictModel()
    for i in range(threadNum):
        t = RegThread(model, lock)
        t.start()

            
