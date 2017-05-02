# -*- coding: UTF-8 -*-
'''
# Created on 2014/4/10
手动更正错误的照片，再考吧到特定的文件夹

@author: rogers
'''
import threading, cookielib
import datetime, sys
from Queue import Queue
from Tkinter import *
from PIL import ImageTk, Image
import pygame
import os, shutil
from utils import Constants


def moveFileto(sourceDir,  targetDir):#复制指定文件到目录 
    shutil.move(sourceDir,  targetDir) 
    
queue = Queue()
regQueue = Queue()
codes = {}

curIndex = 0

def click(event=None):
    global curIndex
    
    code = input.get()
    print(code)
    changeImg(curIndex + 1)
    moveFileto(allImgs[curIndex], Constants.IMG_DIR_TENCENT_REG_UPDATE + code + "_" + str(curIndex) + ".jpg")
    curIndex += 1
    input.delete(0, END)
    

def changeImg(curIndex):
    newImg = ImageTk.PhotoImage(Image.open(allImgs[curIndex]))
    panel.configure(image = newImg)
    panel.image = newImg
    imgPath = allImgs[curIndex]
    img_name = imgPath[imgPath.rindex('/') + 1:]
    if img_name.find('_') > -1 :
        labels = img_name.split('_')[0]
    else:
        labels = img_name[:img_name.index('.')]
    win.title(labels)
    
allImgs = []
def getAllImg(dir_path):
    max = 1000
    index = 0
    for lists in os.listdir(dir_path): 
        path = os.path.join(dir_path, lists) 
        if not os.path.isdir(path): 
            allImgs.append(path)
            index += 1
            if index > max:
                break
            
if __name__ == "__main__":
    getAllImg(Constants.IMG_DIR_TENCENT_REG_WRONG_OLD)
    win = Tk()
    img = ImageTk.PhotoImage(Image.open(allImgs[curIndex]))
    panel = Label(win, image = img)
    panel.pack(expand=YES, fill=BOTH)
    input = Entry(win,text='输入')
    input.pack(expand=YES, fill=BOTH)
    btn = Button(win, text='OK', command=click)
    btn.pack(expand=YES, fill=BOTH)
    win.bind('<Return>', click)
    win.mainloop()
            
