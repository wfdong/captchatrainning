# -*- coding: UTF-8 -*-
'''
Created on 2014年5月24日

@author: rogers
'''
from Tkinter import *
from PIL import ImageTk, Image
from utils import DLUtil
from utils import Util, Constants

'''
绘制UI主界面，其中要加载的图片都有固定的名称
当前要处理的图片是ori.jpg
layer1的图片依次是l1-1, l1-2...
layer2的图片依次是l2-1, l2-2...
UI的大小为800 * 400
0-300为原图片展示区，可以点击左右选择下一张图片，用fm1表示
300-400为识别结果展示区，显示每个结果识别的概率，用fm2表示
400-800为图片卷积的过程展示区，用fm3表示
之后每个控件在各自的frame里布局就行

'''


allImgs = []

def drawUI():
    global oriImg, curIndex, root
    root = Tk()
    root.geometry('800x400')
    curIndex = 0
    #初始化三个frame
    fm1 = Frame(root,width = 300,height = 400)
    fm2 = Frame(root,width = 100,height = 400)
    fm3 = Frame(root,width = 400,height = 400)
    #把三个frame放到相应的位置上
    fm1.place(x = 0,y = 0,anchor = NW)
    fm2.place(x = 300,y = 0,anchor = NW)
    fm3.place(x = 400,y = 0,anchor = NW)
    #
    # for frame1
    # 创建一个Label,显示原始图像
    #
    img = ImageTk.PhotoImage(Image.open(DLUtil.UI_ORI_IMG))
    oriImg = Label(fm1, image = img)
    oriImg.place(in_ = fm1,relx = 0.5,rely = 0.2,anchor = CENTER)
    btnLeft = Button(fm1, text=' << ', command=clickLeft)
    btnLeft.place(in_ = fm1,relx = 0.2,rely = 0.2,anchor = CENTER)
    btnRight = Button(fm1, text=' >> ', command=clickRight)
    btnRight.place(in_ = fm1,relx = 0.8,rely = 0.2,anchor = CENTER)
    
    # 创建一个Label,用于展示结果
    lb2 = Label(fm2,text = 'Result',fg = 'black')
    lb2.place(in_ = fm2,relx = 0.5,rely = 0.2,anchor = CENTER)
    #创建很多歌label，用于显示训练过程
    for i in range(DLUtil.LAYER_1_IMG_COUNT):
        #get layer one image
        #for init UI, we just pass
        pass
    

    
    root.mainloop()

def clickLeft(event=None):
    global oriImg, curIndex
    print("left")
    if curIndex - 1 < 0:
        print("Has been the leftest")
    else:
        curIndex -= 1
        newImg = ImageTk.PhotoImage(Image.open(allImgs[curIndex]))
        oriImg.configure(image = newImg)
        oriImg.image = newImg
    
def clickRight(event=None):
    global oriImg, curIndex
    print("right")
    if curIndex + 1 > len(allImgs) - 1:
        print("Has been the rightest")
    else:
        curIndex += 1
        newImg = ImageTk.PhotoImage(Image.open(allImgs[curIndex]))
        oriImg.configure(image = newImg)
        oriImg.image = newImg
    


def addImgToArray(imgPath):
    allImgs.append(imgPath)
    

if __name__ == "__main__":
    Util.readAllImg(Constants.IMG_DIR_TENCENT_SPLIT_ALL, addImgToArray)
    print(len(allImgs))
    drawUI()
    
