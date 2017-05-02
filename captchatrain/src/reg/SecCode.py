# -*- coding: UTF-8 -*-
'''
Created on 2014/4/7

@author: rogers
'''
'''
解析传输回来的s.js文件，获取里面就的seccode
'''

import Util, zlib

def getSecCode(response):
    content = response.read()
    #content = unicode(content,'GBK').encode('UTF-8')
    content = zlib.decompress(content, 16+zlib.MAX_WBITS)
    print(content)
    print(type(content))
    list = content.split('\n')
    code, lineNum = getqsCode(list)
    print(code)
    #print(lineNum)
    dict, lineNum = getVariable(list, lineNum)
    print(dict)
    #print(lineNum)
    secCode = gerCalc(code, dict, list, lineNum)
    print("secCode=" + secCode)
    return secCode
    #for line in list:
    #    print(line)

def gerCalc(qs, dict, list, lineNum):
    for i in range(lineNum, len(list)):
        line = list[i]
        index = line.find("zcSec.r +=")
        #print(index)
        if(index >= 0):
            return calcSecCode(qs, dict, line)
            
def calcSecCode(qs, dict, line):
    start = 0
    str = ""
    codeNum = 0
    #print(qs)
    #print(dict)
    while(True):
        keyStart = line.find('(', start)
        keyEnd = line.find('[', keyStart)
        key = line[keyStart + 1: keyEnd]
        index = line[keyEnd + 1]
        #print(index)
        str += qs[int(dict[key][int(index)])]
        #print(str)
        start = keyEnd
        codeNum += 1
        if(codeNum >= 10):
            break
    return str
    
    

def getVariable(list, lineNum):
    dict = {}
    varNum = 0
    for i in range(lineNum, len(list)):
        line = list[i]
        index = line.find("var")
        #print(index)
        if(index >= 0):
            varNum = varNum + 1
            keyStart = line.find("var") + 4
            keyEnd = line.find(" ", keyStart + 2)
            key = line[keyStart: keyEnd]
            valStart = line.find("[")
            valEnd = line.find("]")
            #把value转换为list
            value = line[valStart + 1: valEnd].split(',')
            #print(type(value))
            dict[key] = value
            #print(key)
        #因为只有3个var值，这里就hard code了，到了三个就返回
        if(varNum >= 3):
            return dict, i

'''

'''
def convertStrToList():
       pass 

'''
获得qs，即字母字典
'''
def getqsCode(list):
    #直接取第三行
    line = list[2]
    start = line.find("\"")
    end = line.rfind("\"")
    return line[start + 1: end], list.index(line)

if __name__ == "__main__":
    response = Util.sendSJSGet("http://a.zc.qq.com/s.js?t=0.3910518413083368")
    getSecCode(response)