# -*- coding: UTF-8 -*-
'''
Created on 2014/4/15

@author: rogers
'''

import Util, zlib

'''
aq_object are set in m.js
A('l1m0r','x1b1m')
'''
def getAq_Object(response):
    content = response.read()
    #content = zlib.decompress(content, 16+zlib.MAX_WBITS)
    print(content)
    print(type(content))
    index = content.find("A('")
    begin = index + 3
    index = content.find("'", begin)
    end = index
    key = content[begin:end]
    
    index = content.find(",'", end)
    begin = index + 2
    index = content.find("'", begin)
    end = index
    value = content[begin:end]
    
    print("Key:" + key + ", value:" + value)
    return key, value
    

if __name__ == "__main__":
    response = Util.sendGetWithHeader("http://zc.qq.com/chs/m.js?v=0.6149123950831218")
    getAq_Object(response)