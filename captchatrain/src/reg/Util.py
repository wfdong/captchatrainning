# -*- coding: UTF-8 -*-
'''
Created on 2014/3/30

@author: rogers
'''
import urllib, urllib2, cookielib
import time, random
import json
import thread

curName = ""
code = ""

TIMEOUT = 20
    
lock = thread.allocate_lock()  #Allocate a lock   
    
def getTime():
    return (int)(time.time() * 1000)
    
    
def sendPost(url, data):
    print("url=" + url)
    req = urllib2.Request(url, urllib.urlencode(data))  
    print("[POST Request Info]:")
    print(req.header_items())  
    print(req.get_data()) 
    response = urllib2.urlopen(req)
    return response 

def sendPostWithHeader(url, data, headers):
    print("url=" + url)
    req = urllib2.Request(url, urllib.urlencode(data))  
    addCommonHeaders(req)
    print("[POST Request Info]:")
    print(req.header_items())  
    print(req.get_data()) 
    response = urllib2.urlopen(req)
    return response 

def sendPostWithHeaderOpener(url, data, headers, opener):
    print("url=" + url)
    req = urllib2.Request(url, urllib.urlencode(data))  
    addCommonHeaders(req)
    print("[POST Request Info]:")
    print(req.header_items())  
    print(req.get_data()) 
    response = opener.open(req, timeout=TIMEOUT)
    return response 

def sendGet(url):
    print("url=" + url)
    req = urllib2.Request(url)    
    response = urllib2.urlopen(req)    
    return response

def sendGetWithHeader(url):
    print("url=" + url)
    req = urllib2.Request(url)   
    addCommonHeaders(req) 
    response = urllib2.urlopen(req) 
    return response

def sendGetWithHeaderOpener(url, opener):
    print("url=" + url)
    req = urllib2.Request(url)   
    addCommonHeaders(req) 
    response = opener.open(req, timeout=TIMEOUT) 
    return response

def sendPtLoginGet(url):
    print("url=" + url)
    req = urllib2.Request(url)   
    addPtLoginHeader(req) 
    response = None
    try:
        response = urllib2.urlopen(req) 
    except urllib2.URLError as e:
        if hasattr(e, 'code'):
            error_info = e.code
        elif hasattr(e, 'reason'):
            error_info = e.reason
    finally:
        if response:
            response.close()
    return response

def sendPtLoginGet2(url):
    print("url=" + url)
    req = urllib2.Request(url)   
    addPtLoginHeader2(req) 
    response = None
    try:
        response = urllib2.urlopen(req) 
    except urllib2.URLError as e:
        if hasattr(e, 'code'):
            error_info = e.code
        elif hasattr(e, 'reason'):
            error_info = e.reason
    finally:
        if response:
            response.close()
    return response

def sendPtLoginGet3(url):
    print("url=" + url)
    req = urllib2.Request(url)   
    addPtLoginHeader3(req) 
    response = None
    try:
        response = urllib2.urlopen(req) 
    except urllib2.URLError as e:
        if hasattr(e, 'code'):
            error_info = e.code
        elif hasattr(e, 'reason'):
            error_info = e.reason
    finally:
        if response:
            response.close()
    return response
    
def sendMoniKey(url):
    print("url=" + url)
    req = urllib2.Request(url)   
    addMoniKeyHeaders(req) 
    response = urllib2.urlopen(req) 
    return response

def sendMoniKeyOpener(url, opener):
    print("url=" + url)
    req = urllib2.Request(url)   
    addMoniKeyHeaders(req) 
    response = opener.open(req, timeout=TIMEOUT) 
    return response
    
def sendOtherCheckGet(url):
    print("url=" + url)
    req = urllib2.Request(url)   
    addMoniKeyHeaders(req) 
    response = urllib2.urlopen(req) 
    return response

def sendOtherCheckGetOpener(url, opener):
    print("url=" + url)
    req = urllib2.Request(url)   
    addMoniKeyHeaders(req) 
    response = opener.open(req, timeout=TIMEOUT) 
    return response

def sendSJSGet(url):
    print("url=" + url)
    req = urllib2.Request(url)   
    addSJSHeader(req) 
    response = urllib2.urlopen(req) 
    return response
    



'''
保存文件的，主要是验证码图片了
url是验证码的url
targetPath就是图片要保存的路径了，以.jpg结尾的
'''
def saveFileOpener(url, targetPath, opener):
    data = opener.open(url, timeout=TIMEOUT).read()
    f = open(targetPath, 'wb') 
    f.write(data)  
    f.close()
    
def saveFile(url, targetPath):
    data = urllib2.urlopen(url).read()
    f = open(targetPath, 'wb') 
    f.write(data)  
    f.close()
    
cj = cookielib.MozillaCookieJar()

def addSJSHeader(req):
    req.add_header("Accept", "*/*")
    req.add_header("Accept-Encoding", "gzip, deflate")
    req.add_header("Accept-Language", "en-US,en;q=0.5")
    req.add_header("Connection", "keep-alive")
    req.add_header("Host", "a.zc.qq.com")
    req.add_header("Referer", "http://zc.qq.com/chs/index.html")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0")

def getRandom():
    return random.random()

def addPtLoginHeader(req):
    req.add_header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    req.add_header("Accept-Encoding", "gzip, deflate")
    req.add_header("Accept-Language", "en-US,en;q=0.5")
    req.add_header("Connection", "keep-alive")
    req.add_header("Host", "ptlogin2.qq.com")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0")
    
def addPtLoginHeader2(req):
    req.add_header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    req.add_header("Accept-Encoding", "gzip, deflate")
    req.add_header("Accept-Language", "en-US,en;q=0.5")
    req.add_header("Connection", "keep-alive")
    req.add_header("Host", "zc.qq.com")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0")
    
def addPtLoginHeader3(req):
    req.add_header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    req.add_header("Accept-Encoding", "gzip, deflate")
    req.add_header("Accept-Language", "en-US,en;q=0.5")
    req.add_header("Connection", "keep-alive")
    req.add_header("Host", "zc.qq.com:443")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0")

def addMoniKeyHeaders(req):
    req.add_header("Accept", "image/png,image/*;q=0.8,*/*;q=0.5")
    req.add_header("Accept-Encoding", "gzip, deflate")
    req.add_header("Accept-Language", "en-US,en;q=0.5")
    req.add_header("Connection", "keep-alive")
    req.add_header("Host", "a.zc.qq.com")
    req.add_header("Referer", "http://zc.qq.com/chs/index.html")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0")
    
def addCommonHeaders(req):
    req.add_header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    req.add_header("Accept-Encoding", "gzip, deflate")
    req.add_header("Accept-Language", "zh-cn,zh;q=0.8,en-us;q=0.5,en;q=0.3")
    req.add_header("Cache-Control", "no-cache")
    req.add_header("Connection", "keep-alive")
    req.add_header("Content-Type", "text/plain; charset=UTF-8")
    req.add_header("Host", "zc.qq.com")
    req.add_header("Pragma", "no-cache")
    req.add_header("Referer", "http://zc.qq.com/chs/index.html")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0")
    
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-cn,zh;q=0.8,en-us;q=0.5,en;q=0.3',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Type': 'text/plain; charset=UTF-8',
    'Host': 'zc.qq.com',
    'Pragma': 'no-cache',
    'Referer': 'http://zc.qq.com/chs/index.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0'
    }
    
postdata = {
    'verifycode': 'QQQQ',
    'qzone_flag': '1',
    'country': '1',
    'province': '31',
    'city': '1',
    'isnongli': '0',
    'year': '1996',
    'month': '1',
    'day': '2',
    'isrunyue': '0',
    'password': '8b6babcdcb817c1729ef52eeafb04abfc8544bc585b1c955705a8b9c3874031b5f03c5a607b1e290fb21421615e9bc2565fdb2185c0250bd0f9b9b84c220615965b98eb9891a8bbaf1c0f016df05921d72754db42a44993ea3b781beb205d5e1fdd75ef5e1d31c6138d6498a6bd3a5080d07cf78f21f6e6f8874e7870aa8f45b',
    'nick': 'xiaoxie1',
    'email': 'false',
    'other_email': 'false',
    'elevel': '1',
    'sex': '1',
    'qzdate': '',
    'jumpfrom': '58030',
    'csloginstatus': '1'
} 

def setInitData(response, postData):
    content = json.loads(response.read())
    print(content)
    print(type(content))
    #j = json.loads(str(content))
    j = content
    postData["country"] = j["countryid"]
    postData["province"] = j["provinceid"]
    if postData.has_key("cityid"):
        postData["city"] = j["cityid"]
    else:
        postData["city"] = 1

if __name__ == "__main__":
    #response = sendSJSGet("http://a.zc.qq.com/s.js?t=0.3910518413083368")
    str = "{\"city\":\"淄博\",\"cityid\":\"3\",\"country\":\"中国\",\"countryid\":\"1\",\"ec\":0,\"elevel\":\"1\",\"localdate\":\"2014-4-7\",\"province\":\"山东\",\"provinceid\":\"37\"}"
    j = json.loads(str)
    print(j)
    print(j["countryid"])
    print(j["cityid"])
    print(j["provinceid"])
    
