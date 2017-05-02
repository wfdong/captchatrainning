# -*- coding: UTF-8 -*-
'''
Created on 2014/3/30

@author: rogers
'''

import Util, cookielib, SecCode, Aq_Object
import urllib, urllib2, time
import Main, sys
from utils import Constants

class RegQQ:
    def __init__(self):
        pass
    proxy = ""
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
    
    def setProxy(self, proxy):
        self.proxy = proxy
        
    def initPage(self, threadName, opener):
        Util.sendGetWithHeaderOpener("http://reg.qq.com/", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com:443/cgi-bin/common/new_router", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/chs/index.html", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/chs/ver.js?v=0.13725353162646503", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/common/attr?id=252190&r=0.7902983111604577", opener)
        response = Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/chs/numreg/init?r=0.7889855697086484&cookieCode=undefined", opener)
        Util.setInitData(response, self.postdata)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/common/attr?id=173276&r=0.9579917662197109", opener)
        response = Util.sendOtherCheckGetOpener("http://a.zc.qq.com/s.js?t=0.62307900677164", opener)
        secCode = SecCode.getSecCode(response)
        Util.saveFileOpener("http://captcha.qq.com/getimage?aid=1007901&r=0.27110232775538023", Constants.IMG_DIR_TENCENT_REG + threadName + ".jpg", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/common/attr?id=58030&timeused=0&seed=0.029119366533932545", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/common/attr?id=260714&r=0.13299904093872916", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/common/attr?id=278037&r=0.8244298933714894", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/common/attr?id=173279&r=0.10828284929336218", opener)
        Util.sendOtherCheckGetOpener("http://a.zc.qq.com/Cgi-bin/SecCheck?" + secCode + "&0.5102277627441705", opener)
        response = Util.sendGetWithHeaderOpener("http://zc.qq.com/chs/m.js?v=0.6149123950831218", opener)
        aq_key, aq_value = Aq_Object.getAq_Object(response)
        self.postdata[aq_key] = aq_value
        
    def inputValues(self, opener):
        beginTime = Util.getTime()
        #休眠3.5秒再发送下个请求
        time.sleep(3.5)
        #nick
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?0|1|" + str(Util.getTime()), opener)
        #time.sleep(0.2)
        
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/chs/common/dirty_check?nick=xiaoxie1&regType=1&r=0.6158950822767871", opener)
        time.sleep(2.41)
        mouseTime2 = Util.getTime()
        time.sleep(1.73)
        #password
        url = self.getNickLongUrl()
        time.sleep(1.1)
        Util.sendMoniKeyOpener(url, opener)
        time.sleep(1.2)
        
        url = self.getNickLongUrl()
        time.sleep(1.1)
        Util.sendMoniKeyOpener(self.getPwdLongUrl1(), opener)
        time.sleep(1.3)
        
        url = self.getNickLongUrl()
        time.sleep(1.1)
        Util.sendMoniKeyOpener(self.getPwdLongUrl2(), opener)
        time.sleep(1.4)
        
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|7|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|8|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|9|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|10|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|11|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|12|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|13|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|14|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?9|15|" + str(Util.getTime()), opener)
        time.sleep(0.35)
        Util.sendMoniKeyOpener("http://a.zc.qq.com/Cgi-bin/MoniKey?69|16|1396438525187&84|16|1396438525609&16|16|1396438526037&90|16|1396438526251&75|16|1396438526611", opener)
        
        Util.sendOtherCheckGetOpener("http://a.zc.qq.com/Cgi-bin/mouse?|0|16|" + str(beginTime) + "|153-154-268-380-379-461-569-579-582-578-578-578-578-578-578-578|0-5-155-197-197-295-191-171-169-183-228-268-338-440-494-542|0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0|0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0|216-216-216-216-216-298-298-298-298-298-298-298-298-298-298-298|1281-1281-1281-1281-1281-1281-1281-1281-1281-1281-1281-1281-1281-1281-1281-1281|407-675-1012-1247-1460-7971-8174-8378-8590-15459-16957-18872-20315-27393-27659-27958|", opener)
        Util.sendOtherCheckGetOpener("http://a.zc.qq.com/Cgi-bin/mouse?|1|1|" + str(mouseTime2) + "|583|169|1|88|298|1281|8560|", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/common/attr?id=252193&r=0.2005294721022658", opener)
        Util.sendGetWithHeaderOpener("http://zc.qq.com/cgi-bin/common/attr?id=256401&r=0.9292085342262181", opener)
    def getNickLongUrl(self):
        #http://a.zc.qq.com/Cgi-bin/MoniKey?56|5|1396438512189&56|5|1396438512496&56|5|1396438512632&56|5|1396438512781&56|5|1396438512902&56|5|1396438513066&56|5|1396438513298&56|5|1396438513554&56|5|1396438513999
        cur = Util.getTime()
        ret = "http://a.zc.qq.com/Cgi-bin/MoniKey?88|1|" + str(cur) +  \
            "&73|1|" + str(cur + 150) +  \
            "&65|1|" + str(cur + 121) +  \
            "&79|1|" + str(cur + 198) +  \
            "&88|1|" + str(cur + 173) +  \
            "&73|1|" + str(cur + 170) +  \
            "&69|1|" + str(cur + 243) +  \
            "&49|1|" + str(cur + 281) +  \
            "&9|1|" + str(cur + 265)
        return ret

    def getPwdLongUrl1(self):
        #http://a.zc.qq.com/Cgi-bin/MoniKey?88|1|1396360234135&73|1|1396360234282&65|1|1396360234389&79|1|1396360234467&88|1|1396360234576&73|1|1396360234681&69|1|1396360234811&49|1|1396360235016&9|1|1396360235299
        cur = Util.getTime()
        ret = "http://a.zc.qq.com/Cgi-bin/MoniKey?56|5|" + str(cur) + \
            "&56|5|" + str(cur + 307) +  \
            "&56|5|" + str(cur + 136) +  \
            "&56|5|" + str(cur + 149) +  \
            "&56|5|" + str(cur + 121) +  \
            "&56|5|" + str(cur + 164) +  \
            "&56|5|" + str(cur + 232) +  \
            "&56|5|" + str(cur + 256) +  \
            "&56|5|" + str(cur + 445)
        return ret

    def getPwdLongUrl2(self):
        #http://a.zc.qq.com/Cgi-bin/MoniKey?88|1|1396360234135&73|1|1396360234282&65|1|1396360234389&79|1|1396360234467&88|1|1396360234576&73|1|1396360234681&69|1|1396360234811&49|1|1396360235016&9|1|1396360235299
        cur = Util.getTime()
        ret = "http://a.zc.qq.com/Cgi-bin/MoniKey?56|6|" + str(cur) +  \
            "&56|6|" + str(cur + 233) +  \
            "&56|6|" + str(cur + 168) +  \
            "&56|6|" + str(cur + 139) +  \
            "&56|6|" + str(cur + 122) +  \
            "&56|6|" + str(cur + 119) +  \
            "&56|6|" + str(cur + 316) +  \
            "&56|6|" + str(cur + 337) +  \
            "&56|6|" + str(cur + 313)
        return ret

    def getLastLongUrl2(self):
        #http://a.zc.qq.com/Cgi-bin/MoniKey?88|1|1396360234135&73|1|1396360234282&65|1|1396360234389&79|1|1396360234467&88|1|1396360234576&73|1|1396360234681&69|1|1396360234811&49|1|1396360235016&9|1|1396360235299
        cur = Util.getTime()
        ret = "http://a.zc.qq.com/Cgi-bin/MoniKey?69|16|" + str(cur) +  \
            "&84|16|" + str(cur + 422) +  \
            "&16|16|" + str(cur + 428) +  \
            "&90|16|" + str(cur + 214) + \
            "&75|16|" + str(cur + 360)
        return ret

    def register(self, code, opener):
        #code = raw_input("Input verification code on " + threadName + ".jpg : ")
        self.postdata["verifycode"] = code
        print(self.postdata)
        response = Util.sendPostWithHeaderOpener("http://zc.qq.com/cgi-bin/chs/numreg/get_acc?r=" + "0.6154321480146582", self.postdata, Util.headers, opener)
        print(response.info())
        ret = response.read()
        print(ret)
        return ret
        
    def prepareRegister(self, threadName, proxy, predict, model, lock):
        #print(threadName + "-" + proxy)
        cj = cookielib.MozillaCookieJar()
        cj.load("cookie.txt")
        opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj));
        #opener = urllib2.build_opener(urllib2.ProxyHandler({"http" : proxy}), urllib2.HTTPCookieProcessor(cj))
        #urllib2.install_opener(opener);
        self.initPage(threadName, opener)
        self.inputValues(opener)
        for index, cookie in enumerate(cj):
            print '[',index, ']',cookie;
        Util.curName = threadName
        lock.acquire()
        code = predict(model, Constants.IMG_DIR_TENCENT_REG + threadName + ".jpg")
        try:
            ret = self.register(code, opener)
            Main.writeToFile(ret + '\n')
            if  ret.find("vc&nbsp") != -1:
                Main.moveFileto(Constants.IMG_DIR_TENCENT_REG + threadName + ".jpg", Constants.IMG_DIR_TENCENT_REG_WRONG + code + ".jpg")
            else:
                Main.moveFileto(Constants.IMG_DIR_TENCENT_REG + threadName + ".jpg", Constants.IMG_DIR_TENCENT_REG_CORRECT + code + ".jpg")
        except :
            print "Unexpected error:", sys.exc_info()
        
        lock.release()


'''
注意每次发送之前更新下POST DATA的最后一个参数
因为实在没看出他是从哪里把这个参数加上的......
'''
if __name__ == "__main__":
    '''Util.cj.load("cookie.txt")
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(Util.cj));
    #opener = urllib2.build_opener(urllib2.ProxyHandler({"http" : '112.241.179.161:29385'}), urllib2.HTTPCookieProcessor(Util.cj))
    urllib2.install_opener(opener);
    initPage("thread-1")
    inputValues()
    
    for index, cookie in enumerate(Util.cj):
        print '[',index, ']',cookie;
    #Util.cj.save("c.txt")
    code = raw_input("Input verification code:")
    postdata["verifycode"] = code
    print(postdata)
    register()
    #print(getLastLongUrl2())'''
    print("hello")
   