# -*- coding: UTF-8 -*-
'''
Created on 2014/3/26

@author: rogers
'''
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

browser = webdriver.Firefox()

browser.get('https://mail.qq.com')

elem = browser.find_element_by_id('switch')  # Find the search box
elem.click()

#browser.quit()
