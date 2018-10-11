#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
SELENIUM AUTOMATION
Started on the 10/09/2017
theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import bs4 as bs
import numpy as np
import pandas as pd
import requests
import time
import random
from tqdm import tqdm
import webbrowser

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from ekimetrics.web_scrapping import utils


#=============================================================================================================================
# SELENIUM ROBOT
#=============================================================================================================================



class SeleniumRobot(object):
    """
    Selenium Robot base class
    Can be used for direct use or as a base class for more complex scrappers
    """

    def __init__(self,driver_path = "C:/git/chromedriver.exe",driver_type = "chrome",driver = None,verbose = 1,proxy = None):
        """
        Initialization
        """
        self.verbose = verbose
        self.driver_type = driver_type
        assert driver_type in ["chrome","phantomjs"]

        # Connecting to the driver
        self._print(">> Connecting {} Driver".format(driver_type))
        self.driver_path = driver_path
        self.connect_driver(self.driver_path,driver,proxy)



    def _print(self,message,**kwargs):
        """
        Print only if verbose attribute is true
        """
        if self.verbose: print(message,**kwargs)



    def connect_driver(self,driver_path,driver = None,proxy = None):
        """
        Connect to Selenium Driver
        """

        # Proxy arguments
        with_proxy = False
        if proxy is not None:
            with_proxy = True
            if proxy == "auto":
                proxies = utils.find_proxy(n = 3)
                proxies = ["{}:{}".format(p.host,p.port) for p in proxies]
                proxy = proxies[0]
                print("... Connecting to proxy {}".format(proxy))

        # Selecting the driver
        if driver is None:

            # With Chrome Driver
            if self.driver_type == "chrome":

                # Without external proxy
                if not with_proxy:
                    self.driver = webdriver.Chrome(driver_path)

                # With external proxy
                else:
                    chrome_options = webdriver.ChromeOptions()
                    chrome_options.add_argument('--proxy-server={}'.format(proxy))
                    self.driver = webdriver.Chrome(driver_path,chrome_options = chrome_options)

            # With PhantomJS
            elif self.driver_type == "phantomjs":

                # Without external proxy
                if not with_proxy:
                    self.driver = webdriver.PhantomJS(driver_path)

                # With external proxy
                else:
                    service_args = [
                        '--proxy={}'.format(proxy),
                        '--proxy-type=https',
                        ]

                    self.driver = webdriver.PhantomJS(driver_path,service_args = service_args)


        
        else:
            self.driver = driver





    #------------------------------------------------------------------------------------------------------
    # PROXY CONTROL



    def open_gatherproxy(self):
        """
        Open the websites that holds many proxies for safety checks
        """
        webbrowser.open("http://www.gatherproxy.com")



    def find_ip(self,source = "org"):
        """
        Get the current IP address
        """

        print("... Finding current ip")

        # More reliable source, but slow
        if source == "https://www.whatismyip.com/" or source == "com":
            source = "https://www.whatismyip.com/"
            self.driver.get(source)
            ip = self.driver.find_element_by_class_name("ip").text.split("\n")[1]

        # Faster, but less reliable source
        elif source == "http://whatismyip.org/" or source == "org":
            source = "http://whatismyip.org/"
            self.driver.get(source)
            ip = self.driver.find_element_by_tag_name("span").text
        
        # Raise exception if not in the list of source urls
        else:
            raise Exception("Source url for ip is not recognized, try the default https://www.whatismyip.com/")


        self.current_ip = ip
        return ip




    def find_location(self):
        """
        Get the current location of the IP address
        Can be used for checks using proxies while scrapping
        """

        if not hasattr(self,"current_ip"):
            self.find_ip()

        self.connect("https://db-ip.com/{}".format(self.current_ip))
        page = self.get_html()
        
        return {x.find("th").text.strip():x.find("td").text.strip() for table in page.findAll("tbody") for x in table.findAll("tr")}







    #------------------------------------------------------------------------------------------------------
    # SELENIUM CONTROL


    def connect(self,url):
        """
        Connect the driver to a given url
        """
        self.driver.get(url)



    def wait(self,seconds,verbose = 1):
        """
        Wait for a given number of seconds
        """
        seconds = int(seconds)
        for i in range(seconds):
            if verbose: print(".",end = "")
            time.sleep(1)
        if verbose: print('\r',end = "")



    def scroll_down(self):
        """
        Scroll down all the page
        """
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")



    def press_enter(self,element_id = None,input_element = None):
        """
        Press enter on an element of the page
        """
        input_element = self.get_element(element_id,input_element)
        input_element.send_keys(Keys.ENTER)
        return input_element




    def submit(self,element_id = None,input_element = None):
        """
        Submit an element
        """
        input_element = self.get_element(element_id,input_element)
        input_element.submit()
        return input_element



    def type(self,message,element_id = None,input_element = None):
        """
        Type something in an element
        """
        input_element = self.get_element(element_id,input_element)
        input_element.send_keys(message)
        return input_element



    def get_element(self,element_id = None,input_element = None):
        """
        Get an element on a page
        """
        if input_element is None:
            input_element = self.driver.find_element_by_id(element_id)
        return input_element



    def get_html(self):
        """
        Get the html page of the code
        """
        html = self.driver.page_source
        page = bs.BeautifulSoup(html,'html5lib')
        return page



    def select_option(self,option,element_id = None,input_element = None):
        """
        Select an option
        """
        if input_element is None:
            input_element = self.driver.find_element_by_id(element_id)

        if option != input_element.get_attribute("value"):
            input_element.send_keys(option)



    def clear_textarea(self,element_id = None,input_element = None):
        """
        Clear Text Area
        """
        if input_element is None:
            input_element = self.driver.find_element_by_id(element_id)

        input_element.clear()



    def close(self):
        """
        Close the driver
        """
        self.driver.close()








