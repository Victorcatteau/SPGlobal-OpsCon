#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
PAGES JAUNES AUTOMATION
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

from ekimetrics.web_scrapping.selenium import *


#=============================================================================================================================
# PAGES JAUNES ROBOT
#=============================================================================================================================



class PagesJaunesRobot(SeleniumRobot):
    """
    Pages Jaunes Robot class
    Can be used to retrieve phone numbers
    Developed in the scope of MCS project in 2018

    Warning : Pages Jaunes is very sensitive to scrapping, don't scrape too much 
    """

    def __init__(self,**kwargs):
        """
        Initialization
        """

        self.url = "https://www.pagesjaunes.fr/"
        super().__init__(**kwargs)
        self.connect(self.url)



    def search(self,who,where):
        """Search someone on page jaunes website
        """

        if self.driver.current_url != self.url:
            self.connect(self.url)

        # Type who
        self.type(who,element_id="pj_search_quoiqui")

        # Taking it slowly
        time.sleep(np.random.rand()*2)

        # Type where
        self.type(where,element_id="pj_search_ou")

        # Slower slower
        time.sleep(np.random.rand()*2)

        # Pressing enter
        self.press_enter(element_id="pj_search_ou")

        counter = 0
        while self.driver.current_url == self.url:
            time.sleep(1)
            ambiguite = self.get_html().find(class_ = "ambiguite_ou_container")
            if ambiguite is not None:
                first_link = self.driver.find_element_by_class_name("ambiguite_ou_container").find_element_by_css_selector("a")
                first_link.click()
                time.sleep(1)

            counter += 1

            if counter > 5:
                self.connect(self.url)
                raise Exception("Problem with where")





    def get_number_of_results(self):
        """Returns the number of results in a search page
        """

        # Get html code
        page = self.get_html()

        # Get number 
        number = int(page.find("span",class_="denombrement").text.strip().split(" ")[0])

        return number








#=============================================================================================================================
# PAGES JAUNES ROBOT
#=============================================================================================================================



class PagesBlanchesRobot(SeleniumRobot):
    """
    Pages Jaunes Robot class
    Can be used to retrieve phone numbers
    Developed in the scope of MCS project in 2018

    Warning : Pages Jaunes is very sensitive to scrapping, don't scrape too much 
    """

    def __init__(self,**kwargs):
        """
        Initialization
        """

        self.url = "https://www.pagesjaunes.fr/pagesblanches/"
        super().__init__(**kwargs)
        self.connect(self.url)



    def search(self,who,where):
        """Search someone on page jaunes website
        """

        if self.driver.current_url+"/" != self.url:
            self.connect(self.url)

        # Type who
        self.type(who,element_id="pj_search_qui")

        # Taking it slowly
        time.sleep(np.random.rand()*2)

        # Type where
        self.type(where,element_id="pj_search_ou")

        # Slower slower
        time.sleep(np.random.rand()*2)

        # Pressing enter
        self.press_enter(element_id="pj_search_ou")

        counter = 0
        while self.driver.current_url+"/" == self.url:
            time.sleep(1)
            ambiguite = self.get_html().find(class_ = "ambiguite_ou_container")
            if ambiguite is not None:
                first_link = self.driver.find_element_by_class_name("ambiguite_ou_container").find_element_by_css_selector("a")
                first_link.click()
                time.sleep(1)

            counter += 1

            if counter > 5:
                self.connect(self.url)
                raise Exception("Problem with where")





    def get_number_of_results(self):
        """Returns the number of results in a search page
        """

        # Get html code
        page = self.get_html()

        # If no results
        container = page.find(class_ = "container")
        if container is not None:
            if "Oups" in page.find(class_="container").text.strip():
                number = 0
            else:
                # Get number 
                number = int(page.find("span",class_="denombrement").text.strip().split(" ")[0])
        else:
            try:
                number = int(page.find("span",class_="denombrement").text.strip().split(" ")[0])
            except:
                number = None

        return number






