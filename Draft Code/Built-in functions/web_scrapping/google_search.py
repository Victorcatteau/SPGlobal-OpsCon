#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
GOOGLE TRENDS
Started on the 25/04/2017

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


# Usual libraries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import json
from tqdm import tqdm
import bs4 as bs
from collections import defaultdict

# Custom libraries
from ekimetrics.web_scrapping.selenium import SeleniumRobot
from ekimetrics.utils import io
from ekimetrics.visualizations import maps
from ekimetrics.visualizations import charts





#=============================================================================================================================
# GOOGLE QUERIES SELENIUM SCRAPPER
# Temporary
# Made for Google Dubai project by Theo ALVES DA COSTA
#=============================================================================================================================




class GoogleSearchRobot(SeleniumRobot):
    def __init__(self,**kwargs):
        self.url = "https://www.google.com"
        super().__init__(**kwargs)



    #--------------------------------------------------------------
    # CONNECTION TO GOOGLE

    def connect_to_google(self):
        url = "https://www.google.com/"
        self.connect(url)

    def connect_to_google_lebanese(self):
        url = "https://www.google.com.lb/"
        self.connect(url)



    #--------------------------------------------------------------
    # CHANGE LANGUAGE
    def change_language_to_arabic(self):
        languages = self.driver.find_element_by_id("_eEe")
        languages.find_element_by_link_text("العربية").click()



    #--------------------------------------------------------------
    # RETRIEVE DATA

    def get_search_results_data(self,query):
        self.clear_textarea(element_id="lst-ib")
        x = self.type(query,element_id="lst-ib")
        self.press_enter(input_element=x)
        page = self.get_html()
        return page


    #--------------------------------------------------------------
    # PREPROCESS DATA


    def _parse_result(self,result):
        pass

    def _parse_page(self,page):
        pass





