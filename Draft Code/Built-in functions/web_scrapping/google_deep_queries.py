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
from ekimetrics.web_scrapping.google_trends import *
from ekimetrics.web_scrapping.robots import *
from ekimetrics.api.sem_rush import SEMRush




#=============================================================================================================================
# GOOGLE DEEP QUERIES
# Google volume and keyword extension simulation
# Developped for ODDO BHF project
#=============================================================================================================================




class GoogleDeepQueries(SeleniumRobot):
    """
    Google Deep Queries class representation
    Will use the SearchVolumeIoRobot and the UberSuggestRobot
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)



    #---------------------------------------------------------------------------------
    # GETTERS


    def get_semrush_wrapper(self):
        """
        Get the sem rush wrapper object
        If not defined before define it first as attribute
        """
        if not hasattr(self,"semrush"):
            self.semrush = SEMRush()
        
        return self.semrush




    #---------------------------------------------------------------------------------
    # HELPER FUNCTIONS TO DETECT THE SUITABLE METHOD TO RETRIEVE DATA

    def detect_method(self,method = "auto",geo = "US"):
        """
        Detect the method used at every step at the process
        Depending on the geography, some source are more relevant or less limitating
        """
        if type(method) != list: method = [method]*2

        # Purpose : enrichment
        method[0] = self.detect_auto_method_on_purpose(method = method[0],geo = geo,purpose = "enrichment")

        # Purpose : volume
        method[1] = self.detect_auto_method_on_purpose(method = method[1],geo = geo,purpose = "volume")

        return method



    def detect_auto_method_on_purpose(self,method,geo = "US",purpose = "volume"):
        """
        Helper function that traduces an "auto" method to the most relevant method 
        given the geography and the step in the process
        """

        if method == "auto":
            assert purpose in ["volume","enrichment"]

            if purpose == "volume":
                if geo in ["US","UK"]:
                    return "searchvolumeio"
                else:
                    return "semrush"

            elif purpose == "enrichment":
                return "ubersuggest"
        else:
            return method






    #---------------------------------------------------------------------------------
    # MAIN ANALYSIS


    def start_on_keyword(self,keyword,trends = True,geo = "US",max_keywords = 10,method = "auto"):
        """
        Main Google Deep Queries analysis
        - Enrichment phase
        - Volume phase
        - Trends phase
        """

        method_enrichment,method_volume = self.detect_method(method = method,geo = geo)

        # Get similar keywords
        keywords = self.get_similar_queries(keyword,geo = geo,method = method_enrichment)

        # Get volumes
        data = self.get_keywords_volumes(keywords,geo = geo,method = method_volume).head(max_keywords)

        # Get trends
        if trends:
            data = self.get_keywords_trends(data,geo = geo)

        return data





    def get_similar_queries(self,keyword,geo = "US",method = "auto",limit = 100):
        """
        Enrichment phase 
        Find the most related and similar queries from a given keyword and geography
        """

        method = self.detect_auto_method_on_purpose(method,geo = geo,purpose = "enrichment")
        method = "semrush"
        assert method in ["ubersuggest","semrush"]

        print(">> Getting similar queries with {}".format(method))


        if method == "ubersuggest":

            # Create the ubersuggest robot
            robot = UbersuggestRobot(driver = self.driver,verbose = 0)
            
            # Connect to the url
            robot.connect(robot.url)

            # Get similar keywords
            keywords = robot.get_similar_queries(keyword,geo = geo)

            return keywords

        else:

            semrush = self.get_semrush_wrapper()
            keywords = semrush.get_related_keywords(keyword,geo = geo,limit=limit)
            keywords = keywords[["Keyword","Search Volume"]]
            keywords.columns = ["keyword","volume"]
            keywords.set_index("keyword",inplace = True)
            keywords['volume'] = pd.to_numeric(keywords['volume'])
            return keywords







    def get_keywords_volumes(self,keywords,geo = "US",method = "auto"):
        """
        Volume phase
        Find the volumes associated for a given list of keywords in a given geography
        """

        method = self.detect_auto_method_on_purpose(method,geo = geo,purpose = "volume")
        assert method in ["searchvolumeio","semrush"]

        print(">> Getting keywords volumes with {}".format(method))

        # SEM RUSH METHOD TO GET VOLUMES
        if method == "semrush":

            # Get SEM Rush wrapper
            semrush = self.get_semrush_wrapper()
            
            # Get data via SEM rush
            volume_data = semrush.get_keyword_data(keywords,geo = geo,raise_exceptions = False)

            # Format the same way along the sources
            volume_data = volume_data.iloc[:,:2]
            volume_data.columns = ["keyword","volume"]
            volume_data.set_index("keyword",inplace = True)
            volume_data["volume"] = volume_data["volume"].astype(int)

            return volume_data



        # SEARCH VOLUME IO METHOD TO GET VOLUMES
        elif method == "searchvolumeio":

            # Create the search volume robot
            robot = SearchVolumeIoRobot(driver = self.driver,verbose = 0)

            # Connect to the url
            robot.connect(robot.url)

            # Get the volumes
            volume_data = robot.get_volume_data(keywords,geo = geo)

            return volume_data







    def get_keywords_trends(self,data,geo = "US"):
        """
        Trends phase
        Find the trends associated for a given list of keywords in a given geography
        """

        print(">> Getting keywords trends with Google Trends")

        # Create the Google Trends robot
        robot = GoogleTrendsRobot(method = "selenium",driver = self.driver)

        # Get the keywords
        keywords = list(data.index)

        # Get the trends
        robot.get_trends(keywords,geo = geo)

        # Merge with volume data
        robot.get_volume_data(volume_data = data,geo = geo)

        self.test = robot

        # Format the data
        for i,keyword in enumerate(robot.data):
            data = robot.data[keyword][geo].rename(columns = {"volume":keyword})
            if i == 0:
                all_data = data[["dates",keyword]]
            else:
                all_data = pd.concat([all_data,data[[keyword]]],axis = 1)
                
        all_data = all_data.set_index("dates").transpose()

        self.cache = all_data

        return all_data


