#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
NYT API SEARCH

Started on the 04/10/2017
Thami BENGHAZI AKHLAKI
------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import requests
import json
import time



eki_key_0 = 'd9e86634786144babab86d459a2a1f79' # ekirobot1@mail.com  mdp=55pierrecharron

eki_key_1 = '334cb2f8e218403fb615f1f0e67e16cd'

eki_key_2 = 'cc2b0ea851b043db93878d3472cc063c'

eki_key_3 = '82c7d358b8724cb39dcc095169098f76'

eki_key_4 = '6d92895f2f7547f6bf6450d210633105'

eki_key_5 = '15f404dc64d94f97a1d5e3a04166a6d0'

eki_key_6 = '0733f2cc6d2743a3a4417bd6c1c15d2f' # ekirobot2@gmx.com  mdp=55pierrecharron

eki_key_7 = 'fb2464362928426aaec868caa743685c' # ekirobot3@laposte.net mdp=55Pierrecharron

#=============================================================================================================================
# NEW YORK TIMES API: SEARCH API
#=============================================================================================================================

class NewYorkTimes(object):
    
    def __init__(self, api_key = eki_key_0):
        self.api_key = api_key
        self.remaining_requests = 1000

    def search_concept(self, concept, filter_concept = None, filter_type = None, begin_date = None, end_date = None):
        """
        concept is the specific concept you're looking for in NYT
        One can further filter the search by adding a filter_concept
        Moreover the filter type can be specified in filter_by, this argument can take many values:
        Example: 
        -organizations
        -persons
        -source
        -subject
        more on: https://github.com/NYTimes/public_api_specs/blob/master/article_search/article_search_v2.md

        begin_date format: YYYYMMDD, if not specified the count will start from 1851

        """
        url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?q="{}"'.format(concept)
        
        if filter_type:
            if filter_concept:
                url = url + '&fq={0}:("{1}")'.format(filter_type, filter_concept)
            else:
                print("You need to specify a filter_concept")
        else:
            if filter_concept:
                url = url + '&fq="{}"'.format(filter_concept)
        if begin_date:
            url = url + '&begin_date={}'.format(begin_date)

        if end_date:
            url = url + '&end_date={}'.format(end_date)

        url = url + '&api_key={}'.format(self.api_key)

        page = requests.get(url)
        self.remaining_requests = int(page.headers["X-RateLimit-Remaining-day"])
        nyt_page = page.json()

        return nyt_page



    def get_number_of_mentions(self, concept, filter_concept = None, filter_type = None, begin_date = None, end_date = None):
        """
        Returns the number of articles the concept you're looking for is mentionned in the NYT
        by specifying the filter_concept you get the number of articles where both concept are mentionned
        """
        
        page = self.search_concept(concept = concept, filter_concept = filter_concept, filter_type = filter_type, begin_date = begin_date, end_date = end_date)
        
        return page["response"]["meta"]["hits"]


    def get_headlines(self, concept, filter_concept = None, filter_type = None, begin_date = None, end_date = None):
        """
        Returns 10 headlines among the ones mentionning the concept
        If more headlines are needed one should change the search_concept function by adding "&page=" + a number to the url
        """
        page = self.search_concept(concept = concept, filter_concept = filter_concept, filter_type = filter_type, begin_date = begin_date, end_date = end_date)

        try:
            docs = page["response"]["docs"]
            good = 1
        except BaseException as e:
            good = 0
        
        if good:
            headlines = [docs[i]["headline"]["main"] for i in range(len(docs))]
        else:
            headlines = "No Headlines available"    
        
        return headlines


    def get_article_urls(self, concept, filter_concept = None, filter_type = None, begin_date = None, end_date = None):
        """
        Returns 10 urls of articles among the ones mentionning the concept
        """
        page = self.search_concept(concept = concept, filter_concept = filter_concept, filter_type = filter_type, begin_date = begin_date, end_date = end_date)

        docs = page["response"]["docs"]

        articles = [docs[i]["web_url"] for i in range(len(docs))]
        
        return articles


    def get_snippets(self, concept, filter_concept = None, filter_type = None, begin_date = None, end_date = None):
        """
        Returns 10 snippets among the ones mentionning the concept
        """
        page = self.search_concept(concept = concept, filter_concept = filter_concept, filter_type = filter_type, begin_date = begin_date, end_date = end_date)

        docs = page["response"]["docs"]

        snippets = [docs[i]["snippet"] for i in range(len(docs))]
        
        return snippets


#=============================================================================================================================
# NYTimes score  
# Developped for ODDO BHF project
#=============================================================================================================================
    

    def get_company_score_in_theme(self, company, theme, percentage = False, begin_date = "20150101", end_date = None):
        """
        Returns the number of mentions since 01 Jan 2015
        """
        if percentage:
            mentions_company_and_theme = self.get_number_of_mentions(concept = theme, filter_concept = company,
                                                                        begin_date = begin_date, end_date = end_date )
            time.sleep(0.5)
            mentions_theme_only = self.get_number_of_mentions(concept = theme, begin_date = begin_date, end_date = end_date )

            score = mentions_company_and_theme / mentions_theme_only
        else:
            time.sleep(1.5)
            mentions_company_and_theme = self.get_number_of_mentions(concept = theme, filter_concept = company,
                                                                        begin_date = begin_date, end_date = end_date )
            score = mentions_company_and_theme


        return score



    def get_company_trend_within_theme(self, company, theme):
        """
        Returns the trend on the last 3 years (2016-2014) if no data returns 2-year-trend else returns 3 [arbitrary]
        trend = yearly evolution of the number of articles where both the theme and company are mentionned
        Can be improved by including date arguments within the call of the function
        """

        mentions_company_and_theme_1 = self.get_number_of_mentions(concept = theme, filter_concept = company, begin_date = "20140101", end_date = "20150101")
        time.sleep(0.1)
        mentions_company_and_theme_2 = self.get_number_of_mentions(concept = theme, filter_concept = company, begin_date = "20150101", end_date = "20160101" )
        time.sleep(0.1)
        mentions_company_and_theme_3 = self.get_number_of_mentions(concept = theme, filter_concept = company, begin_date = "20160101", end_date = "20170101")


        if mentions_company_and_theme_1!=0:
            trend = (mentions_company_and_theme_3 - mentions_company_and_theme_1) / mentions_company_and_theme_1
        else:
            if mentions_company_and_theme_2 == 0:
                if mentions_company_and_theme_3 > 0:
                    trend = 3
                else:
                    trend = 0
            else:
                trend = (mentions_company_and_theme_3 - mentions_company_and_theme_2) / mentions_company_and_theme_2

        return trend



    def get_theme_trend(self, theme):
        """
        Please check the documentation of the function "get_company_trend_within_theme" as the same applies to this function
        """

        mentions_theme_1 = self.get_number_of_mentions(concept = theme, begin_date = "20140101", end_date = "20150101" )
        time.sleep(0.25)
        mentions_theme_2 = self.get_number_of_mentions(concept = theme, begin_date = "20150101", end_date = "20160101" )
        time.sleep(0.25)
        mentions_theme_3 = self.get_number_of_mentions(concept = theme, begin_date = "20160101", end_date = "20170101" )

        if mentions_theme_1!=0:
            trend = (mentions_theme_3 - mentions_theme_1) / mentions_theme_1
        else:
            if mentions_theme_2 == 0:
                if mentions_theme_3 > 0:
                    trend = 3
                else:
                    trend = 0
            else:
                trend = (mentions_theme_3 - mentions_theme_2) / mentions_theme_2

        return trend


