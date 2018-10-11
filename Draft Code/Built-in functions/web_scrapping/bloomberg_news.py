#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
BLOOMBERG NEWS SCRAPPING
Started on the 11/10/2017

Theo ALVES DA COSTA
Thami BENGHAZI AKHLAKI
------------------------------------------------------------------------
"""





import bs4 as bs
import nltk
import pandas as pd
import numpy as np
import requests
import time
from tqdm import tqdm
from collections import defaultdict

from ekimetrics.utils.time import sec_to_hms
from ekimetrics.web_scrapping.utils import *
from ekimetrics.web_scrapping.selenium import SeleniumRobot
from ekimetrics.utils import io













#=============================================================================================================================
# BLOOMBERG BASE CLASS
# Made for ODDO project by Theo ALVES DA COSTA
#=============================================================================================================================





class BloombergRobot(object):
    def __init__(self):
        """
        Initialization
        """
        self.data = defaultdict(list)



    #----------------------------------------------------------------------------------------------
    # Scrapping helper functions


    def parse_article_result_on_page(self,article):
        """
        Helper function to parse an article result on a page
        :param bs Object article: the beautiful soup object of an article result
        :returns: dict -- with the parsed data
        """

        # Get main containers
        main_container = article.find("div",class_ = "search-result-story__container")
        meta_data = main_container.find("div",class_ = "search-result-story__metadata")
        headline = main_container.find("h1",class_ = "search-result-story__headline")
        summary = main_container.find("div",class_ = "search-result-story__body").text
        
        # Image
        image_tag = article.find("img",class_ = "search-result-story-thumbnail__image")
        image = article.find("img",class_ = "search-result-story-thumbnail__image").attrs["src"] if image_tag is not None else None
        
        # Meta data
        source_link = meta_data.find("a")
        if source_link is not None:
            source = source_link.text
            source_url = source_link.attrs["href"]
        else:
            source,source_url = None,None
        date = pd.to_datetime(meta_data.find("time").attrs["datetime"])
        
        # Headline
        title = headline.find("a").text
        url = headline.find("a").attrs["href"]
        
        # Store the data in a dictionary
        data_article = {
            "source":source,
            "source_url":source_url,
            "date":date,
            "image":image,
            "title":title,
            "url":url,
            "summary":summary,
        }
        
        return data_article




    def build_query_url(self,query,page_number = 1,start_date = None,end_date = None):
        """
        Build a search query url on a given page
        """

        url = "https://www.bloomberg.com/search?query={}&sort=time:desc".format(query.replace(" ","+"))
        url += "&page={}".format(page_number)

        if start_date is not None:
            start_date = str(pd.to_datetime(start_date))[:10]
            url += "&startTime={}".format(start_date)

        if end_date is not None:
            end_date = str(pd.to_datetime(end_date))[:10]
            url += "&startTime={}".format(end_date)

        return url
        



    def get_search_page_data(self,query,page_number = 1,start_date = None,end_date = None):
        """
        Get and parse a result of a search page
        """
        page_url = self.build_query_url(query,page_number,start_date,end_date)
        page_data = scrapping(page_url)
        articles = page_data.findAll("article")
        if len(articles) > 0:
            articles_data = [self.parse_article_result_on_page(a) for a in articles]
        else:
            articles_data = []

        return articles_data






    #----------------------------------------------------------------------------------------------
    # Scrapping helper functions




    def search(self,query,max_page = 10,inplace = True,start_date = None,end_date = None):
        """
        Search articles given a query
        :param str query: the query
        :param max_page: the maximum number of page scrapped
        """

        # Initialize the page
        page_number = 1
        query_data = []
        exceptions = []
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Loop
        while True:
            try:
                if (max_page is not None and page_number <= max_page) or max_page is None:
                    articles_data = self.get_search_page_data(query,page_number,start_date,end_date)
                    if len(articles_data) > 0:
                        query_data.extend(articles_data)
                        print("\r>> [{}] Page {} scrapped with {} articles".format(query,page_number,len(articles_data)),end = "")
                        page_number += 1
                    else:
                        print(" - DONE")
                        break
                else:
                    print(" - DONE")
                    break

            except Exception as e:
                exceptions.append(e)
                print(e)
                if len(exceptions) > 5:
                    raise ValueError("Too many exceptions during the scrapping, there must be a problem")
            


        if inplace:
            self.data[query] = query_data
        else:
            for article in query_data:
                article["query"] = query
            return query_data


    def search_multiple_queries(self,queries,max_page = None,alert_by_slack = None,start_date = None,end_date = None,method = None):
        if alert_by_slack is not None:
            from ekimetrics.api.slack import Slack
            slack = Slack()

        for query in queries:
            s = time.time()

            if method is None:
                self.search(query,max_page = max_page,start_date = start_date,end_date = end_date)
            elif method["name"] == "mongodb":
                data = self.search(query,max_page = max_page,start_date = start_date,end_date = end_date,inplace = False)
                io.save_data_in_collection(data,**method["parameters"],verbose = 0)
            else:
                raise ValueError("You did not provide a correct method")

            e = time.time()
            duration = sec_to_hms(e-s)
            if alert_by_slack is not None:
                slack.send_message(to = alert_by_slack,message = "BLOOMBERG : query '{}' scrapped in {} - {} articles".format(query,duration,len(data)))



                








#=============================================================================================================================
# BLOOMBERG ARTICLE
# Developped for ODDO BHF project by Thaminator
#=============================================================================================================================


class BloombergArticle(object):
    def __init__(self,url):
        self.url = url
        self.page = self.get_page()
        
    def get_page(self):
        page = scrapping(self.url)
        return(page)


    def get_content(self, print_it = False):
        """
        return the body content of the article
        with print_it = True it prints a readable format of the content
        """
        body = self.page.find("div", attrs={"class": "body-copy"})
        paragraphs = body.findAll("p")
        body_raw = ''
        body_print = ''
        for p in paragraphs:
            body_raw = body_raw + p.text
            body_print = body_print + "\n" + p.text
        if print_it:
            print(body_print)
        else:
            return body_raw


    def get_tags(self):
        """
        return the list of tags within the articles
        """
        body = self.page.find("div", attrs={"class": "body-copy"})
        paragraphs = body.findAll("p")
        tags = []
        for p in paragraphs:
            for a in p.findAll("a"):
                tags.append({"tag":a.text, "ref":a.attrs["href"]})
        return tags










