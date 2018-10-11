#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
RENTAL WEBSITES
Started on the 20/10/2017

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
import urllib.parse as urlparse

# Custom libraries
from ekimetrics.web_scrapping.selenium import SeleniumRobot
from ekimetrics.web_scrapping.utils import scrapping
from ekimetrics.utils import io





#=============================================================================================================================
# HOSEASONS.CO.UK
# Made for a commercial proposal for a UK private equity firm (Ekimetrisc UK) by Theo ALVES DA COSTA
#=============================================================================================================================




class HoSeasonsRobot(SeleniumRobot):
    def __init__(self,**kwargs):
        self.url = "http://www.hoseasons.co.uk"
        super().__init__(**kwargs)



    def build_url(self,rental_type = "lodges",page = 1):
        url = self.url
        url += "/" + rental_type
        url += "/" + "all-regions"
        url += "/?page={}".format(page)
        return url



    def get_search_results(self,rental_type = "lodges",max_page = None):
        
        # Connect to the first page
        url = self.build_url(rental_type = rental_type,page = 1)
        self.connect(url)

        # Get data
        title,all_results,pagination = self.get_page_data()

        # Get last page
        if max_page is None:
            max_page = self.parse_max_page(pagination)

        if max_page > 1:            
            # Get every page
            for i in tqdm(range(2,int(max_page) + 1),desc = "Scrapping each page"):
                url_page = self.build_url(rental_type = rental_type,page = i)
                self.connect(url_page)
                _,results,_ = self.get_page_data()
                all_results.extend(results)


        # In case the parsing don't work everything is stored in the cache
        self.cache = all_results

        data = []
        # Parse every page
        for result in tqdm(all_results,desc = "Parsing each result"):
            data.append(self.parse_result_on_page(result))

        return data







    def get_page_data(self):
        page = self.get_html()
        title,results,pagination = self.parse_page_data(page)
        return title,results,pagination



    def parse_page_data(self,page):
        
        # Title 
        title = page.find("div",class_ = "heronew--inner").find("h1").text

        # Results
        results = page.find("div",class_ = "listings").findAll("li",class_ = "listing")

        # Next page
        pagination = page.find("div",class_ = "pagination-control")
        pagination = pagination.findAll("li")

        return title,results,pagination
        


    def parse_result_on_page(self,result):
        
        # Initialize storage
        data = {}
        
        # Get subresults
        images = result.find("div",class_ = "listing__image")
        info = result.find("div",class_ = "listing__info")
        features = result.find("div",class_ = "listing__features")
        
        # Parse the images
        carousel = images.find("span",class_ = "json")
        if carousel is None:
            carousel = []
        else:
            carousel = json.loads(carousel.text)

        rating = images.find("div",class_ = "reevoo-box")
        score = rating.find("span",class_ = "RevooScore").text
        score_out_of = rating.find("span",class_ = "RevooOutOf").text
        number_reviews = rating.find("span",class_ = "RevooReview").text.replace(" reviews","")
        data["images"] = carousel
        data["score"] = score
        data["score_out_of"] = score_out_of
        data["number_reviews"] = number_reviews
        
        # Parse the info
        name = info.find("a")
        url = name.attrs["href"]
        name = name.text.strip()
        geo = info.find('h4').text.strip().split("\n")
        geo = " ".join([x.strip() for x in geo])
        data["geo"] = geo
        data["url"] = url
        data["name"] = name
        
        # Parse the features
        essentials = features.find("ul").findAll("li")
        essentials = [x.text.strip() for x in essentials]
        perks = features.find("ul",class_ = "accommodation-listing-perks").findAll("li")
        perks = [x.text.strip() for x in perks if x.text.strip() not in essentials]
        from_price = features.find("div",class_ = "total-price").text.strip()
        data["from_price"] = from_price
        data["perks"] = perks
        data["essentials"] = essentials
        
        return data


    def parse_max_page(self,pagination):
        max_page = pagination[-1].find("a").attrs["href"]
        max_page = urlparse.parse_qs(urlparse.urlparse(max_page).query)["page"][0]
        return max_page











#=============================================================================================================================
# COTTAGES.COM
# Made for a commercial proposal for a UK private equity firm (Ekimetrisc UK) by Theo ALVES DA COSTA
#=============================================================================================================================



class CottagesComRobot(SeleniumRobot):
    def __init__(self,with_selenium = True,**kwargs):
        self.url = "http://www.cottages.com"
        self.with_selenium = with_selenium

        if with_selenium:
            super().__init__(**kwargs)


    def build_url(self,page = 1):
        url = self.url
        url += "/" + "all-regions"
        url += "/?page={}".format(page)
        return url



    def get_search_results(self):

        # Connect to the first page
        url = self.build_url(page = 1)

        # Get data
        title,all_results,pagination = self.get_page_data(url = url)

        # Get last page
        max_page = self.parse_max_page(pagination)

        # Get every page
        for i in tqdm(range(2,3+ 1),desc = "Scrapping each page"):
            url_page = self.build_url(page = i)
            _,results,_ = self.get_page_data(url = url_page)
            all_results.extend(results)

        self.cache = all_results

        data = []
        # Parse every page
        for result in tqdm(all_results,desc = "Parsing each result"):
            data.append(self.parse_result_on_page(result))

        return data



    def get_page_data(self,url = None):
        if self.with_selenium:
            self.connect(url)
            page = self.get_html()
        else:
            page = scrapping(url)

        title,results,pagination = self.parse_page_data(page)
        return title,results,pagination



    def parse_page_data(self,page):
        
        # Title
        title = None

        # Results
        results = page.findAll("div",class_ = "prop-container")

        # Pagination
        pagination = page.find("ul",class_ = "PaginationControllerList")

        return title,results,pagination
        


    def parse_result_on_page(self,result):
        return result


    def parse_max_page(self,pagination):
        max_page = pagination.find("li",class_ = "padr").text.strip()
        return max_page



