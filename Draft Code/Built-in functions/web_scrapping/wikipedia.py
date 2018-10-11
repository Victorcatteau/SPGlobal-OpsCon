#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
WIKIPEDIA

Started on the 10/08/2017
Thami BENGHAZI
------------------------------------------------------------------------
"""


import pandas as pd
from tqdm import tqdm

from ekimetrics.web_scrapping.utils import *
import wikipedia as wiki




#=============================================================================================================================
# SCRAP TABLE
#=============================================================================================================================


url = "https://en.wikipedia.org/wiki/List_of_largest_European_companies_by_revenue"

def scrap_wikitable (url):
    page = scrapping(url)
    table = page.find(tab for tab in page.findAll("table") if tab["class"]=="wikitable")
    rows = table.findAll("tr")
    wikitable = {}
    for i in range(len(rows)-2):
        wikitable[i] = [r.text for r in (rows[i+1].findAll("th") or rows[i+1].findAll("td"))]
        if i>0:
            links = [0,0]
            links[0]= "https://en.wikipedia.org" + rows[i+1].findAll("td")[1].find("a")["href"]
            links[1] = "https://en.wikipedia.org" + rows[i+1].findAll("td")[4].find("a")["href"]
            wikitable[i] = wikitable[i] + links                 
    wikitable[0] = wikitable[0] + ["Wiki-Company link","this column will be dropped"]
    wikitable = pd.DataFrame(wikitable)
    wikitable = wikitable.transpose()
    wikitable.columns = wikitable.iloc[0]
    wikitable.drop(0,axis=0, inplace=True)
    return wikitable[wikitable.columns[:-1]]










#=============================================================================================================================
# COMPANY CLASS
#=============================================================================================================================

class WikipediaCompany(object):
    def __init__(self, url):
            self.url = object
            self.page = scrapping(url)

    def get_isin(self): 
        v = self.page.find("table")
        list = [tr for tr in v.findAll("tr")]
        found = 0
        for r in list:
            try:
                if r.find("th").text == "ISIN":
                    found = 1
                    return r.find("td").text
            except BaseException as e:
                    pass
        if found == 0:
            return "Can't find an ISIN on this page"


    def get_industry(self):
        v = self.page.find("table")
        list = [tr for tr in v.findAll("tr")]
        for r in list:
            try:
                if r.find("th").text == "Industry": 
                    return r.find("td").text
            except BaseException as e:
                    pass



    def get_products(self):
        v = self.page.find("table")
        list = [tr for tr in v.findAll("tr")]
        for r in list:
            try:
                if r.find("th").text == "Products":
                    return r.find("td").text                
            except BaseException as e:
                    pass



    def get_description(self):
        p = wiki.page(self.get_name())
        description = p.summary
        return description


    def get_name(self):
        try:
            return self.page.find("h1").text
        except BaseException as e:
            pass


    def get_info(self):
        info = {}
        info["ISIN"] = self.get_isin()
        info["Industry"] = self.get_industry()
        info["Products"] = self.get_products()
        info["Description"] = self.get_description()
        info["Name"] = self.get_name()
        return info









#=============================================================================================================================
# SCRAP MULTIPLE COMPANIES
#=============================================================================================================================



def scrap_multiple_companies(wikipedia_link_list):
    print(">> Scrapping {} companies wikipedia data".format(len(wikipedia_link_list)))
    data = []
    for url in tqdm(wikipedia_link_list):
        company = Company(url)
        data.append(company.get_info())
    return data