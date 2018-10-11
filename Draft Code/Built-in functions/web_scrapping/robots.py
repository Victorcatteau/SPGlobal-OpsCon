#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
SCRAPPING ROBOTS
Started on the 06/06/2017
theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""





import bs4 as bs
import pandas as pd
import numpy as np
import requests
import time
from tqdm import tqdm

from ekimetrics.web_scrapping.utils import *
from ekimetrics.web_scrapping.websites import *
from ekimetrics.web_scrapping.selenium import SeleniumRobot

from selenium import webdriver
from selenium.webdriver.common.keys import Keys










#=============================================================================================================================
# COFACE FRAUD ROBOT CLASS
# Made for Coface project by Theo ALVES DA COSTA
#=============================================================================================================================


class CofaceError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)



class CofaceRobot(object):
    def __init__(self,company_name,siret = None,compute = True,via_google = False):

        # INITIALIZATION
        self.company_name = company_name
        self.company_siret = str(siret) if siret is not None else None
        self.meta_data = {}
        self.meta_data["warnings"] = {}

        # AUTOMATION
        if compute:

            if via_google:
                # Google Search
                self.search_on_google()
                self.find_societecom_in_results()

            else:
                # Concatenate and try direct lookup on Societe.com
                self.societecom_url = "x-" + self.company_siret + ".html"


            # Societe.com extraction
            self.extract_data_societecom()
            self.fraud_detection_societecom()



    #-----------------------------------------------------------------------
    # GOOGLE SEARCH

    def search_on_google(self):
        if self.company_siret is None:
            company_name = self.company_name + " societe"
        else:
            company_name = self.company_name + " " + self.company_siret

        self.google_search = GoogleSearch(company_name,domain = "fr",compute = True,verbose = 0)




    def find_societecom_in_results(self):
        if len(self.google_search.results) > 0:

            societecom_results = [x["url"] for x in self.google_search.results if "http://www.societe.com/societe/" in x["url"]]

            if len(societecom_results) > 0:
                if len(societecom_results) > 1:
                    self.meta_data["warnings"]["results societe.com"] = "Warning there were multiple companies registered at societe.com under that name"

                self.societecom_url = societecom_results[0]

            else:
                self.meta_data["warnings"]["results societe.com"] = "Warning there were no companies registered at societe.com under that name"
                self.societecom_url = None

        else:
            self.meta_data["warnings"]["results google search"] = "There was a problem when scrapping Google"
            self.societecom_url = None





    #-----------------------------------------------------------------------
    # SOCIETE.COM


    def extract_data_societecom(self):
        self.meta_data["url"] = self.societecom_url
        self.societecom = SocieteCom(self.societecom_url,compute = True,verbose = 0)

        if "Erreur 404" in self.societecom.page.get_text() or "Aucun résultat" in self.societecom.page.get_text():
            raise CofaceError("The page {} was not found on societe.com".format(self.societecom_url))





    def fraud_detection_societecom(self):


        if self.societecom is not None:


            #----------------------------------------------
            # OBSERVATIONS

            # LIQUIDATED
            self.meta_data["is_liquidated"] = self.societecom.is_liquidated()

            # FAMILY MANAGED
            self.meta_data["is_family_managed"] = self.societecom.is_family_managed()

            # FEW EMPLOYEES
            self.meta_data["has_few_employees"] = self.societecom.has_few_employees()

            # RISK
            self.meta_data["is_deregistered"] = self.societecom.is_deregistered()

            

            #----------------------------------------------
            # FEATURES

            # MANAGEMENT
            self.meta_data["management"] = self.societecom.get_management()


            # ADDRESS
            self.meta_data["address"] = self.societecom.get_address()














#=============================================================================================================================
# UBERSUGGEST
# Google similar queries simulation
# Developped for ODDO BHF project by Theo ALVES DA COSTA
#=============================================================================================================================


MAPPING_GEO = {
    "US":"English / United States",
    "FR":"French / France",
}



class UbersuggestRobot(SeleniumRobot):
    def __init__(self,**kwargs):
        self.url = "https://ubersuggest.io/"
        super().__init__(**kwargs)


    def map_correct_geo(self,geo = "US"):
        return MAPPING_GEO[geo]


    def get_similar_queries(self,query,geo = "US"):
        """
        Given a query, find the similar queries on Google starting by the input query
        :param str query: the input query to look up (or a list of strings)
        :returns: list of strings -- the similar queries (or a dictionary of list of strings)
        """

        if type(query) != list:

            # Connect (or reset)
            self.connect(self.url)

            # Type query as input
            input_query = self.type(query,element_id = "keywordBox")

            # Select correct geo
            geo = self.map_correct_geo(geo)
            self.select_option(geo,element_id="s2id_autogen2")
            self.press_enter(element_id = "s2id_autogen2_search")

            # Press enter
            input_query = self.press_enter(input_element = input_query)

            # Wait for the loading is finished
            queries = []
            tries = 0
            while len(queries) == 0 and tries < 5:
                self.wait(5)

                # Get the html
                page = self.get_html()

                # Parse the results
                queries = [x.text.strip() for x in page.findAll(class_ = "btn-link")]
                tries +=1
        else:
            queries = {}
            for q in query:
                queries[q] = self.get_similar_queries(q)

        return queries







class UbersuggestQuery(object):
    def __init__(self,query,similar):
        self.query = query
        self.similar = similar
        self.preprocess()

    def __repr__(self):
        return "{} : {} similar queries".format(self.parent,len(self.similar))

    def _lower(self):
        self.similar = [query.lower() for query in self.similar]

    def _clean(self):
        for i,query in enumerate(self.similar):
            for subparent in self.query.split():

                query = query.replace(subparent.lower(),"").strip()
            
            query = query.split()


            self.similar[i] = query




    def preprocess(self):
        self._lower()
        self._clean()


    def get_text(self):
        return " ".join([" ".join(query) for query in self.similar]).strip()






class UbersuggestQueries(object):
    def __init__(self,data):
        self.queries = []
        for key in data:
            self.queries.append(UbersuggestQuery(query = key,similar = data[key]))

    def build_X(self,method = "bow",**kwargs):

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        assert method in ["bow","tfidf"]
        texts = [x.get_text() for x in self.queries]

        vectorizer = CountVectorizer(**kwargs) if method == "bow" else TfidfVectorizer(**kwargs)

        X = vectorizer.fit_transform(texts).todense()
        X = pd.DataFrame(X,columns = [x[0] for x in sorted(vectorizer.vocabulary_.items(),key = lambda x : x[1])])

        return X





#=============================================================================================================================
# SEARCH VOLUME.IO
# Google search volume simulation
# Developped for ODDO BHF project by Theo ALVES DA COSTA
#=============================================================================================================================


class SearchVolumeIoRobot(SeleniumRobot):
    def __init__(self,**kwargs):
        self.url = "https://searchvolume.io/"
        super().__init__(**kwargs)

    def get_volume_data(self,keywords,geo = "US"):

        # Protection for keywords length
        if len(keywords) > 800:
            raise ValueError("Too many keywords asked, reduce to maximum 800")

        # Transform input
        keywords = "\n".join(keywords)

        # Type them in the input text area
        self.clear_textarea(element_id = "input")
        self.type(keywords,element_id = "input")
        self.wait(1)

        # Select the country
        country = self.map_geo_to_country(geo)
        self.select_option(country,element_id = "country")
        self.wait(1)

        # Press enter
        self.press_enter(element_id = "submit")

        # Wait for the results to appear
        self.wait(3)

        # Get the results
        page = self.get_html()
        data = [(x.findAll("td")[0].text,int(x.findAll("td")[1].text.replace(",",""))) for x in page.find("table").findAll("tr")[1:]]
        return pd.DataFrame(data,columns = ["keyword","volume"]).sort_values("volume",ascending = False).set_index("keyword")



    def map_geo_to_country(self,geo = "US"):
        geo = geo.lower()
        if geo in ["us","usa","united states"]:
            country = "usa"
        elif geo in ["uk","united kingdom"]:
            country = "uk"
        elif geo in ["denmark"]:
            country = "denmark"
        else:
            country = "usa"
        return country







#=============================================================================================================================
# SENSE2VEC ROBOT
# Topics enrichment
# Developped for Google Deep Queries R&D project by Theo ALVES DA COSTA
#=============================================================================================================================




class Sense2VecRobot(SeleniumRobot):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def build_url(self,topic):
        url = "https://demos.explosion.ai/sense2vec/?word={}&sense=auto".format(topic.replace(" ","%20"))
        return url


    def enrich_topic(self,topic):
        url = self.build_url(topic)
        self.connect(url)
        page = self.get_html()
        topics = [x.text for x in page.findAll("span",class_ = "sense2vec-word")]
        return topics   

