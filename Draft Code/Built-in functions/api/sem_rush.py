#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
SEM RUSH
Started on the 20/10/2017

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import requests
from tqdm import tqdm




#=============================================================================================================================
# SEM RUSH CONSTANTS
#=============================================================================================================================


# Warning, it will use expensive API units, please contact Theo ALVES DA COSTA or Othmane ZRIKEM before using the API
API_KEY = "28acfe21526195fdd81e1a62ace911f0" 

AVAILABLE_GEOS = {
    "us":"(google.com)",
    "uk":"(google.co.uk)",
    "ca":"(google.ca)",
    "ru":"(google.ru)",
    "de":"(google.de)",
    "fr":"(google.fr)",
    "es":"(google.es)",
    "it":"(google.it)",
    "br":"(google.com.br)",
    "au":"(google.com.au)",
    "bing-us":"(bing.com)",
    "ar":"(google.com.ar)",
    "be":"(google.be)",
    "ch":"(google.ch)",
    "dk":"(google.dk)",
    "fi":"(google.fi)",
    "hk":"(google.com.hk)",
    "ie":"(google.ie)",
    "il":"(google.co.il)",
    "mx":"(google.com.mx)",
    "nl":"(google.nl)",
    "no":"(google.no)",
    "pl":"(google.pl)",
    "se":"(google.se)",
    "sg":"(google.com.sg)",
    "tr":"(google.com.tr)",
    "mobile-us":"(google.com)",
    "jp":"(google.co.jp)",
    "in":"(google.co.in)",
    "hu":"(google.hu)",
    "af":"(google.af)",
    "al":"(google.al)",
    "dz":"(google.dz)",
    "ao":"(google.ao)",
    "am":"(google.am)",
    "at":"(google.at)",
    "az":"(google.az)",
    "bh":"(google.bh)",
    "bd":"(google.bd)",
    "by":"(google.by)",
    "bz":"(google.bz)",
    "bo":"(google.bo)",
    "ba":"(google.ba)",
    "bw":"(google.bw)",
    "bn":"(google.bn)",
    "bg":"(google.bg)",
    "cv":"(google.cv)",
    "kh":"(google.kh)",
    "cm":"(google.cm)",
    "cl":"(google.cl)",
    "co":"(google.co)",
    "cr":"(google.cr)",
    "hr":"(google.hr)",
    "cy":"(google.cy)",
    "cz":"(google.cz)",
    "cd":"(google.cd)",
    "do":"(google.do)",
    "ec":"(google.ec)",
    "eg":"(google.eg)",
    "sv":"(google.sv)",
    "ee":"(google.ee)",
    "et":"(google.et)",
    "ge":"(google.ge)",
    "gh":"(google.gh)",
    "gr":"(google.gr)",
    "gt":"(google.gt)",
    "gy":"(google.gy)",
    "ht":"(google.ht)",
    "hn":"(google.hn)",
    "is":"(google.is)",
    "id":"(google.id)",
    "jm":"(google.jm)",
    "jo":"(google.jo)",
    "kz":"(google.kz)",
    "kw":"(google.kw)",
    "lv":"(google.lv)",
    "lb":"(google.lb)",
    "lt":"(google.lt)",
    "lu":"(google.lu)",
    "mg":"(google.mg)",
    "my":"(google.my)",
    "mt":"(google.mt)",
    "mu":"(google.mu)",
    "md":"(google.md)",
    "mn":"(google.mn)",
    "me":"(google.me)",
    "ma":"(google.ma)",
    "mz":"(google.mz)",
    "na":"(google.na)",
    "np":"(google.np)",
    "nz":"(google.nz)",
    "ni":"(google.ni)",
    "ng":"(google.ng)",
    "om":"(google.om)",
    "py":"(google.py)",
    "pe":"(google.pe)",
    "ph":"(google.ph)",
    "pt":"(google.pt)",
    "ro":"(google.ro)",
    "sa":"(google.sa)",
    "sn":"(google.sn)",
    "rs":"(google.rs)",
    "sk":"(google.sk)",
    "si":"(google.si)",
    "za":"(google.za)",
    "kr":"(google.kr)",
    "lk":"(google.lk)",
    "th":"(google.th)",
    "bs":"(google.bs)",
    "tt":"(google.tt)",
    "tn":"(google.tn)",
    "ua":"(google.ua)",
    "ae":"(google.ae)",
    "uy":"(google.uy)",
    "ve":"(google.ve)",
    "vn":"(google.vn)",
    "zm":"(google.zm)",
    "zw":"(google.zw)",
    "ly":"(google.ly)",
    "mobile-uk":"(google.com)",
    "mobile-ca":"(google.ca)",
    "mobile-de":"(google.de)",
    "mobile-fr":"(google.fr)",
    "mobile-es":"(google.es)",
    "mobile-it":"(google.it)",
    "mobile-br":"(google.com.br)",
    "mobile-au":"(google.com.au)",
    "mobile-dk":"(google.dk)",
    "mobile-mx":"(google.com.mx)",
    "mobile-nl":"(google.nl)",
    "mobile-se":"(google.se)",
    "mobile-tr":"(google.com.tr)",
    "mobile-in":"(google.co.in)",
    "mobile-id":"(google.co.id)",
    "mobile-il":"(google.co.il)"
}
















#=============================================================================================================================
# SEM RUSH WRAPPER
#=============================================================================================================================




class SEMRush(object):
    """
    SEM Rush wrapper class
    """
    def __init__(self,api_key = API_KEY):
        """
        Initialization
        """
        self.endpoint = "http://api.semrush.com/"
        self.api_key = api_key


    #-----------------------------------------------------------------------
    # OPERATORS

    def __repr__(self):
        """
        String representation
        """
        r = """
        SEM Rush Wrapper by Ekimetrics 
        - API units remaining : {}
        """.format(self.get_remaining_API_units())
        return r


    def __str__(self):
        """
        String representation
        """
        return self.__repr__()





    #-----------------------------------------------------------------------
    # GETTERS

    def get_remaining_API_units(self):
        """
        Retrieve the remaining number of API units
        Free call to the API in API units
        """
        url = "http://www.semrush.com/users/countapiunits.html?key={}".format(self.api_key)
        return int(requests.get(url).content)


    def get_available_geos(self):
        """
        Return the available geographies possible to request
        """
        return AVAILABLE_GEOS  


    def get_correct_geo(self,geo):
        """
        Map and correct the geographies
        Raise a Value Error if the geography is not available
        """
        if self.is_geo_available(geo):
            return geo.lower()
        else:
            raise ValueError("The geo is not available, enter one of the geo from self.get_available_geos()")



    #-----------------------------------------------------------------------
    # HELPER FUNCTIONS



    def _format_output(self,output):
        """ 
        The usual output is a string containing tabular data
        Format this string to a Pandas DataFrame
        """
        if "ERROR" not in output:
            output = [y.split(";") for y in output.split("\r\n")]
            output = pd.DataFrame(output[1:],columns = output[0])
        else:
            output = pd.DataFrame(columns = ["Keyword","Search Volume","CPC","Competition","Number of Results"])
        return output



    def build_url(self,type,**kwargs):
        """
        Helper function to build the request url from a request type and any keywords additional arguments
        """
        url = self.endpoint
        url += "?key={}&".format(self.api_key)
        url += "type={}".format(type)

        if len(kwargs) > 0:
            other_args = ["{}={}".format(k,str(v).replace(" ","+")) for k,v in kwargs.items()]
            url += "&" + "&".join(other_args)
        return url



    def query_api(self,query_url):
        """
        Request and format on a query url
        Raise a Value Error if request number is other than 200
        """
        r = requests.get(query_url)
        if r.ok:
            output = r.content.decode("utf8")
            return self._format_output(output)
        else:
            raise ValueError("Problem with the SEM Rush API")

    

    def is_geo_available(self,geo):
        """
        Helper function that decides if the geo is available
        """
        return geo.lower() in AVAILABLE_GEOS




    #-----------------------------------------------------------------------
    # KEYWORD METHODS



    def get_keyword_data(self,keyword,geo = "US",raise_exceptions = True):
        """
        Get keyword or keywords data
        The usual columns will be "keyword","volume","cpc","competition","number of results"
        Cost 10 API units per call
        Documentation available at : https://www.semrush.com/api-analytics/#phrase_all
        """

        # Map to correct geo
        geo = self.get_correct_geo(geo)

        # If given a list of keywords
        if type(keyword) == list:
            columns = ["Keyword","Search Volume","CPC","Competition","Number of Results"]
            data = pd.DataFrame()
            for i,k in tqdm(enumerate(keyword)):

                if raise_exceptions:
                    d = self.get_keyword_data(keyword = k,geo = geo)
                    data = data.append(d,ignore_index = True)
                else:
                    try:
                        d = self.get_keyword_data(keyword = k,geo = geo)
                        data = data.append(d,ignore_index = True)
                    except:
                        print("Problem with keyword {}".format(k))

            return data[columns]

            

        # If given only one keyword
        else:

            # Define the dictionary of arguments for the request
            kwargs = {
                "export_columns":"Ph,Nq,Cp,Co,Nr",
                "phrase":keyword,
                "database":geo,
            }

            # Create the url with the arguments
            url = self.build_url("phrase_this",**kwargs)

            # Retrieve the data
            data = self.query_api(url)

            return data
            



    def get_related_keywords(self,keyword,limit = 100,geo = "US"):
        """
        Documentation : https://www.semrush.com/api-analytics/#phrase_fullsearch
        Cost 40 API units per line of the report
        """
        

        # Define the dictionary of arguments for the request
        kwargs = {
            "export_columns":"Ph,Nq,Cp,Co,Nr,Td",
            "phrase":keyword,
            "database":geo,
            "display_limit":limit,

        }

        # Create the url with the arguments
        url = self.build_url("phrase_fullsearch",**kwargs)

        # Retrieve the data
        data = self.query_api(url)

        return data






    def get_alternate_keywords(self,keyword,limit = 100,geo = "US"):
        """
        Documentation : https://www.semrush.com/api-analytics/#phrase_fullsearch
        Cost 20 API units per line of the report
        """
        

        # Define the dictionary of arguments for the request
        kwargs = {
            "export_columns":"Ph,Nq,Cp,Co,Nr,Td",
            "phrase":keyword,
            "database":geo,
            "display_limit":limit,

        }

        # Create the url with the arguments
        url = self.build_url("phrase_related",**kwargs)

        # Retrieve the data
        data = self.query_api(url)

        return data