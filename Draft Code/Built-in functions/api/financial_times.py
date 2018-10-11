#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
FINANCIAL TIMES API
Started on the 09/09/2017
https://developer.ft.com/docs/api_v1_reference/search/tutorial
https://developer.ft.com/docs/api_v1_reference/search/discovery/facets

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import json



API_KEY = "ndjrtcfysxsxg5zhn8yfdb9r"


def query(query_string = None,json_data = None,api_key = API_KEY,max_elements = 50):
    """
    Query the Financial Times.com API
    Requires an API Key, please register as a developer and use your own API key
    :param str query_string: the query to request
    :param dictionary json_data: the full query to do a more complicated query (override the query_string)
    :param str api_key: the api key required to do the queries
    :returns: a requests object -- for the query
    """

    if query_string is None:
        data = json_data
    else:
        data = {
            "queryString": query_string,
            "resultContext" : {
                 "aspects" :[  "title","lifecycle","location","summary","editorial" ],
                 "facets" : {"names":[ "organisations","people","regions","subjects","topics"],"maxElements":max_elements,"minThreshold":1}
            }
        }

    data = json.dumps(data)

    url = "http://api.ft.com/content/search/v1?apiKey={}".format(api_key)
    r = requests.post(url,data = data,headers = {'Content-type': 'application/json'})
    return r








def query_organization(query_string,api_key = API_KEY,max_elements = 50,year = None):
    """
    Query the Financial Times.com API for articles related to an organization or company 
    """
    json_data = {
        "queryString": query_string + build_year_filter(year = year),
        "queryContext" : {"curations" : [ "ARTICLES"]},
        "resultContext" : {
            "maxResults": 1,
            "facets" : {"names":[ "organisations","people","regions","subjects","topics"],"maxElements":max_elements,"minThreshold":1}
        }
    }

    return query(json_data = json_data,api_key = api_key)







def build_year_filter(year = None):
    """
    Helper function that creates a query filter on date for the ft API
    """
    if year is not None:
        filter_before = "lastPublishDateTime:>{}-01-01T00:00:00Z".format(year)
        filter_after = "lastPublishDateTime:<{}-01-01T00:00:00Z".format(year+1)
        return " {} {}".format(filter_before,filter_after)
    else:
        return ""


def get_facets(json_data = None, facet = None, pandas_df = False):
    """
    Gets the names of the organisations cited in the articles
    """
    facets = json_data['results'][0]['facets']
    facets_names = [facet['name'] for facet in facets]

    ind_facet = facets_names.index(facet)
    names = facets[ind_facet]['facetElements']

    if pandas_df is True:
        names = pd.DataFrame(names)
        
    return names

