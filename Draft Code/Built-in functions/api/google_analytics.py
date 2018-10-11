#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import print_function


"""--------------------------------------------------------------------
GOOGLE ANALYTICS
Started on the 12/04/2017

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs

requirements : 
pip install --upgrade google-api-python-client

------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import requests

import argparse
from apiclient.discovery import build
import httplib2
from oauth2client import client
from oauth2client import file
from oauth2client import tools



SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
DISCOVERY_URI = ('https://analyticsreporting.googleapis.com/$discovery/rest')
CLIENT_SECRETS_PATH = 'client_secrets.json' 




def initialize_analyticsreporting(client_secrets_path = CLIENT_SECRETS_PATH):
    """Initializes the analyticsreporting service object.
    Returns:
    analytics an authorized analyticsreporting service object.
    """
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,parents=[tools.argparser])
    flags = parser.parse_args([])

    # Set up a Flow object to be used if we need to authenticate.
    flow = client.flow_from_clientsecrets(client_secrets_path, scope=SCOPES,message=tools.message_if_missing(CLIENT_SECRETS_PATH))

    # Prepare credentials, and authorize HTTP object with them.
    # If the credentials don't exist or are invalid run through the native client
    # flow. The Storage object will ensure that if successful the good
    # credentials will get written back to a file.
    storage = file.Storage('analyticsreporting.dat')
    credentials = storage.get()
    if credentials is None or credentials.invalid:
        credentials = tools.run_flow(flow, storage, flags)
    http = credentials.authorize(http=httplib2.Http())

    # Build the service object.
    analytics = build('analytics', 'v4', http=http, discoveryServiceUrl=DISCOVERY_URI)
    return analytics







#==================================================================================================================================
# PLOTTING UTILS
#==================================================================================================================================



class GoogleAnalytics(object):
    def __init__(self,view_id,client_secrets_path):
        self.view_id = view_id
        self.analytics = initialize_analyticsreporting(client_secrets_path)


    def get_report(self,metrics = None,dimensions = None,segments = None,date_range = None,max_results = 10000,next_page_token = None,dimensions_filters = None):
        dictionary_request = self.build_dictionary(metrics,dimensions,segments,date_range,max_results,next_page_token,dimensions_filters)
        return self.analytics.reports().batchGet(body = {"reportRequests":[dictionary_request]}).execute()

    def build_dictionary(self,metrics = None,dimensions = None,segments = None,date_range = None,max_results = 10000,next_page_token = None,dimensions_filters = None):
        base_dict = {
            "viewId":self.view_id,
            "pageSize": str(max_results)
            }

        if date_range is None:
            date_range = {'startDate': '7daysAgo', 'endDate': 'today'}

        base_dict["dateRanges"] = [date_range]

        if metrics is not None:
            metrics = [metrics] if type(metrics)!=list else metrics
            base_dict["metrics"] = [{"expression":"ga:{}".format(metric)} for metric in metrics]

        if dimensions is not None:
            dimensions = [dimensions] if type(dimensions)!=list else dimensions
            base_dict["dimensions"] = [{"name":"ga:{}".format(dimension)} for dimension in dimensions]

        if segments is not None:
            segments = [segments] if type(segments)!=list else segments
            base_dict["segments"] = [{"segmentId":"gaid::{}".format(segment)} for segment in segments]
            base_dict["dimensions"].append({"name":"ga:segment"})

        if next_page_token is not None:
            base_dict["pageToken"] = str(next_page_token)

        if dimensions_filters is not None:
            base_dict["dimensionFilterClauses"] = [{"filters": [{"dimensionName": "ga:{}".format(x),"operator": dimensions_filters[x][0],"expressions": dimensions_filters[x][1]} for x in dimensions_filters]}]


        return base_dict



    def convert_report(self,report):
        report = report["reports"][0]
        next_page = report["nextPageToken"] if "nextPageToken" in report else None
        
        #COLUMNS
        columns = report["columnHeader"]
        dimensions_columns = [x.replace("ga:","") for x in columns["dimensions"]]
        metrics_columns = [x["name"].replace("ga:","") for x in columns["metricHeader"]["metricHeaderEntries"]]
        columns = dimensions_columns + metrics_columns
        
        #ROWS
        rows = []
        for row_element in report["data"]["rows"]:
            row = row_element["dimensions"] + [x["values"] for x in row_element["metrics"]][0]
            rows.append(row)
            
        #DATAFRAME INITIALIZATION
        data = pd.DataFrame(rows,columns = columns)       


        
        return data,next_page


    def get_data(self,metrics = None,dimensions = None,segments = None,date_range = None,max_results = 10000,n_pages = None,dimensions_filters = None):
        pages_processed = 0
        next_page = True
        next_page_token = None
        full_data = pd.DataFrame()
        while (n_pages is None or pages_processed < n_pages) and next_page is not None:
            print("\rProcessing page {}".format(pages_processed + 1),end = "")
            report = self.get_report(metrics = metrics,dimensions = dimensions,segments = segments,date_range = date_range,max_results = max_results,next_page_token = next_page_token,dimensions_filters = dimensions_filters)
            data,next_page = self.convert_report(report)
            next_page_token = next_page
            pages_processed += 1
            full_data = full_data.append(data)


        return full_data

