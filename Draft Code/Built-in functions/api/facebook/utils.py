#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''---------------------------------------------------------------------
    FACEBOOK SCRAPPING
    author : Theo ALVES DA COSTA
    date started : 01/12/2016 
   ---------------------------------------------------------------------
'''


#-------------------------------------------------------------------------
# LIBRARIES

# Usual
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import requests
import json
import datetime
from urllib import request
from collections import defaultdict

# Others
import unidecode
import bs4
from scipy.interpolate import UnivariateSpline

# Ekimetrics library
from ekimetrics.nlp.utils import extract_emojis
from ekimetrics.nlp.models import *
from ekimetrics.utils.time import sec_to_hms
from ekimetrics.utils import gender
from ekimetrics.visualizations import charts




#================================================================================================================================================
# Utils
#================================================================================================================================================


def safeget(dictionary, keys, default=None):  # Utiliser *args
    if type(keys) == str:
        keys = [keys]
        print(keys)
    for key in keys:
        # Returns None if key not in dict keys
        dictionary = dictionary.get(key, default)

    return dictionary




def smooth_series(s,k = 5):
    x = range(len(s))
    spline = UnivariateSpline(x,s,k = 3)
    s = pd.Series(spline(x),index = s.index)
    return s




class Period(object):
    """Period representation"""
    def __init__(self,date_start = None,date_end = None):
        self.s = str(date_start) if date_start is not None else None
        self.e = str(date_end) if date_end is not None else None


    def __repr__(self):
        if self.s is None and self.e is None:
            return "on all the historical data"
        elif self.s is None and self.e is not None:
            return "before {}".format(str(self.e)[:10])
        elif self.s is not None and self.e is None:
            return "after {}".format(str(self.e)[:10])
        else:
            return "between {} and {}".format(str(self.s)[:10],str(self.e)[:10])



    def __str__(self):
        s = "..." if self.s is None else str(self.s)[:10]
        e = "..." if self.e is None else str(self.e)[:10]
        return " - ".join([s,e])

    def get(self):
        return {"date_start":self.s,"date_end":self.e}





def generate_yearly_periods(date_start = "2014",date_end = None):
    if date_end is None: date_end = str(pd.to_datetime('today').year)
    periods = [Period(s,s+1) for s in range(int(date_start),int(date_end)+1)]
    return periods




def generate_monthly_periods(date_start = "2014",date_end = "today"):
    date_range = pd.date_range(start = "{}-01-01".format(date_start),end = "today",freq = "M")
    periods = [Period(s+ pd.DateOffset(day = 1),s) for s in date_range]
    return periods










#================================================================================================================================================
# FACEBOOK CONNECTION
#================================================================================================================================================


class FacebookConnection(object):
    """
    Connection to Facebook API utils
    """

    def __init__(self,token = None):
        """
        Initialisation
        """
        if token is None: token = "1029168113872598|1e0760366892544f54a455edea11c688"
        self.access_token = token


    def get_all_data(self, data, count=None, message=""):
        """
        Getter for all the data including paging
        """
        if data is not None:
            all_data = data['data']
            while 'paging' in data and 'next' in data['paging']:

                data = self.get_data(data['paging']['next'])
                all_data += data['data']

                if count is not None:
                    print("\r{} {}/{} retrieved".format(message,len(all_data), count), end="")

            return all_data
        else:
            return []

    def get_data(self, url, connection_done=True):
        """
        Get one page of data
        """
        success = False
        i = 0
        while not success and i < 5:
            i += 1
            try:
                data = requests.get(url).json()
                if 'error' not in data.keys():
                    return data
                else:
                    if i == 1:
                        if connection_done:
                            print('... ... There was an error trying to access Facebook API : '
                                  .format(data['error']['message']))
                        else:
                            return {}
                    else:
                        print('... ... Trying again')
                    # if i <= 5:
                    time.sleep(5)
                    # else:
                    #    time.sleep(20)

            except Exception as e:
                print('... ... "{}" : error accessing Facebook API on {} URL'
                      .format(e, url))
                time.sleep(5)