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
from ekimetrics.network import utils as nk

from ekimetrics.api.facebook.page import *
from ekimetrics.api.facebook.posts import *
from ekimetrics.api.facebook.users import *
from ekimetrics.api.facebook.comments import *
from ekimetrics.api.facebook.utils import *






#================================================================================================================================================
# PAGES
#================================================================================================================================================



class Pages(object):
    """
    Ensemble of pages data representation 
    Can be used to compare the performances with multiple brands
    """
    def __init__(self,pages = None,paths = None):
        if paths is not None:
            pages  = [Page(file_path = path) for path in tqdm(paths,desc = "Loading pages")]
        self.pages = {page.name : page for page in pages}




    #-------------------------------------------------------------------------
    # OPERATORS


    def __iter__(self):
        """
        Post iterator operator
        """
        return iter(self.pages)


    def __getitem__(self,key):
        """
        Get a post
        """
        return self.pages[key]


    #-------------------------------------------------------------------------
    # GETTERS


    def get_pages_names(self):
        return list(self.pages.keys())


    #-------------------------------------------------------------------------
    # ANALYSIS



    def compare_performances(self):
        pass



    def compare_posts_types(self):
        return pd.concat([self.pages[name].compute_posts_type(rename = True) for name in self.pages],axis = 1).fillna(0.0)




    def compare_ages_distribution(self,ages_data,smoothed = True,with_plotly = False,on_notebook = False,**kwargs):
        for i,page_name in enumerate(self.pages):
            page = self.pages[page_name]
            if not hasattr(page,"fanbase"):
                fanbase = page.build_users_fanbase(inplace = False)
            else:
                fanbase = page.fanbase
            distribution = fanbase.compute_ages_distribution(ages_data = ages_data,smoothed = smoothed)
            distribution = pd.DataFrame(distribution,index = distribution.index,columns = [page_name])
            if i == 0:
                all_distributions = distribution
            else:
                all_distributions = all_distributions.join(distribution)

        fig = charts.plot_line_chart(all_distributions,with_plotly = with_plotly,on_notebook = on_notebook,
                                                       figsize = (10,7),title = "Ages distribution benchmark",
                                                       xlabel = "Age",ylabel = "Probability")

        return fig



    def compare_followers_engagement(self):
        pass


    def compare_interpenetration(self):
        names = list(self.pages.keys()) # changer en le getter
        combinations = nk.get_combinations(names)

        data = defaultdict(dict)

        for combination in tqdm(combinations):
            name1,name2 = combination
            page1,page2 = self.pages[name1],self.pages[name2]
            page1.build_users_fanbase(verbose = 0)
            page2.build_users_fanbase(verbose = 0)
            summary,_ = page1.fanbase.interpenetration(page2.fanbase)

            interpenetration_count = summary[name1]["interpenetration"]
            page1_count = summary[name1]["total"]
            page2_count = summary[name2]["total"]

            data[name1][name2] = interpenetration_count
            data[name2][name1] = interpenetration_count
            data[name1][name1] = page1_count
            data[name2][name2] = page2_count

        return pd.DataFrame(data)










