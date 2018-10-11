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

from ekimetrics.api.facebook.utils import *



#================================================================================================================================================
# USER
#================================================================================================================================================


class User(object):
    """
    User data representation
    """

    def __init__(self, data):
        """
        Initialization
        :param dict data: the input json data
        :sets: self.meta_data -- the raw json data
        :sets: self.id -- the facebook id of the user
        :sets: self.name -- the facebook name of the user
        :sets: self.first_name -- the first name extracted of the user
        """
        self.meta_data = data
        self.id = data['id']
        self.name = data.get('name', None)
        self.first_name = self.get_first_name()


    #-------------------------------------------------------------------------------------------
    # REPRESENTATIONS AND OPERATORS

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.id == other.id



    #-------------------------------------------------------------------------------------------
    # GETTERS


    def get_data(self):
        """
        Get the raw json data
        """
        return self.meta_data


    def get_first_name(self):
        """
        Create from the user name the first name of the user
        """
        if self.name is not None:
            return self.name.split(" ")[0]
        else:
            return None


    def get_FIRST_NAME(self):
        if self.first_name is None:
            return None
        else:
            return unidecode.unidecode(self.first_name).upper()


    def guess_gender(self,detector = None,inplace = True):
        """
        Guess the gender of the user
        """
        if self.first_name is None:
            return None
        else:
            value = gender.guess_gender(self.first_name,detector = detector)
            if inplace:
                self.gender = value
            else:
                return value



    def guess_age(self,data,inplace = True):

        FIRST_NAME = self.get_FIRST_NAME()

        if FIRST_NAME in data.index:
            age = data.loc[FIRST_NAME]
        else:
            age = None

        if inplace:
            self.age = age
        else:
            return age




    def get_age(self,as_distribution = True,smoothed = True):
        if self.age is None:
            return None

        else:
            if as_distribution:
                age = self.age / self.age.sum()
            else:
                age = self.age

            if smoothed:
                age = smooth_series(age)

            return age






#================================================================================================================================================
# USERS
#================================================================================================================================================


class Users(object):
    """
    Ensemble of users class representation
    """
    def __init__(self,users):
        """
        Initialization
        """
        self.users = users



    #-------------------------------------------------------------------------------------------
    # OPERATORS

    def __iter__(self):
        return iter(self.users)


    def __getitem__(self,key):
        return self.users[key]



    #-------------------------------------------------------------------------------------------
    # AGE ANALYSIS


    def compute_ages_data(self,ages_data,as_ages = True,as_distribution = True,smoothed = False,max_age = 80,min_age = 12):

        # Get the first names
        users_name = pd.DataFrame([user.get_FIRST_NAME() for user in self.users],columns = ["name"]).set_index("name")

        # Merge with the ages data
        self.ages = users_name.join(ages_data)

        # Drop the first names without ages distribution data
        self.ages = self.ages.dropna(axis = 0)

        # Filter on max and min age
        year_today = pd.to_datetime("today").year
        if max_age is not None:
            year_max = year_today - max_age
            correct_years = [year for year in self.ages.columns if int(year)>=year_max]
            self.ages = self.ages[correct_years]

        if min_age is not None:
            year_min = year_today - min_age
            correct_years = [year for year in self.ages.columns if int(year)<year_min]
            self.ages = self.ages[correct_years]


        # Convert to ages
        if as_ages:
            self.ages.columns = [year_today-int(year) for year in self.ages.columns]
            self.ages = self.ages[sorted(self.ages.columns)]


        # Convert quantities to distributions
        if as_distribution:
            self.ages = self.ages.divide(self.ages.sum(axis = 1),axis = 0)

        # Smooth the distributions
        if smoothed:
            pass




    def compute_ages_distribution(self,ages_data,smoothed = True,**kwargs):

        # Recompute the ages data for all first names
        self.compute_ages_data(ages_data = ages_data,as_distribution = True,smoothed = False,**kwargs)

        # Take the mean distribution
        distribution = self.ages.mean(axis = 0)

        # Smooth the distribution by interpolation
        if smoothed:
            distribution = smooth_series(distribution)

        return distribution


        


    def show_ages_distribution(self,ages_data,smoothed = True,with_plotly = True,on_notebook = True,**kwargs):

        # Compute the mean distribution
        distribution = self.compute_ages_distribution(ages_data = ages_data,smoothed = smoothed,**kwargs)

        # Plot with the helper function
        fig = charts.plot_line_chart(pd.DataFrame(distribution),with_plotly = with_plotly,on_notebook = on_notebook,
                                    figsize = (10,7),title = "{}'s ages distribution".format(self.page_name),
                                    xlabel = "Age",ylabel = "Probability",
                                    width = 800,height = 600)
        
        return fig        






#================================================================================================================================================
# FANBASE
#================================================================================================================================================




class Fanbase(Users):
    def __init__(self,data,page_name = "",ages_data = None):
        
        # Store the data
        self.data = data
        self.size = len(data)
        self.genders_detected = False
        self.ages_detected = False
        self.ages_data = ages_data
        self.page_name = page_name

        # Extract the users and initialize the parent class
        users = list(self.data["user"])
        super().__init__(users = users)




    #-------------------------------------------------------------------------------------------
    # OPERATORS

    def __repr__(self):
        return "{} users that interacted with {}'s page".format(self.get_size(),self.page_name)

    def __str__(self):
        return self.__repr__()




    #-------------------------------------------------------------------------------------------
    # GETTERS


    def get_size(self):
        return self.size



    def get_top_users(self,top = 0.1):
        top = int(top*len(self.data))
        return Users(users = self.users[:top])


    def get_ages_data(self):
        if self.ages_data is None:
            raise ValueError("No ages data provided, please set it")
        else:
            return self.ages_data




    #-------------------------------------------------------------------------------------------
    # SETTERS


    def set_ages_data(self,ages_data,to_each_user = False):
        self.ages_data = ages_data

        if to_each_user:
            tqdm.pandas(desc="Setting the age to each user")
            self.data["user"].progress_map(lambda user : user.guess_age(data = ages_data,inplace = True))







    #-------------------------------------------------------------------------------------------
    # GENDER ANALYSIS


    def guess_fanbase_gender(self,top = 1,force_guess = False,with_details = False):
        """
        Find the gender distribution estimation on the fanbase (users having interacted with the page)
        """

        print(">> Guessing gender distribution of the fanbase")

        # Initialization of the parameters
        if type(top) != list: top = [top]

        # Register tqdm with pandas
        tqdm.pandas(desc="Guessing gender")

        # Instantiate the gender detector
        detector = gender.Detector(case_sensitive = False)

        # Find all the genders
        if force_guess or not self.genders_detected:
            self.data["user"].progress_map(lambda user : user.guess_gender(detector = detector,inplace = True))
            self.genders_detected = True

        # Iterating on each segment of the fanbase
        for i,limit in enumerate(top):
            limit_str = "top "+str(int(limit*100))+"%"
            limit = int(limit*len(self.data))
            users = self.data.head(limit)[["user"]]
            users["gender"] = users["user"].map(lambda user : user.gender)
            users = users.groupby("gender").count()
            users.columns = [limit_str]
            users[limit_str+" ratio"] = (users[limit_str]/users[limit_str].sum()).map(lambda x:round(x,3))

            if i == 0:
                df = users.copy()
            else:
                df = pd.concat([df,users],axis = 1)


        # Remove androgynous and unkwown
        if not with_details:
            df.drop(["androgynous","unknown"],inplace = True)
            for field in df.columns:
                if 'ratio' in field:
                    df[field] = df[field]/df[field].sum()


        return df.fillna(0)





    def show_gender(self,by_post = False,top = 1,with_plotly = True,on_notebook = True,kind = "pie",ratios = True,with_details = False):
        assert kind in ["pie","bar"]

        # Get the data
        data = self.guess_fanbase_gender(top = top,with_details = with_details)
        select_field = lambda field : ratios if "ratio" in field else not ratios
        data = data[[f for f in data.columns if select_field(f)]]


        # Title
        title = "{}'s users gender distribution".format(self.page_name)

        # If plot pie chart
        if kind == "pie":
            fig = charts.plot_pie_chart(data,with_plotly = with_plotly,on_notebook = on_notebook,title = title)

        # If plot bar chart
        else:
            fig = charts.plot_bar_chart(data,with_plotly = with_plotly,on_notebook = on_notebook)

        if not on_notebook:
            return fig





    def compute_engagement(self,top = 0.1,axis = "total_reactions"):

        # Initialization of the parameters
        if type(top) != list: top = [top]
        top.append(1)

        # Initialize a dictionary to store the data
        engagement = {}

        # Iterating on each segment of the fanbase
        for i,limit in enumerate(top):
            limit_str = "top "+str(int(limit*100))+"%" if limit != 1.0 else "100%"
            limit = int(limit*len(self.data))
            value = self.data.head(limit)[axis].sum()
            engagement[limit_str] = value

        return engagement




    def show_interaction_type(self,with_plotly = False,on_notebook = True):
        data = self.data.sum().drop(["total_reactions","total_interactions","reactions_ratio"])
        data = pd.DataFrame(data.sort_values(ascending = False))
        data.index = [x.upper() for x in data.index]

        charts.plot_bar_chart(data,with_plotly = with_plotly,on_notebook = on_notebook,
                                title = "Number of interactions by type",
                                figsize = (10,4),
                                xlabel = "Interaction",ylabel = "Number")



    #-------------------------------------------------------------------------------------------
    # AGE ANALYSIS







    def compare_fanbase_age(self,ages_data = None,top = 0.1,smoothed = True,with_plotly = False,on_notebook = True,**kwargs):
        # Initialization of the parameters
        if type(top) != list: top = [top]
        top.append(1)

        # Get ages data
        ages_data = self.get_ages_data() if ages_data is None else ages_data

        # Iterating on each segment of the fanbase
        for i,limit in enumerate(top):
            limit_str = "top "+str(int(limit*100))+"%" if limit != 1.0 else "100%"

            users = self.get_top_users(top = limit)
            distribution = users.compute_ages_distribution(ages_data = ages_data,smoothed = smoothed,**kwargs)
            distribution = pd.DataFrame(distribution,index = distribution.index,columns = [limit_str])
            if i == 0:
                all_distributions = distribution
            else:
                all_distributions = all_distributions.join(distribution)

        fig = charts.plot_line_chart(all_distributions,with_plotly = with_plotly,on_notebook = on_notebook,
                                                       figsize = (10,7),title = "{}'s ages distribution by engagement".format(self.page_name),
                                                       xlabel = "Age",ylabel = "Probability",
                                                       width = 800,height = 600)

        return fig







    #-------------------------------------------------------------------------------------------
    # INTERPENETRATION


    def interpenetration(self,other,how = "inner"):
        
        # Merge the interactions data via the user facebook id
        data = self.data.join(other.data,how = how,lsuffix = " ("+self.page_name+")",rsuffix = " ("+other.page_name+")")

        # Simplify the "users X" columns
        col1,col2 = [x for x in data.columns if "user" in x]
        data = data.rename(columns = {col1:"user"}).drop(col2,axis = 1)

        # Calculate the interpenetration summary
        summary = {}
        summary[self.page_name] = {"total":self.size,"interpenetration":len(data),"ratio":len(data)/self.size}
        summary[other.page_name] = {"total":other.size,"interpenetration":len(data),"ratio":len(data)/other.size}

        return pd.DataFrame(summary),Fanbase(data = data,page_name = self.page_name + " & "+other.page_name)




