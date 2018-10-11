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
from ekimetrics.computer_vision import utils as cv_utils
from ekimetrics.computer_vision.models import ComputerVisionModel

from ekimetrics.api.facebook.users import *
from ekimetrics.api.facebook.comments import *
from ekimetrics.api.facebook.utils import *





#================================================================================================================================================
# POST
#================================================================================================================================================


class Post(object):
    """
    Post data representation
    """

    def __init__(self, data, details=True, reload=False, page="", post_number=0, posts_count=1):
        """
        Initialization
        """

        # DATA PARSING
        self.meta_data = data
        self.id = data['id']
        self.type = data['type']
        self.creator = data['from']['name']
        self.page_name = page
        self.admin = (self.creator == page) if page != "" else None
        self.message = data.get('message', 'No post message')
        self.name = data.get('name', None)
        self.link = data.get('link', None)
        self.picture = data.get('full_picture', None)
        self.date = str(pd.to_datetime(data['created_time']))
        self.connection = FacebookConnection()

        # IF PARSING THE LIKES AND COMMENTS
        # NB : Likes are included in the reactions
        if details and not reload:
            message = "... Post {}/{} : ".format(post_number + 1, posts_count)

            # Parsing the comments
            self.comments_count = safeget(data, ['comments', 'summary', 'total_count'], 0)
            self.comments = self.connection.get_all_data(data.get('comments', None), count=self.comments_count, message=message + "comments")
            self.meta_data["comments"]["data"] = self.comments
            
            # Parsing the reactions
            self.reactions_count = data['reactions']['summary']['total_count']
            self.reactions = self.connection.get_all_data(data.get('reactions', None), count=self.reactions_count, message=message + "reactions")
            self.meta_data["reactions"]["data"] = self.reactions

        # RELOADING FROM A SAVED JSON
        elif reload:
            # Parsing the comments
            self.comments_count = safeget(data, ['comments', 'summary', 'total_count'], 0)
            self.comments = safeget(data, ["comments", "data"], [])

            # Parsing the reactions
            self.reactions_count = safeget(data, ["reactions", "summary", "total_count"], 0)
            self.reactions = safeget(data, ["reactions", "data"], {})

        # OTHERWISE JUST GETTING THE COUNT
        else:

            # Parsing the comments
            self.comments = []
            self.comments_count = safeget(data, ['comments', 'summary', 'total_count'], 0)

            # Parsing the reactions
            self.reactions = {}
            self.reactions_count = safeget(data, ['reactions', 'summary', 'total_count'], 0)

        # PARSING AND SORTING REACTIONS AND COMMENTS
        # Parsing the comments
        self.comments = [Comment(x) for x in self.comments]

        # Sorting the reactions by type
        self.reactions = self.sort_reactions_by_type(self.reactions)



    #-------------------------------------------------------------------------------------------
    # OPERATOR

    def __repr__(self):
        return self.message


    def _repr_html_(self):
        html = """
        <h4>{} post</h4>
        <p>{}</p>
        <ul>
            <li><b>Id</b> : {}</li>
            <li><b>Date</b> : {}</li>
            <li><b>Type</b> : {}</li>
            <li><b>Likes</b> : {}</li>
            <li><b>Comments</b> : {}</li>
        </ul>
        <img src = '{}'/>
        """.format(self.page_name,self.message,
                    self.id,
                    self.date,
                    self.type,
                    self.get_performances().get("likes",0),
                    self.comments_count,
                    self.picture)
        return html



    def __str__(self):
        return self.message


    def __gt__(self,date):
        if date is None:
            return True
        else:
            date = pd.to_datetime(date)
            return pd.to_datetime(self.date) > date

    def __ge__(self,date):
        if date is None:
            return True
        else:
            date = pd.to_datetime(date)
            return pd.to_datetime(self.date) >= date

    def __lt__(self,date):
        if date is None:
            return True
        else:
            date = pd.to_datetime(date)
            return pd.to_datetime(self.date) < date

    def __le__(self,date):
        if date is None:
            return True
        else:
            date = pd.to_datetime(date)
            return pd.to_datetime(self.date) <= date



    #-------------------------------------------------------------------------------------------
    # GETTERS




    def get_data(self):
        """
        Returns the original json of meta data for the post from the API
        Will be used to reload and save the data in a json file
        """
        return self.meta_data



    def get_performances(self):
        """
        Get a dictionary with the different performances indicators for the post
        """
        performances = {}
        performances["comments"] = self.comments_count
        for reaction in self.reactions:
            key = reaction.lower() if reaction != "LIKE" else "likes"
            performances[key] = len(self.reactions[reaction])
        return performances




    def get_intersection_reactions_comments(self):
        """
        Find the intersection between user that commented and reacted
        """
        users_with_comments = [comment.user.id for comment in self.comments]
        reactions = self.reactions

        intersection = {}

        for reaction_type in reactions:
            users_with_reactions = [reaction.id for reaction in self.reactions[reaction_type]]
            users_intersection = list(set(users_with_comments).intersection(set(users_with_reactions)))
            intersection[reaction_type] = [comment for comment in self.comments if comment.user.id in users_intersection]

        return intersection






    #-------------------------------------------------------------------------------------------
    # UTILS

    def sort_reactions_by_type(self, data):
        """
        Sort the reaction by type
        """
        sorted_data = defaultdict(list)
        for reaction in data:
            sorted_data[reaction['type']].append(User(reaction))

        return sorted_data




    def guess_fanbase_gender(self):
        pass





    def build_comments_corpus(self):
        pass




    #-------------------------------------------------------------------------------------------
    # COMPUTER VISION


    def _is_photo(self):
        """
        Helper function that indicates if a post is a picture or not
        """
        return self.type == "photo" and self.picture is not None


    def get_picture(self):
        """
        Gets the Image if the post is a photo
        """
        if self._is_photo():
            return cv_utils.open_image_from_url(self.picture)
        else:
            return "No picture"



    def download_picture(self,path = None,base_path = "",store_as_attribute = False):
        """
        Download the picture if the post is a photo
        """

        if self._is_photo():

            if path is None:
                path = "{}.png".format(self.id)

            img = cv_utils.open_image_from_url(self.picture)

            if store_as_attribute:
                self.img = img

            img.save(os.path.join(base_path,path))




    def store_picture(self):
        """
        Retrieve and store the image from the picture url
        """
        if self._is_photo():
            self.img = cv_utils.open_image_from_url(self.picture)
        else:
            self.img = None






    def analyze_picture(self,model):
        """
        Analyze the picture using computer vision algorithms
        """

        if self._is_photo():

            # Getting the image
            img = self.get_picture()

            # Preparing the computer vision model
            if type(model) == str:
                model = ComputerVisionModel(name = model)

            # Prediction
            prediction = model.predict(img)

            return prediction

        else:
            return "No picture to analyze"

            
















