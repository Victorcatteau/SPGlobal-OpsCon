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

from ekimetrics.api.facebook.users import *
from ekimetrics.api.facebook.utils import *





#================================================================================================================================================
# COMMENT
#================================================================================================================================================


class Comment(Document):
    """
    Comment data representation
    """
    def __init__(self, data,post_number=0, posts_count=1, get_replies=False):
        """
        Initialization
        """

        # PARAMETERS
        self.meta_data = data
        self.id = data.get('id', None)
        self.tags = data.get('message_tags', None)
        self.user = User(data['from'])
        self.date = str(pd.to_datetime(data['created_time']))
        self.emojis = []
        self.message = data.get('message', 'No comment')
        message = "... Post {}/{} : ".format(post_number + 1, posts_count)

        # PARSING THE COMMENTS
        if get_replies:
            self.connection = FacebookConnection()

            self.comments_count = safeget(data, ['comments', 'summary', 'total_count'], 0)
            self.comments = self.connection.get_all_data(data.get('comments', None), count=self.comments_count, message=self.message + "comments")
            self.comments = [Comment(x, get_replies=False) for x in self.comments]
           
            # Parsing the reactions
            self.reactions_count = data['reactions']['summary']['total_count']
            self.reactions = self.connection.get_all_data(data.get('reactions', None), count=self.reactions_count, message=message + "reactions")
            self.reactions = self.sort_reactions_by_type(self.reactions)



    #-------------------------------------------------------------------------------------------
    # REPRESENTATION


    def __repr__(self):
        return self.message

    def __str__(self):
        return self.message


    #-------------------------------------------------------------------------------------------
    # GETTERS

    def get_data(self):
        return self.meta_data



    def get_number_replies(self):
        pass


        
    def get_activity(self, limit=None):
        users = defaultdict(dict)

        reaction_types = []
        
        for reaction in self.reactions:
            if reaction not in reaction_types:
                reaction_types.append(reaction)

            for user in self.reactions[reaction]:
                users[user.name].setdefault('id', user.id)
                users[user.name][reaction] = users[
                    user.name].setdefault(reaction, 0) + 1
        
        if users: # If there are reactions or comments
            df = (pd.DataFrame(users).transpose().fillna(0)[['id'] + reaction_types])

            df['total_interactions'] = np.zeros(len(df['id']))

            for reaction_type in reaction_types:
                df['total_interactions'] += df[reaction_type]

            df.sort_values('total_interactions', ascending=False, inplace=True)

            return df.head(limit)
        else:
            return "No activity on the comment {}".format(self.id)




    #-------------------------------------------------------------------------------------------
    # NLP 


    def start_nlp_analysis(self,**kwargs):
        """
        Start nlp analysis 
        """
        super().__init__(text=self.message,**kwargs)


    def remove_tags(self,strict = False):
        """
        Remove tags from comment body
        """
        if self.tags is not None:
            message = self.message
            for tag in self.tags:
                start,length = tag["offset"],tag["length"]
                message = message[:start]+message[start+length:]
                message = message.strip()

            if strict:
                self.message = message
            else:
                if len(message) == 0:
                    self.message = None
                else:
                    pass                    



    def filter_emojis(self):
        self.set_text(extract_emojis(self.text)[0])
        self.emojis = extract_emojis(self.text)[1]



    def sort_reactions_by_type(self, data):
        sorted_data = defaultdict(list)
        for reaction in data:
            if "type" in reaction:
                sorted_data[reaction['type']].append(User(reaction))

        return sorted_data




#================================================================================================================================================
# COMMENTS
#================================================================================================================================================




class Comments(Corpus):
    """
    Ensemble of comments data methodology
    """
    def __init__(self, comments):
        super().__init__(documents=comments)


    def start_nlp_analysis(self,**kwargs):
        for comment in tqdm(self):
            comment.start_nlp_analysis(**kwargs)


    def remove_tags(self,strict = False):
        comments = []
        for comment in tqdm(self):
            comment.remove_tags(strict = strict)
            if comment.message is not None:
                comments.append(comment)
        self.documents = comments



    def build_jsonl_file(self,file_path,meta = None,encoding = "utf8"):
        with open(file_path,"w",encoding = encoding) as json_file:
            if meta is None:
                meta = {"source":"facebook"}
            elif type(meta) == dict:
                meta["source"] = "facebook"
            else:
                raise Exception("You must provide a dict of meta")

            for comment in tqdm(self):
                if not pd.isnull(comment.message):
                    json_file.write(json.dumps({"text":comment.message,"meta":meta})+"\n")