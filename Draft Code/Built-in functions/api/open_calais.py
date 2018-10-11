#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
OPEN CALAIS
Started on the 11/10/2017

Thami BENGHAZI AKHLAKI
Theo ALVES DA COSTA
------------------------------------------------------------------------
"""


import bs4 as bs
import nltk
import pandas as pd
import numpy as np
import requests
import time
from tqdm import tqdm
from collections import defaultdict
import sys

from oddo.data import CalaisEntity



api_token ='7cQXuGZLwymrHIB7Td7g7okmvXAqSvmG' # Open Calais Api Token

key1 = 'goU9hnehKib1TGv93OJk6j6w4ey4GWPG' # ekirobot1@mail.com mpd_email= 55pierrecharron mdp_opencalais = 55Pierrecharron

key2 = 'sY917kHGFWXdcReyIZxKN6URjQyzeAAY' #  ekirobot3@laposte.net mdp=55Pierrecharron

key3 = 'T1jZydLIo0J1q8zaAqrfRQTelr9obHku'

key4 = 'ZvQAG73e4cZhU6EQdEu3St2YGcCpjmjn'

key5 = ''

key6 = '4jay1yHc1tvlkzlmwdjwEyNvsNw3ljdt'

key7 = ''



class OpenCalais(object):
    """
    This class access the Open Calais API via an api_token
    Open Calais is used to tag entities within a text data 

    """
    def __init__(self, api_token):
        self.api_token = api_token

    def get_json(self, input_data, data_type = 'text/raw', encode = False):
        """
        return a json file with information on the Open Calais tags
        data_type can be in pdf format: data_type = 'application/pdf'
        """
        calais_url = 'https://api.thomsonreuters.com/permid/calais'
        headers = {'X-AG-Access-Token' : self.api_token, 'Content-Type' : data_type , 'outputformat' : 'application/json'}
        
        # This is to avoid some encoding errors
        if encode:
            input_data = input_data.replace("’","'").encode(encoding='latin-1', errors='ignore') 

        try:
            response = requests.post(calais_url, data=input_data, headers=headers, timeout=80)
        
        except Exceptions as e:
            print(">>> Error when posting the request! Try using:  encode = True")

        response_json = response.json()
        
        # Deleting some uninteresting parts of the Json
        # del response_json["doc"]

        return response_json

    def get_tagged_companies(self, input_data):
        """
        returns the list of tagged companies within the data
        """
        
        data_calais = self.get_json(input_data = input_data)
        
        keys = list(data_calais.keys())
        
        calais = [CalaisEntity(data_calais[key]) for key in keys if key!= 'doc']
        
        tags = [{'name':cl.get_name(), 'offset': cl.get_offset(), 'context':cl.get_context(), 'public':cl.is_public(), 'info':cl.get_info_url()} for cl in calais if cl.is_company()]

        return tags

    
    def tag_list(self, list_articles):
        """
        Open Calais tags for your list of articles

        list_articles must be a list of strings (texts)
        """
        list_tags = []

        list_group = concat_articles(list_articles = list_articles, chunk_size = 99000)

        if len(list_group) > 5000:

            print("\n \n >>>>>>>> The list is too large for the number of api calls permitted daily with Open Calais, Try reducing it")
        else:

            for element in list_group:

                list_tags.extend(allocate_tags(concat_data = element['concat_data'], 
                    tags = self.get_tagged_companies(element['concat_data']), landmark_positions = element['landmarks']))

            return list_tags



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# helper function

def sub_concat_articles(list_articles, chunk_size = 99000):
    """
    OpenCalais permits 100 KB size texts per request

    This function is used is a sub function to the next
    """
    articles = [("||" + article).replace("’","'").encode(encoding='latin-1', errors='ignore') for article in list_articles]
    concat_data = b''
    landmarks = [0]

    size = 0
    list_concatenated = []

    for article in articles:
        size = size + sys.getsizeof(article)
        if size < chunk_size:
            list_concatenated.append(article)
            concat_data = concat_data + article
            landmarks.append(len(concat_data))
        else: break

    remaining_articles = articles[len(list_concatenated):]
    
    output = {"concat_data":concat_data, "list_concatenated": list_concatenated, "remaining_articles": remaining_articles, "landmarks": landmarks}
    
    return output


def concat_articles(list_articles, chunk_size = 99000):
    """
    Concatenates the articles within the list_articles and returns a list of dictionnaries 
    containing the concatenated data each with a size smaller than chunk_size
    """
    list_group = []

    output = sub_concat_articles(list_articles = list_articles, chunk_size = chunk_size)
    
    list_group.append(output)
    
    while len(output["remaining_articles"])!= 0:
        
        output = sub_concat_articles(list_articles = output["remaining_articles"], chunk_size = chunk_size)
        
        list_group.append(output)
    
    return list_group


###################################################################################
"""
Then Tagging with Open Calais happens here ! put them in "tags"
"""
###################################################################################


def all_occurences(file, str):
    """
    Returns all the positions (indices) of str in file
    """
    initial = 0
    occurences = []
    
    while True:
        initial = file.find(str, initial)
        
        if initial == -1: 
            return occurences
        
        occurences.append(initial)
        initial += len(str)



def allocate_tags(concat_data, tags, landmark_positions):
    """
    Returns a list of the articles with their corresponding tags
    """
    list_article_tags = []
    
    i = 0
    
    while i <= len(landmark_positions)-2:
        
        list_article_tags.append([tag for tag in tags if (landmark_positions[i] <= tag['offset'] < landmark_positions[i+1])]) 
        
        i += 1


    return list_article_tags






 
