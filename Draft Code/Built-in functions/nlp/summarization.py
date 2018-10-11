#!/usr/bin/env python
# -*- coding: utf-8 -*- 




"""--------------------------------------------------------------------
NATURAL LANGUAGE PROCESSING SUMMARIZATIONS FUNCTIONS

Started on the 23/11/2017


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk

from gensim import summarization as sm

from ekimetrics.nlp import utils as nlp_utils




#=============================================================================================================================
# SUMMARIZATION
#=============================================================================================================================



def extract_keywords(text,filter_stopwords = None):
    """
    Keyword extraction function with stopwords filtering
    """
    
    keywords = sm.keywords(text).split("\n")


    if filter_stopwords is not None:
        if type(filter_stopwords) != list: filter_stopwords = [filter_stopwords]
        stopwords = nlp_utils.get_stop_words(filter_stopwords)
        keywords = list(set(keywords) - set(stopwords))


    return keywords


