#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
HSBC
theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import json




#=============================================================================================================================
# BLACKLIST MATCHING
#=============================================================================================================================



def match_with_blacklist(url,blacklist):
    meta = {}
    for token in blacklist:
        token = str(token)
        if token in str(url):
            if token in meta:
                meta[token] += 1
            else:
                meta[token] = 1

    return meta



def extract_count_from_meta(meta):
    return np.sum(list(meta.values()))




