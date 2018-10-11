#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
UTILS
Started on the 25/04/2017

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json








#=============================================================================================================================
# PARSING DICTIONARY
#=============================================================================================================================




def parse(dictionary,list_of_keys,none_value = None):
    """
    Recursive function that explores a dictionary to returns the values
    - It takes as input a list of keys giving the path in the tree like dictionary structure. 
    - Returns None for a node value if the key is not existing
    - You can set a list of list of keys to split the search in branches and returns multiple results

    Example for a dictionary like 
    d = {
        "A":10,
        "B":{
            "a":[{"english":"hello","french":"bonjour","german":"hallo"},{"english":"bye","french":"aurevoir","german":"auf wiedersehen"}],
            "b":30
            }
        }

    - parse(d,"A") returns 10
    - parse(d,"B") returns the dictionary {"a": ... , "b": ... }
    - parse(d,["B","b"]) returns 30
    - parse(d,["B","c"]) returns None
    - parse(d,["B","a","english"]) returns ["hello","bye"]
    - parse(d,["B","a",["english","french"]]) returns [{"english":"hello","french":"bonjour"},{"english":"bye","french":"aurevoir"}] (without the german part)

    """

    if type(list_of_keys) != list : list_of_keys = [list_of_keys]

    key = list_of_keys[0]
    other_keys = list_of_keys[1:]


    if type(key) != list:
        if key in dictionary.keys():
            data = dictionary[key]

            if len(other_keys) > 0:
                if type(data) == dict:
                    return parse(data,other_keys)
                elif type(data) == list:
                    return [parse(element,other_keys) for element in data]
                else:
                    pass

            else:
                return data

        else:
            return none_value

    else:
        return {k:parse(dictionary,k) for k in key}
        




#=============================================================================================================================
# HELPERS
#=============================================================================================================================



def build_url(base_url,**kwargs):
    """
    Helper function to build API urls with list of arguments
    """
    url = base_url + "?" + "&".join(["{}={}".format(key,kwargs[key]) for key in kwargs])
    return url
