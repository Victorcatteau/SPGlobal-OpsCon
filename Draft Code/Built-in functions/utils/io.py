#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
CONNECTIONS

Started on the 31/08/2017

Query operator : https://docs.mongodb.com/manual/reference/operator/query/


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



# USUAL
import pandas as pd
import numpy as np
import time
import datetime
import json
from tqdm import tqdm
import requests
import random
import h5py


try:
    from pymongo import MongoClient
except:
    print("Ekimetrics warning : Pymongo is not installed, please install it to use the mongodb capabilities")




#=============================================================================================================================
# JSON UTILS
#=============================================================================================================================


def open_json_data(json_path,encoding = 'utf8'):
    return json.loads(open(json_path,"r",encoding = encoding).read())


def save_data_as_json(data,json_path,sort_keys = True):
    with open(json_path, 'w') as file:
        json.dump(data, file,indent = 4,sort_keys = sort_keys)





#=============================================================================================================================
# MONGODB UTILS
#=============================================================================================================================


def save_data_in_collection(data,database,collection,client = None,host = None,port = 27017,verbose = 1):
    if type(data)!= list:
        data = [data]
    if client is None:
        client = MongoClient(host,port)
    collection_db = client[database][collection]
    collection_db.insert_many(data)
    if verbose: print(">> Correctly uploaded data in the database")




def export_collection_to_json(json_path,database,collection,client = None,host = None,port = 27017):
    if client is None:
        client = MongoClient(host,port)
    data = retrieve_data_from_collection(database,collection,client)
    save_data_as_json(data,json_path)




def export_json_to_collection(json_path,database,collection,client = None,host = None,port = 27017):
    if client is None:
        client = MongoClient(host,port)
    data = open_json_data(json_path)
    save_data_in_database(data,database,collection,client)




def retrieve_data_from_collection(database,collection,client = None,host = None,port = 27017):
    collection_db = open_collection_data(database,collection,client,host,port)
    data = []
    for d in tqdm(collection_db.find()):
        data.append(d)
    return data



def open_collection_data(database,collection,client = None,host = None,port = 27017):
    if client is None:
        client = MongoClient(host,port)
    collection_db = client[database][collection]
    return collection_db




#=============================================================================================================================
# HDFS UTILS
#=============================================================================================================================



def open_h5_data(file_path,keys):
    h5f = h5py.File(file_path,'r')
    if type(keys) == list:
        outputs = []
        for key in keys:
            outputs.append(h5f[key][:])

        h5f.close()
        return outputs

    else:
        output = h5f[keys][:]
        h5f.close()
        return output




def save_h5_data(file_path,keys,data):
    h5f = h5py.File(file_path, 'w')
    if type(keys) == list:
        assert type(data) == list
    else:
        keys = [keys]
        data = [data]

    for i,key in enumerate(keys):
        h5f.create_dataset(key, data=data[i])
    h5f.close()