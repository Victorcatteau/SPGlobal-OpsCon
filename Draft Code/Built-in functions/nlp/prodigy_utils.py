#!/usr/bin/env python
# -*- coding: utf-8 -*- 


__author__ = "Theo"


"""--------------------------------------------------------------------
PRODIGY UTILS
Grouping various scripts and functions for nlp 

Started on the 2018/01/22

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import json
import pandas as pd




def export_jsonl_file(data,meta,file_path,text_column = None,encoding = 'utf8'):
    if type(data) == list:
        assert type(meta) == dict
        with open(file_path,"w",encoding = encoding) as json_file:
            for text in data:
                if not pd.isnull(text):
                    json_file.write(json.dumps({"text":text,"meta":meta})+"\n")


    else:
        assert type(meta) == list
        assert text_column is not None
        with open(file_path,"w",encoding = encoding) as json_file:
            for i,row in data.iterrows():
                if not pd.isnull(row[text_column]):
                    json_file.write(json.dumps({"text":row[text_column],"meta":{x:row[x] for x in meta}})+"\n")

