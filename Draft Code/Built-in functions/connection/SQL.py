#!/usr/bin/env python
# -*- coding: utf-8 -*- 

'''---------------------------------------------------------------------
    EKIMETRICS

    author : Theo ALVES DA COSTA
    date started : 01/12/2016 

    25/11/2016
   ---------------------------------------------------------------------
'''



import numpy as np
import pandas as pd
import pyodbc as odbc
import time
import os
from datetime import datetime




'''---------------------------------------------------------------------
   SQL CONNECTION
   - http://docs.sqlalchemy.org/en/rel_0_9/dialects/mssql.html#dialect-mssql-pyodbc-connect
   ---------------------------------------------------------------------
'''

class SQL_connection():
    def __init__(self, SQL_server = "EkiDataCenter", ID = "talvesdacosta", database = "NCD_DM_DATAMINING_ANALYSIS", verbose = True):

        # self.connection_string = 'DSN={0};'.format(SQL_server)
        self.connection_string = 'Server={0};'.format(SQL_server)
        self.connection_string += 'Description=Ekimetrics;'
        self.connection_string += 'UID={0};'.format(ID)
        self.connection_string += 'Trusted_Connection=Yes;'
        self.connection_string += 'Statistics Common;'
        self.connection_string += 'WSID=MANUELDAVY-PC;'
        self.connection_string += 'Driver={SQL SERVER};'
        self.connection_string += 'DATABASE={0}'.format(database)

        self.connection = odbc.connect(self.connection_string)
        
        if verbose:
            print(">> Connection established to %s"%database, end = '\n\n')

    def query(self,request,file=""):
        SQL_request = request
        if len(file)>0:
            pass

        return pd.read_sql(SQL_request,self.connection)













    

