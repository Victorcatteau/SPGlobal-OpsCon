#!/usr/bin/env python
# -*- coding: utf-8 -*- 


from functools import wraps
import datetime
import platform
from ekimetrics.api.slack import Slack
import time
from openpyxl import load_workbook
import win32com.client
import pandas as pd


from ekimetrics.gdpr.utils import *



#=======================================================================================================================================
# DECORATOR
#=======================================================================================================================================



def GDPR(
    register=None,
    slack=False,slack_to="#nissanscoring",
    teams=False,
    file=None,
    mail=False,mail_to="theo.alvesdacosta@ekimetrics.com",
    ):
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            # Arguments
            function_name = func.__name__
            now = datetime.datetime.now()
            username = platform.node()
            
            message = f"[GDPR] User '{username}' has used the function {function_name} at {now}"
            print(message)
            
            # Send to slack
            if slack:
                slack_client = Slack()
                slack_client.send_message(message.replace("[GDPR] ",""),to=slack_to,icon_emoji=":classical_building:",bot_name="GDPR Bot")
                
            # Send by mail
            if mail:
                send_mail(message,to=mail_to)
                
            # Save in the register
            if register is not None:
                if not register.endswith(".xlsx"):
                    mission = register
                    registre = mission_registre[mission]
                else:
                    mission = registre_mission[register]
                    registre = register
                    
                # Open workbook
                workbook = load_workbook(registre)
                sheet = workbook.worksheets[0]
                
                # Prepare row data
                responsable = user_data.loc[username,"name"]
                departement = user_data.loc[username,"departement"]
                date = now.date()
                date_complete = now
                if not function_name in task_data.index: function_name = "misc"
                tache = task_data.loc[function_name,"name"]
                description = task_data.loc[function_name,"description"]
                langage = "Python"
                environnement = "local"
                
                # Define row data
                row = [mission,responsable,departement,date,date_complete,tache,description,langage,environnement]
                
                # Append row to workbook
                sheet.append(row)
                workbook.save(registre)
                    
                
                

            return func(*args, **kwargs)
        return wrapper
    return decorator