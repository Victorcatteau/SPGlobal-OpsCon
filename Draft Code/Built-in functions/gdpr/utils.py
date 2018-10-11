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

import math


#=======================================================================================================================================
# HELPERS
#=======================================================================================================================================


import win32com.client

def send_mail(message,to="theo.alvesdacosta@ekimetrics.com"):
    olMailItem = 0x0
    obj = win32com.client.Dispatch("Outlook.Application")
    newMail = obj.CreateItem(olMailItem)
    newMail.Subject = 'GDPR Report'
    newMail.Body = message
    newMail.To = to
    newMail.Send()




#=======================================================================================================================================
# CREATE A REGISTER
#=======================================================================================================================================



def convert_markdown_to_excel_register(file_path,save_excel = True,n_by_level = 100):

    # Read file
    md = open(file_path,encoding = "utf-8").read()


    # Clean markdown file
    for i in range(2,10):
        md = md.replace("\n"*i,"\n")

    # Split by entry in register
    md = md.strip().split("\n")

    # Find possible tags and initialize counter
    tags = list(set([entry.split(" ")[0] for entry in md]))
    tags = [tag for tag in tags if len(tag) > 0]
    max_power = len(tags)
    power_multiplier = int(math.log10(n_by_level))
    counter = {10**(power_multiplier*i):0 for i in range(max_power)}

    # Placeholders for for loop
    last_power = max_power
    register = []

    # Loop over each entry in the register
    for entry in md:
        split = entry.find(" ")
        tag = entry[:split]
        if len(tag) == 0: continue
        entry = entry[split+1:]
        power = len(tags)-len(tag)
        level = 10**(power_multiplier*power)
        raw_number = counter[level] + level
        number = sum([counter[10**(power_multiplier*i)] for i in range(power,max_power)]) + level
        counter[level] = raw_number
        
        if power > last_power:
            for i in range(0,power):
                counter[10**(power_multiplier*i)] = 0
        
        last_power = power
        
        register.append({"number":number,"entry":entry})

    # Create a dataframe with numbers
    register = pd.DataFrame(register,columns = ["number","entry"])

    # Save as excel
    if save_excel:
        file_name = file_path.replace(".md","-md.xlsx")
        register.to_excel(file_name)
        print(f"Register saved at {file_name}")

    return register