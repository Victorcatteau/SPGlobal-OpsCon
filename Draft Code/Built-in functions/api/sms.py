#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''---------------------------------------------------------------------
SMS API
author : Theo ALVES DA COSTA
date started : 11/10/2017


API documentation : 
- https://www.twilio.com/docs/libraries/python

Other APIs: 
https://textbelt.com/
http://www.domotique-info.fr/2014/06/nouvelle-api-sms-chez-free/
------------------------------------------------------------------------
'''




# Usual
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import time


# Twilio
from twilio.rest import Client






#================================================================================================================================================
# TWILIO CLASS
#================================================================================================================================================


# Please contact me before using the Twilio API as these are my authentications tokens and id, Theo



class Twilio(object):
    def __init__(self,account_sid,auth_token,from_number):
        self.client = Client(account_sid,auth_token)
        self.from_number = from_number


    def send_sms(self,to,message):
        self.client.messages.create(to=to, from_= self.from_number,body=message)