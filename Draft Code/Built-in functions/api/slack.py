#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''---------------------------------------------------------------------
SLACK API
author : Theo ALVES DA COSTA
date started : 11/10/2017


API documentation : 
- http://slackapi.github.io/python-slackclient/

------------------------------------------------------------------------
'''




# Usual
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import time
import requests


# Twilio
from slackclient import SlackClient







#================================================================================================================================================
# SLACK CLASS
#================================================================================================================================================


# Please contact me before using the Slack API as these are my authentications tokens and id, Theo
API_TOKEN = "xoxp-3485995581-93390501718-254900920308-cf68329611990844f6639b6de18f9358"



class Slack(object):
    def __init__(self,token = API_TOKEN):
        self.token = token
        self.client = SlackClient(token)


    def send_message(self,message,to = "#general",icon_emoji = ':panda_face:',bot_name = "MLBot",**kwargs):
        call_type = "chat.postMessage"
        call_data = {
            "channel":to,
            "text":message,
            "username":bot_name,
            "icon_emoji":icon_emoji,
            **kwargs,
        }
        self.client.api_call(call_type,**call_data)


    def upload_file(self,
            filepath,
            channels,
            filename=None,
            content=None,
            title=None,
            initial_comment=None):
        """Upload file to channel

        Note:
            URLs can be constructed from:
            https://api.slack.com/methods/files.upload/test
        """

        if filename is None:
            filename = os.path.basename(filepath)

        data = {}
        data['token'] = self.token
        data['file'] = filepath
        data['filename'] = filename
        data['channels'] = channels

        if content is not None:
            data['content'] = content

        if title is not None:
            data['title'] = title

        if initial_comment is not None:
            data['initial_comment'] = initial_comment

        filepath = data['file']
        files = {
            'file': (filepath, open(filepath, 'rb'), 'image/jpg', {
                'Expires': '0'
            })
        }
        data['media'] = files
        response = requests.post(
            url='https://slack.com/api/files.upload',
            data=data,
            headers={'Accept': 'application/json'},
            files=files)

        return response.text