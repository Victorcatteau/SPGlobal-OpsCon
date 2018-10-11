#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
GOOGLE STREETVIEW
Started on the 2018/01/08

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import requests
from PIL import Image
from io import BytesIO

# Custom library
from ekimetrics.api import utils






class StreetView(object):
    def __init__(self,api_key):
        """
        Initialization
        """
        print(">> Instantiation of Streetview API wrapper")
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
        self.api_key = api_key


    def build_image_url(self,location = "48.8727114,2.2996011",size = "400x400",fov = 90,heading = 235,pitch = 10,**kwargs):
        """
        Does not require an API KEY
        """
        return utils.build_url(self.base_url,location = location,size = size,fov = fov,heading = heading,pitch = pitch,**kwargs)

    def build_meta_url(self,location = "48.8727114,2.2996011",size = "400x400",fov = 90,heading = 235,pitch = 10,**kwargs):
        """
        Requires an API KEY
        """
        return utils.build_url(self.base_url+"/metadata",location = location,size = size,fov = fov,heading = heading,pitch = pitch,key = self.api_key,**kwargs)


    def get_image(self,location = "48.8727114,2.2996011",as_image = True,meta = False,**kwargs):
        """
        Get an image from Streetview API
        """
        if meta:
            url = self.build_meta_url(location = location,**kwargs)
            return requests.get(url).json()
        else:
            url = self.build_image_url(location = location,**kwargs)
            bytes_content = BytesIO(requests.get(url).content)

            if as_image:
                return Image.open(bytes_content)
            else:
                return url




