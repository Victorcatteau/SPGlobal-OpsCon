#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
GEOCODING API WRAPPER
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
import geopy
from geopy.geocoders import Nominatim

# Custom library
from ekimetrics.api import utils




COUNTRY_CODE = {
    "France":"FR",
    "La Réunion":"RE"
}

CODE_COUNTRY = {v:k for k,v in COUNTRY_CODE.items()}


#==========================================================================================================================================================================
# GEOCODER MAIN CLASS
#==========================================================================================================================================================================



class Geocoder(object):
    """
    Geocoder using various APIs methods
    """
    def __init__(self,method = "nominatim",google_api_token = "AIzaSyAzjp6H9ZHtmUjf7b0mlgkGOrjD7sktFHQ"):
        """
        Initialisation
        """

        # Checks and assertions
        assert method in ["google","nominatim"]
        if method == "google" and google_api_token is None:
            raise Exception("Google method requires a token")

        # Preparing geolocator for nominatim
        if method == "nominatim":
            self.geolocator = Nominatim()

        # Storing attributes
        self.method = method
        self.token = google_api_token



    def get(self,address,method = None,country = None,first = True,google_api_token = None):
        """
        Transform an address to geo coordinates latitutes and longitudes
        :param address: address to look up
        :param country: a list of countries to target the address or a single country, None takes all the countries
        :param first: returns the best result, otherwise give a list of results
        """

        method = self.method if method is None else method
        token = self.token if google_api_token is None else google_api_token

        # Using Google geocoding
        if method == "google":
            results = requests.get("https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}".format(address,token)).json()["results"]
            results = [GoogleAddress(r) for r in results]
        
        # Using Nominatim geocoding
        elif method == "nominatim":
            results = self.geolocator.geocode(address,exactly_one = False)
            if results is None:
                results = self.get(address,method = "google",google_api_token = token,first = False,country = None)
                country = self.get_country_code(country)
            else:
                results = [NominatimAddress(r) for r in results]
                country = self.get_country_name(country)

        if results is None:
            return None

        # Filtering on the countries
        if country is not None:
            if type(country) == list:
                all_results = []
                for c in country:
                    all_results.extend([r for r in results if r.is_in_country(c)])
                results = all_results
            else:
                results = [r for r in results if r.is_in_country(country)]

        # Select the first results or not
        if not first:
            return results
        else:
            if len(results) > 0:
                return results[0]
            else:
                return None



    def get_country_name(self,code):
        if type(code) == list:
            return [self.get_country_name(c) for c in code]
        else:
            if code in CODE_COUNTRY:
                return CODE_COUNTRY[code]
            else:
                return code


    def get_country_code(self,country):
        if type(country) == list:
            return [self.get_country_code(c) for c in country]
        else:
            if country in COUNTRY_CODE:
                return COUNTRY_CODE[country]
            else:
                return country




#==========================================================================================================================================================================
# GOOGLE GEOCODING API RESULTS PARSER
#==========================================================================================================================================================================



class GoogleAddress(object):
    """
    Google Address ontology
    """
    def __init__(self,data):
        """
        Initialization
        """
        self.data = data


    #------------------------------------------------------------------------
    # OPERATORS

    def __repr__(self):
        return self.get_name()

    def __str__(self):
        return self.get_name()


    #------------------------------------------------------------------------
    # GETTERS

    def get_name(self):
        return self.data.get("formatted_address")

    def get_country(self,short = True):
        return self.get_component(axis = "country",short = short)

    def get_component(self,axis = "country",short = True):
        components = self.data.get("address_components")
        component = [x for x in components if axis in x.get("types")][0]
        if short:
            return component["short_name"]
        else:
            return component["long_name"]


    def get_coordinates(self):
        coordinates = self.data["geometry"]["location"]
        lat,lng = coordinates["lat"],coordinates["lng"]
        return lat,lng


    def get_data(self):
        lat,lng = self.get_coordinates()
        name = self.get_name()

        return {"lat":lat,"lng":lng,"label":name}

    def is_in_country(self,country):
        return self.get_country() == country



#==========================================================================================================================================================================
# NOMINATIM GEOCODING API RESULTS PARSER
#==========================================================================================================================================================================


class NominatimAddress(object):
    def __init__(self,location):
        self.location = location

    #------------------------------------------------------------------------
    # OPERATORS

    def __repr__(self):
        return self.get_name()

    def __str__(self):
        return self.get_name()

    #------------------------------------------------------------------------
    # GETTERS

    def get_name(self):
        return self.location.address


    def get_coordinates(self):
        lat = self.location.latitude
        lng = self.location.longitude
        return lat,lng

    def get_data(self):
        lat,lng = self.get_coordinates()
        name = self.get_name()
        return {"lat":lat,"lng":lng,"label":name}


    def is_in_country(self,country):
        return country.lower() in self.location.address.lower()
