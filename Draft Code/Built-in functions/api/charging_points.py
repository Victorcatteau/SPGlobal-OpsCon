#PATH python
# -*- coding: utf-8 -*-

'''---------------------------------------------------------------------
    Charging Points
    13/11/2017

    Last Update:
    19/12/2017

   ---------------------------------------------------------------------
'''


# Usual librairies
import numpy as np
import pandas as pd
import os
import requests
import json
import matplotlib.pyplot as plt
import sys
from time import time
from gc import collect
import copy
import folium
import geopy
from geopy.geocoders import Nominatim
from geopy.distance import vincenty, great_circle
import sys
import operator

# Ekimetrics custom library
sys.path.append('C:/Users/gcorda/Documents/GitHub/eki-python-library/')
from ekimetrics.utils import io 



'''---------------------------------------------------------------------
   URL CONSTRUCTOR CLASS - CREATE URL TO PARSE DATA
   ---------------------------------------------------------------------
'''

class UrlConstructor(object) :


    def __init__(self):
        API_url = "https://api.openchargemap.io/v2/poi/?output=json"
        self.url = API_url


    def __repr__(self) :
        return self.url


    def edit_url(self, inplace=False,show_url = 0, **kwargs):
        """
        Helper function to build the request url from a request type and any keywords additional arguments
        """

        if len(kwargs) > 0 :
            other_args = ["{}={}".format(k,str(v).replace(" ","+")) for k,v in kwargs.items()]
            new_url = self.url + "&" + "&".join(other_args)
            if show_url: print(new_url) 

            if "maxresults" not in kwargs : 
                print("Be careful : This request will only display the first 100 results.")

            if inplace:
                self.url = new_url

    
    def params_helper(self,**kwargs):
        """
        Helper function to return operations you can do on OpenChargeMap API
        """

        dic = {'output' : 'json, xml, kml',
        'maxresults' : 'limit on max number of results returned ; Default is limited to 100',
        'countrycode' : 'GB, US etc ISO Country Code ==> Only 2 caracters !',
        'latitude' : 'latitude reference for distance calculation',
        'distance' : 'return results based on specified distance from specified latitude/longitude',
        'distanceunit' : 'Miles or km',
        'operatorid' : 'exact match on a given EVSE operator id (comma separated list)',
        'connectiontypeid' : ' exact match on a given connection type id (comma separated list)',
        'countryid' : 'exact match on a given country id (comma separated list)',
        'levelid' :  'exact match on a given charging level (1-3) id (comma separated list)',
        'minpowerkw' : 'minimum output power in kW (this information is not known for many locations)',
        'usagetypeid' : 'exact match on a given usage type id (comma separated list) ',
        'statustypeid' : ' exact match on a given status type id (comma separated list)',
        'dataproviderid ' : 'exact match on a given data provider id id (comma separated list). Use opendata=true for only OCM provided ("Open") data.',
        'modifiedsince' : 'POIs modified since the given date (UTC) e.g. 2016-09-15T09:30',
        'opendata' : ' true or false. Set to true to include only Open Data licensed content, false to return only non-open licensed data. By default all available data is returned.',
        'includecomments' : ' true or false. Set to true to also include user comments and media items (photos) per charging location. Default = false.',
        'verbose ' : ' true or false. Set to false to get a smaller result set with null items removed. Default =  true.',
        'compact ' : 'true or false. Set to true to remove reference data objects from output (just returns IDs for common reference data such as DataProvider etc). Default = false.',
        'camelcase' : 'true or false. Set to true to get a property names in camelCase format. Default = false',
        'callback' : 'specify the name of the JSONP callback (if required), JSON response type only.'
        }

        if len(kwargs)==0 :

            for key in dic.keys() :
                print(key)

        else :
             
             for k in kwargs:   
                print(dic.get(k))
        

        


'''---------------------------------------------------------------------
   CHARGE MAP CLASS - COLLECT DATA FROM THE API 
   ---------------------------------------------------------------------
'''

class ChargingPoint(object):


    def __init__(self,data):
        self.data = data

    def __repr__(self):

        return "You just collected data from the charging point at {address}, {country}.".format(
            address=self.get_address(), 
            country=self.get_country()
        )
    

    def get_address(self):
        """
        Function collecting charging points address
        """
        
        if "'" in self.data.get("AddressInfo").get("AddressLine1") :
            self.data.get("AddressInfo").get("AddressLine1").replace("'","")

        return self.data.get("AddressInfo").get("AddressLine1")
    
    
    def get_country(self):
        return str(self.data.get("AddressInfo").get("Country").get("Title"))
    
    def get_town(self):
        return str(self.data.get("AddressInfo").get("Town"))

   
    
    
    def get_latitude(self):
        return self.data.get('AddressInfo').get('Latitude')

    def get_longitude(self):
        return self.data.get('AddressInfo').get('Longitude')

    def get_location(self):
        latitude = self.get_latitude()
        longitude = self.get_longitude()
        return [latitude, longitude]

    def get_distance(self):
        return self.data.get('AddressInfo').get('Distance')


    def get_number_of_points(self):
        return "{number} charging points.".format(number = self.data.get('NumberOfPoints'))

    
    def map_display(self):

        location = self.get_location()
        map_folium = folium.Map(location = location, tiles='Stamen Terrain',zoom_start=5)
        folium.Marker(location).add_to(map_folium)

        return map_folium


'''---------------------------------------------------------------------
   CHARGING MAP CLASS - COLLECT DATA ON CHARGING POINTS 
   ---------------------------------------------------------------------
'''

class ChargeMap(object):


    def __init__(self,countrycode = "FR", max_results = 10, json_path = None, data=None):

        url = UrlConstructor()
        url.edit_url(maxresults=max_results, countrycode=countrycode, inplace=True)

        self.get_data(url.url,path = json_path, data=data)
        
    

    def __repr__(self):
        return "{} charging points".format(len(self.data))


    def __iter__(self):
        return iter(self.data)


    def __getitem__(self,key):
        return self.data[key]



    def get_data(self,url=None, path=None, data=None):

        if data is not None :
            pass

        elif url is not None:
            data = requests.get(url).json()

        elif path is not None :
            data = self.load_data(json_path=path)

        else:
            raise Exception("You must provide either an url or a json path")
        
        # Parsing data
        self.parse_data(data)


    def parse_data(self, data):
        
        self.data =[]

        for data_charging_point in data:
            cp = ChargingPoint(data_charging_point)
            self.data.append(cp)


        

    def save_data(self,json_path = "chargemap_{}.json".format(str(pd.to_datetime("today"))[:10])):
        io.save_data_as_json(self.data,json_path)


    def load_data(self,json_path):
        return io.open_json_data(json_path)


    def show(self):
        return self.data



    
    def map_all_spots(self):

        map_folium = folium.Map(tiles='Stamen Terrain',zoom_start=5)
        
        for i,cp in enumerate(self):
            location = cp.get_location()
            html = folium.Html('<b>Address:</b> {0} <br/> <b>Country:</b> {1}'.format(cp.get_address(), cp.get_country()), script=True)
            folium.Marker(location, popup=folium.Popup(html)).add_to(map_folium)

        return map_folium


    def get_lat_long(self, address='address'):
        geolocator=Nominatim()
        loc=geolocator.geocode(address)
    
        return [loc.latitude, loc.longitude] 

    
    def compute_distance(self, address_ini=None, loc_ini=None, address_fin=None, loc_fin=None, method='vincenty'):   

        if loc_ini==None and address_ini==None :
            print("Error: Enter starting coordinates or address")

        if loc_fin==None and address_fin==None :
            print("Error: Enter finish coordinates or address")

        if loc_ini==None :
            loc_ini = self.get_lat_long(address_ini)

        if loc_fin == None :
            loc_fin = self.get_lat_long(address_fin) 

        if method == 'vincenty' :    
            dist = vincenty(loc_ini, loc_fin).kilometers
        
        else : 
            pass
        
        return dist


    def closest_spots(self, address=None, lat=None, lng=None, use_API=True,verbose = 0, **kwargs):

        # Results are more accurate when using lat and longi instead of a real address
        if address == None and (lat==None or lng==None) :
            return('Enter an address or coordinates please.')
        
        elif address != None and (lat==None or lng==None) : 
            lat,lng = self.get_lat_long(address)

        
        if use_API :

            url = UrlConstructor()
            url.edit_url(verbose="false",show_url = verbose, latitude=lat, longitude=lng, distance='', inplace=True, distanceunit='km', **kwargs)
            data = requests.get(url.url).json()
            cm = ChargeMap(data=data)

            for i,cp in enumerate(cm):

                dist = round(cp.get_distance(),2)
                addr,town,country = cp.get_address(),cp.get_town(),cp.get_country()

                town = town +", " if town == "None" else ""
                dist_unit = "m" if dist <= 1.0 else "km"
                dist = dist*1000 if dist <= 1.0 else dist

                if verbose: print('{} : {} {}, {} ; Distance from it = {}{}'.format(i+1, addr, town, country, dist,dist_unit))
            return cm

        else :
            dist_dic = {}
            
            for pt in self.data :
                cp = ChargingPoint()
                cp.get_data(pt)
                loc_fin = cp.get_lat_long()
                loc = ",".join(loc_fin)
                dist_i = self.compute_distance(loc_ini=loc_ini, loc_fin=loc_fin)
                dist_dic[loc]= dist_i

            for i in range(0,10) :
                min_key = min(dist_dic.iteritems(), key=operator.itemgetter(1))[0]
                addr = geolocator.reverse(min_key)
                print("{0} ; Distance from it = {1}".format(addr, min_key))
                del dist_dic[min_key]

        
           


   
   
    