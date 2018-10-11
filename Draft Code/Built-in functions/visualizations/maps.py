#!/usr/bin/env python
# -*- coding: utf-8 -*- 


__author__ = "Theo"



"""--------------------------------------------------------------------
MAPS
Started on the 25/09/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


# USUAL LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import geopy



# FOLIUM
import folium





def create_marker_map(locations,with_path = False,zoom_start = 3):
    """
    Create a folium map with markers
    locations must be either a list of tuple of coordinates,
    or a list of dict {"lat","lng","label"}
    :param locations: the list of markers to add to the map
    :param bool with_path: traces the path between each marker in the order
    :param int zoom_start: the initial zoom, 1 being the widest zoom
    :returns: a folium map object
    """

    # Convert the input to the same format
    if type(locations[0]) != dict:
        locations = [{"lat":loc[0],"lng":loc[1],"label":str(i).replace("'","")} for i,loc in enumerate(locations)]


    get_coord = lambda location: (location["lat"],location["lng"])

    # Create the map
    m = folium.Map(location = get_coord(locations[0]),zoom_start=zoom_start)

    # Add the markers
    for i,marker in enumerate(locations[:20]):
        folium.Marker(get_coord(marker), popup=marker["label"]).add_to(m)

        # Add the path
        if i+1 < len(locations) and with_path:
            from_point = get_coord(locations[i])
            to_point = get_coord(locations[i+1])
            line=folium.PolyLine(locations=[from_point,to_point],weight=1,color = 'blue')
            m.add_child(line)

    return m




def create_map_from_any_location(loc):
    geocoder = geopy.geocoders.Nominatim()
    position = geocoder.geocode(loc)
    coord = position.latitude,position.longitude
    m = folium.Map(location = coord,zoom_start=4)
    folium.Marker(coord).add_to(m)
    return m