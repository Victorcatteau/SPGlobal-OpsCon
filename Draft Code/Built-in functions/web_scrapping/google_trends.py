#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
GOOGLE TRENDS
Started on the 25/04/2017

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


# Usual libraries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import json
from tqdm import tqdm
import bs4 as bs
from collections import defaultdict

# Custom libraries
# from ekimetrics.web_scrapping import tor
from ekimetrics.web_scrapping.robots import SearchVolumeIoRobot
from ekimetrics.web_scrapping.selenium import SeleniumRobot
from ekimetrics.utils import io
from ekimetrics.visualizations import maps
from ekimetrics.visualizations import charts





#=============================================================================================================================
# GOOGLE TRENDS SCRAPPER
#=============================================================================================================================


def get_correct_geo(geo):
    geo = geo.lower()
    if geo in ["us","usa","united states","united states of america"]:
        return "US"
    elif geo in ["uk","gb","great britain","england","united kingdom"]:
        return "GB"
    elif geo in ["fr","france"]:
        return "FR"
    elif geo in ["de","deutschland","germany","ger"]:
        return "DE"
    elif geo in ["china","cn"]:
        return "CN"
    elif geo in ["japan","jp"]:
        return "JP"
    else:
        return "ALL"



def get_google_trends_data(keyword,geo = "ALL",session = None):
    """
    Get the data from the google trends website

    """

    # Get the correct geo mapping
    geo = get_correct_geo(geo)

    # Define the url 
    url = 'http://www.google.com/trends/fetchComponent?q={0}&cid=TIMESERIES_GRAPH_0&export=3'.format(keyword.replace(" ","%20"))
    if geo != 'ALL':
        url += "&geo={}".format(geo)

    # Get query with requests
    if session is not None:
        text = session.get(url).text
    else:
        text = requests.get(url).text



    # If IP blocked by google the request is the html
    if "<!doctype html>" in text:
        raise ValueError("IP blocked by google")

    # If there are not enough results available
    elif '"status":"error"' in text:
        return pd.DataFrame()

    else:
        pass


    def clean_date(d):
        """
        Helper function to clean a date format from Google Trends
        """
        # parse 'new Date(2004,0,1)' to '2004,0,1'
        d = d[9:-1] 
        
        # Split the date
        year,month,day = d.split(",")
        
        # Convert december month
        if month == "0":
            month = "12"
            year = str(int(year)-1)

        # return the datetime
        return pd.to_datetime("{}-{}-{}".format(year,month,day))


    # Parse the output as json
    json_table = json.loads(text[62:-2].replace('new Date','"new Date').replace(',1)',',1)"'))

    # Parse the csv output to a dataframe
    data = pd.DataFrame([{
            "dates":clean_date(x["c"][0]["v"]),
            "googleindex":x["c"][1]["v"]}
        for x in json_table["table"]["rows"]],columns = ["dates","googleindex"])


    # Drop N/A
    data.dropna(subset = ["googleindex"],inplace = True)

    return data










def get_google_trends_growth(data,rollback = [3,6,12,24,36,60]):
    """
    Analyze the trends time series data to calculate growth index on the Google Index 
    :param pandas.DataFrame data: the data coming from the get_google_trends_data() function
    :param list of int rollback: the number of months on which to calculate the growth
    :returns: dict -- the growth rates by number of months
    """
    if type(rollback) != list: rollback = [rollback]
    growth = {}
    last_date = list(data.dates)[-1]
    last_index = float(list(data.googleindex)[-1])
    for r in rollback:

        # index = list(data.loc[data["dates"] == last_date - pd.DateOffset(months = r)].googleindex)[0]
        index = data.googleindex.iloc[-r-1]
        if index == 0.0:
            growth[r] = 0.0
        else:
            growth[r] = (last_index-index)/index

    if len(growth) == 1:
        return growth[rollback[0]]
    else:
        return growth








#=============================================================================================================================
# GOOGLE TRENDS ROBOT
# Using Tor proxies
#=============================================================================================================================



class GoogleTrendsRobot(object):
    """
    Scrapping robot for Google Trends, that has two methods to work : 
        - works with a Tor Proxy to avoid IP blocking
        - or by using selenium with browser automation
    """
    def __init__(self,method = "selenium",ports = [9050,9051],password = "ekimetrics",driver = None):
        """
        Initialization
        :param list ports: respectively the Tor port and the controller port as integers
        :param str password: the password to access the Tor proxy
        """
        self.data = defaultdict(dict)
        self.method = method
        
        if method == "tor":
            # Initalize parameters
            self.ports = ports
            self.password = password
            self.skipped = []
            self.exceptions = []

            # Create Tor session
            self.session = tor.get_tor_session(port = self.ports[0])
            self.ips = [tor.get_ip(self.session)]
        elif method == "selenium":
            self.robot = GoogleTrendsSeleniumRobot(driver = driver)
        else:
            pass


    #---------------------------------------------------------------------------------
    # OPERATORS


    def __getitem__(self,key):
        """
        Get a keyword data with getitem supercharge operator
        :param str key: the keyword
        :returns: dict -- the data available for this keyword
        """
        return self.data[key]


    def has_data(self,key,geo = "US"):
        return key in self.data and geo in self.data[key] and len(self.data[key][geo]) > 0



    #---------------------------------------------------------------------------------
    # IO


    def serialize(self):
        data = defaultdict(dict)
        for keyword in self.data:
            for geo in self.data[keyword]:
                dict_data = self.data[keyword][geo].copy()
                dict_data["dates"] = dict_data["dates"].map(str)                
                data[keyword][geo] = dict_data.set_index("dates").to_dict()
        return data


    def unserialize(self,data):
        self.data = defaultdict(dict)
        for keyword in data:
            for geo in data[keyword]:
                dict_data = pd.DataFrame(data[keyword][geo])
                dict_data = dict_data.reset_index().rename(columns = {"index":"dates"})
                dict_data["dates"] = dict_data["dates"].map(pd.to_datetime)
                self.data[keyword][geo] = dict_data.copy()



    def save_data(self,file_path = "google_trends.json"):
        """
        Save all the data as json
        Will serialize the dataframes holding the data as dictionaries
        :param str file_path: the json path where to save the file
        :creates: a .json file with the data
        """
        data = self.serialize()
        io.save_data_as_json(data,file_path)





    def load_data(self,file_path = "google_trends.json"):
        """
        Reload the data from a json file
        :param str file_path: the json path where is the file
        :sets: self.data the data placeholder
        """

        data = io.open_json_data(file_path)
        self.unserialize(data)



    def close(self):
        self.robot.close()







    #---------------------------------------------------------------------------------
    # GETTERS

    def get_keywords_with_trends(self,geo,strict = False):
        """
        Get the keywords with the trends already scrapped for a given geography
        Will be used to avoid repeating scrapping for existing keyword data
        :param str geo: the geography to get
        :returns: list of str -- the list of already scrapped keywords for trends
        """
        if not strict:
            keywords = [keyword for keyword in self.data if geo in self.data[keyword]]
        else:
            keywords = [keyword for keyword in self.data if geo in self.data[keyword] and len(self.data[keyword][geo]) > 0]
        return keywords



    def get_keywords_with_volumes(self,geo):
        """
        Get the keywords with the search volume already scrapped for a given geography
        Will be used to avoid repeating scrapping for existing keyword data
        :param str geo: the geography to get
        :returns: list of str -- the list of already scrapped keywords for volumes
        """
        keywords = [keyword for keyword in self.data if geo in self.data[keyword] and "volume" in self.data[keyword][geo].columns]
        return keywords




    def get_keywords_without_volumes(self,geo):
        """
        Get the keywords without the search volume already scrapped for a given geography
        Will be used to avoid repeating scrapping for existing keyword data
        :param str geo: the geography to get
        :returns: list of str -- the list of not already scrapped keywords for volumes
        """
        keywords = [keyword for keyword in self.data if geo in self.data[keyword] and "volume" not in self.data[keyword][geo].columns]
        return keywords




    def get_geos(self):
        """
        Get the geographies already scrapped 
        Will be used to avoid repeating scrapping for existing keyword data
        :returns: list of str -- the list of already scrapped geographies
        """
        geos = list(set([geo for keyword in self.data for geo in self.data[keyword]]))
        return geos








    #---------------------------------------------------------------------------------
    # SETTERS


    def set_new_ip(self):
        """
        Change the IP of the proxy
        """
        # Change the IP using the Stem Tor controller
        tor.set_new_ip(password = self.password,controller_port = self.ports[1])

        # Append the IP to a cache for further use
        self.ips.append(tor.get_ip(self.session))






    #---------------------------------------------------------------------------------
    # MAIN FUNCTIONS

    def get_trends(self,keywords,geo = "US", date_range = None):
        """
        Look for Google Trends data for a list of keywords at a given geography
        The robot will try requesting the Google Trends data every 5-ish seconds
        If blocked it will change its IP to avoid being blocked
        One keyword can be skipped if the robots unsuccessully tried scrapping it 5 times

        :param list of str keywords: a list of keywords to look for
        :param str geo: the geography where you want to look for trends
        :sets: dict -- self.data, a dictionary for all the data retrieved
        :sets: list -- self.exceptions, a list with all the exceptions encountered
        :sets: list -- self.skipped, a list of all the keywords skipped in the process
        """

        # Uniformize the input
        if type(keywords) != list: keywords = [keywords]

        # Uniformize the geo
        geo = get_correct_geo(geo)

        # Iterate over every missing keyword
        print(">> Scrapping google trends")
        for keyword in tqdm([k for k in keywords if k not in self.get_keywords_with_trends(geo = geo)]):


            # USING DIRECT SCRAPPING VIA TOR
            if self.method == "tor":
                # Initialize the tries counter
                tries = 0

                # Tries loop start
                while True:

                    # Scrapping phase
                    try:
                        self.data[keyword][geo] = get_google_trends_data(keyword,geo = geo,session = self.session)
                        break

                    # If exception
                    except Exception as e:

                        print("... Changing IP",end = "")
                        self.set_new_ip()
                        print(" - ok")
                        tries += 1

                        # If error
                        if tries == 5:
                            exception = "Keyword {} - exception {}".format(keyword,e)
                            print(exception)
                            self.exceptions.append(exception)
                            self.skipped.append(keyword)
                            break


                    # Waiting between each request, with a random lapse 
                    time.sleep(5 + random.random())

            # USING SELENIUM AUTOMATION
            elif self.method == "selenium":
                self.data[keyword][geo] = self.robot.get_data(keyword,geo = geo, date_range = date_range)

            else:
                pass



        if self.method == "tor" and len(self.skipped) > 0: print("... Skipped {}".format(len(self.skipped)))





    

    def analyze_geo(self,with_path = True):
        """
        Display for fun where Tor has us located during the process
        Will analyze all the IPs stored in self.ips
        :param bool with_path: plotting the paths between each ip
        :returns: a folium map object
        """

        # Getting all the locations
        coordinates = [tor.get_location(ip) for ip in self.ips]
        
        # Formatting the data for the custom functions to be used
        get_coordinates = lambda x : list(map(float,x["Coordinates"].split(",")))
        coordinates = [{"coordinates":get_coordinates(x),"description":"<b>"+x["Country"]+"</b><br>"+x["State / Region"]}for x in coordinates]

        # Creating the folium map with the custom function
        m = maps.create_marker_map(coordinates,with_path = with_path)

        return m





    def get_volume_data(self,geo = "US",robot = None,volume_data = None):
        if type(geo) != list:

            # Uniformize geo
            geo = get_correct_geo(geo)

            # Get the keywords
            keywords = self.get_keywords_without_volumes(geo = geo)

            # SEARCH VOLUME.IO
            if geo not in ["UK","US"] and volume_data is None:
                for keyword in tqdm(keywords):
                    self.data[keyword][geo]["volume"] = 0.0
            else:
                # Create the search volume robot
                if volume_data is None:
                    if robot is None:
                        robot = SearchVolumeIoRobot()

                    # Get the data
                    self.volume_data = robot.get_volume_data(keywords,geo = geo)
                else:
                    self.volume_data = volume_data


                # APPLY VOLUME FACTOR
                # Iterate over every keyword
                for keyword in tqdm(keywords):
                    if keyword in self.volume_data.index:
                        average_volume = self.volume_data.loc[keyword,"volume"]
                        monthly_data = len(self.data[keyword][geo]["dates"].map(lambda x : x.day).unique()) > 1
                        monthly_data = False # change if weekly
                        rollback = -52 if monthly_data else -12
                        multiplier = 1 if monthly_data else 0.25
                        average_index = self.data[keyword][geo].iloc[-rollback:]["googleindex"].mean()
                        self.data[keyword][geo]["volume"] = self.data[keyword][geo]["googleindex"] * (average_volume*multiplier/average_index)

                return robot

        else:
            robot = None
            for g in geo:
                robot = self.get_volume_data(geo = g,robot = robot)
            robot.close()





    def analyze_growth(self,keyword,geo = "US",**kwargs):
        return get_google_trends_growth(self.data[keyword][geo],**kwargs)





    #---------------------------------------------------------------------------------
    # DATA VISUALIZATIONS


    def show_time_series(self,keywords = None,geo = "US",axis = "volume",with_plotly = True,on_notebook = True):

        # Assertions
        assert axis in ["volume","googleindex"]

        # Get keywords
        if keywords is None: keywords = list(self.data.keys())

        # Iterate over every keyword
        for i,keyword in enumerate(keywords):
            new_data = self.data[keyword][geo][["dates",axis]].set_index("dates").rename(columns = {axis:keyword})
            if i == 0:
                data = new_data
            else:
                data = data.join(new_data)

        fig = charts.plot_line_chart(data,with_plotly = with_plotly,on_notebook = on_notebook)
        return fig





    def show_country_mapping(self,keywords = None,display = False,with_plotly = True,on_notebook = True,geo = ["US","FR"],rollback = 3,**kwargs):
        
        geo1,geo2 = geo

        # Get the keywords
        if keywords is None:
            keywords_1 = self.get_keywords_with_trends(geo = geo1,strict = True)
            keywords_2 = self.get_keywords_with_trends(geo = geo2,strict = True)
            keywords = list(set(keywords_1).intersection(set(keywords_2)))


        # Iterate over the keywords
        data = []
        for keyword in tqdm(keywords):
            data_keyword = {"keyword":keyword}
            data_keyword[geo1] = self.analyze_growth(keyword,geo1,rollback = rollback)
            data_keyword[geo2] = self.analyze_growth(keyword,geo2,rollback = rollback)
            data.append(data_keyword)

        # Return result as dataframe
        data = pd.DataFrame(data,columns = ["keyword",geo1,geo2])

        if display:
            # Plot visualization
            fig = charts.plot_scatter_chart(data,geo1,geo2,"keyword",color = "color",with_plotly = with_plotly,on_notebook = on_notebook,**kwargs)

            return fig
        else:
            return data


















#=============================================================================================================================
# GOOGLE TRENDS SELENIUM ROBOT
# Google trends simulation
# Developped for ODDO BHF project
#=============================================================================================================================



class GoogleTrendsSeleniumRobot(SeleniumRobot):
    """docstring for GoogleTrendsSeleniumRobot"""
    def __init__(self,driver_path = "C:/git/chromedriver.exe",driver = None):
        self.url = "https://trends.google.fr/trends/explore"
        super().__init__(driver_path = driver_path,driver = driver)


    def build_url(self,query,geo = "ALL", date_range = None):
        if date_range is None:
            date_range = "date=today%205-y"
        elif date_range == "all":
            date_range = "date=all" 
        else:
            print("Please specify date range")

        url = "https://trends.google.fr/trends/explore?"+date_range
        url += "&q={}".format(query.replace(" ","%20"))
        if geo != "ALL":
            url += "&geo={}".format(geo)
        return url


    def get_data(self,query,geo = "ALL", date_range = None):
        self.connect(self.build_url(query = query,geo = geo, date_range = date_range))
        self.wait(2+random.random(),verbose = 0)
        page = self.get_html()
        return self.parse_table(page)


    def parse_table(self,page):
        page = str(page)
        start = page.find("<table>")
        end = page.find("</table>")+len("</table>")
        page = bs.BeautifulSoup(page[start:end],"html5lib")
        rows = page.findAll("tr")
        rows = [{"dates":date.text,"googleindex":float(index.text)} for date,index in [row.findAll("td") for row in rows[1:]]]
        data = pd.DataFrame(rows,columns = ["dates","googleindex"])
        data["dates"] = data["dates"].map(self.convert_date_to_datetime)
        return data



    def convert_date_to_datetime(self,date):
        day,month,year = date.split(" ")
        day = date[1:3].strip()
        year = year[:4]
        if month in ["janv."]:
            month = "01"
        elif month in ["févr."]:
            month = "02"
        elif month in ["mars"]:
            month = "03"
        elif month in ["avr."]:
            month = "04"
        elif month in ["mai"]:
            month = "05"
        elif month in ["juin"]:
            month = "06"
        elif month in ["juil."]:
            month = "07"
        elif month in ["août"]:
            month = "08"
        elif month in ["sept."]:
            month = "09"
        elif month in ["oct."]:
            month = "10"
        elif month in ["nov."]:
            month = "11"
        elif month in ["déc."]:
            month = "12"
        if len(day) == 1:
            day = "0" + str(day)
        date = "{}-{}-{}".format(year,month,day)
        # return pd.to_datetime(datetime.datetime(int(year),int(month),1))
        return pd.to_datetime(date)




