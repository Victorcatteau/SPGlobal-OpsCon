#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
WEB SCRAPPING

Started on the 02/02/2017



Requirements
- BeautifulSoup : bs4
- html5lib
- nltk

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""




import bs4 as bs
import pandas as pd
import numpy as np
import requests
import time
import re

from ekimetrics.utils import io


#=============================================================================================================================
# SCRAPPING BASE FUNCTION
#=============================================================================================================================



def scrapping(url,timeout = 15,session = None):
    """
    Scrapping function that takes an URL to get a beautiful soup object of the source code of the URL
    A timeout is set by default at 15 seconds

    Returns : a beautiful soup object
    """
    if session is not None:
        html = session.get(url,timeout = timeout).content
    else:     
        html = requests.get(url,timeout = timeout).content
    return parsing(html)



def parsing(html):
    return bs.BeautifulSoup(html,'lxml')




def find_proxy(n = 10):
    """"
    Find proxies 
    """

    import asyncio
    from proxybroker import Broker

    all_proxies = []
    async def show(proxies):
        while True:
            proxy = await proxies.get()
            if proxy is None: break
            all_proxies.append(proxy)
            print('Found proxy: %s' % proxy)

    proxies = asyncio.Queue()
    broker = Broker(proxies)
    tasks = asyncio.gather(
        broker.find(types=['HTTP', 'HTTPS'], limit=10),
        show(proxies))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(tasks)
    return all_proxies

        





#=============================================================================================================================
# PAGE CLASS
#=============================================================================================================================



class Page(object):
    def __init__(self,url = None,html = None,verbose = 1,compute = True):

        # INITIALIZATION
        self.verbose = verbose
        self.exceptions = []


        if url is not None:
            self._print(">> Scrapping page at url {}".format(url))
            self.url = url
            self.parse_url(self.url)

            # PARSING
            self._print("... Launch the scrapper")
            self.parse(url = self.url)

        elif html is not None:
            self._print(">> Parsing the html provided")
            self.html = html
            self.parse(html = self.html)


        # FINDING DATA
        
        if compute: self.find_data()


        if self.__class__.__name__ == "Page":
            self._print(">> Scrapping finished with {} exceptions".format(len(self.exceptions)))



    #-----------------------------------------------------------------------
    # REPRESENTATIONS

    def __repr__(self):
        str_repr = self.url
        str_repr += ""
        return str_repr

    def __str__(self):
        return self.__repr__()

    def _print(self,string):
        if self.verbose:
            print(string)



    #-----------------------------------------------------------------------
    # PARSE THE SOURCE CODE

    def parse_url(self,url):
        url = url.split("?")
        if len(url) > 1:
            self.base_url = url[0]
            self.parameters_url = url[1]
        else:
            self.base_url = url[0]
            self.parameters_url = None


        if self.base_url[-1] != "/":
            self.base_url += "/"



    def parse(self,url = None,html = None):
        if url is not None:
            self.page = scrapping(url)
        elif html is not None:
            self.page = parsing(html)


        # REMOVE THE SCRIPT AND STYLE SCRIPTS FROM THE DATA
        for script in self.page(["script", "style"]):
            script.extract()    # rip it out


    #-----------------------------------------------------------------------
    # FIND DATA


    def find_data(self):
        self._print("... Finding meta data")
        self.find_meta()

        self._print("... Finding text")
        self.find_text()

        self._print("... Finding images")
        self.find_images()

        self._print("... Finding urls")
        self.find_urls()



    #-----------------------------------------------------------------------
    # FIND META


    def find_meta(self):
        self.find_title()
        self.find_description()


    def find_title(self):
        try:
            self.title = self.page.find("title").text
        except Exception as e:
            self.exceptions.append("title")
            self.title = "No title"

        


    def find_description(self):
        try:
            meta = self.page.find("head").find("meta",attrs = {"name":"description"})

            if meta is None:
                metas = [x for x in self.page.find("head").findAll("meta") if "property" in x.attrs and "description" in x["property"]]
                if len(metas) >= 1:
                    self.description = metas[0]["content"]
                else:
                    self.description = "No description"
            else:
                self.description = meta["content"]

        except Exception as e:
            self.exceptions.append("description")
            self.description = "No description"



    #-----------------------------------------------------------------------
    # FIND TEXT


    def find_text(self):
        try:
            self.full_text = " ".join([x.strip() for x in self.page.get_text().strip().split("\n") if x.strip() != ""])
            self.full_text = re.sub('<[^<]+?>', '', self.full_text).strip()

        except Exception as e:
            self.exceptions.append("text")
            self.full_text = None





    #-----------------------------------------------------------------------
    # FIND IMAGES


    def find_images(self):
        try:
            images = self.page.findAll("img")
            if len(images) > 0:
                self.images = [x["src"] for x in self.page.findAll("img")]
            else:
                self.images = []
        except Exception as e:
            self.exceptions.append("images")
            self.images = []





    #-----------------------------------------------------------------------
    # FIND URLS


    def find_urls(self):
        try:
            candidates = [x.attrs["href"] for x in self.page.findAll("a") if "href" in x.attrs]

            # FINDING DIRECT LINKS ON THE WEBSITE
            self.direct_links = list(filter(lambda x : x != "" , set([self.base_url + (x[1:] if x.startswith("/") else x) for x in candidates if x.startswith("/")])))

            # FINDING OTHER LINKS
            self.links = list(filter(lambda x : x != "" , set([x for x in candidates if x.startswith("http")])))

            # FINDING MAILS
            self.mails = list(filter(lambda x : x != "" , set([x.replace("mailto:","") for x in candidates if x.startswith("mailto:")])))

            # FINDING TELEPHONE NUMBER
            self.tels = list(filter(lambda x : x != "" , set([x.replace("tel:","").strip() for x in candidates if x.startswith("tel:")])))


        except Exception as e:
            self.exceptions.append("urls")
            self.direct_links = []
            self.links = []
            self.mails = []
            self.tels = []



    def get_data(self):
        data = {}
        data["text"] = self.full_text
        data["url"] = self.url
        data["links"] = self.direct_links
        data["title"] = self.title
        data["description"] = self.description
        data["images"] = self.images
        data["external_links"] = self.links
        data["mails"] = self.mails
        data["tels"] = self.tels
        return data



    def save_data_in_mongodb(self,database,collection,host,port = 27017):
        
        # Get the data as a dictionary to be suited for mongodb NoSQL format
        data = [self.get_data()]

        # Save the data in Mongo using utils function
        io.save_data_in_collection(data,database,collection,host = host,port = port)















#=============================================================================================================================
# VOCABULARY
#=============================================================================================================================




html_characters = ["script","function","jquery","langify","js","return","document","true","false","shopifyapi",
                   "id","value","length","userlanguage","callback","com","view",
                ]






#=============================================================================================================================
# CLEANSING FUNCTION
#=============================================================================================================================


def clean_html_text(text):
    text = text.strip()
    text = "\n".join([x.strip() for x in text.split("\n") if x.strip() != ""])
    return text








#=============================================================================================================================
# WEBSITE CLASS
#=============================================================================================================================



class Website(object):
    def __init__(self,url,max_recursivity = 10,verbose = 1):


        # INITIALIZATION
        self.verbose = verbose
        self.max_recursivity = max_recursivity
        self.url = url
        self.full_text = ""
        self.mails = []
        self.tels = []
        self.direct_links = []
        self.links_yet_to_explore = []
        self.links = []
        self.exceptions = []
        self.description = ""
        self.texts = []

        # FINDING ALL THE DATA
        self.find_all_data()


    def __repr__(self):
        return self.url

    def __str__(self):
        return self.__repr__()


    def print(self,message,**kwargs):
        if self.verbose: print(message,**kwargs)


    def find_all_data(self):

        # Scrap the first page and store the data
        first_page = Page(self.url,verbose = 0)
        self.title = first_page.title
        self.description = first_page.description
        self.append_page_results(first_page)
        i = 0

        # Loop over the links already discovered until either there is no more links or max_recursivity is reached
        while len(self.links_yet_to_explore) > 0 and (i < self.max_recursivity if self.max_recursivity is not None else True):
            # New url to explore
            new_url = self.links_yet_to_explore[0]
            self.print("\r[{}/{}] urls to explore after url {}".format(i+1,len(self.direct_links),new_url),end = "")

            # Pop the first element
            self.links_yet_to_explore = self.links_yet_to_explore[1:]

            # Scrape the new page
            page = Page(new_url,verbose = 0)

            # Append the results
            self.append_page_results(page)

            # Increment the counter
            i += 1






    def append_page_results(self,page):

        self.texts.append(page.full_text)

        if page.full_text is not None:
            self.full_text += " " + page.full_text

        self.mails += [x for x in page.mails if x not in self.mails]
        self.tels += [x for x in page.tels if x not in self.tels]

        new_links = [x for x in page.direct_links if x not in self.direct_links]
        self.direct_links += new_links
        self.links_yet_to_explore += new_links
        self.links += [x for x in page.links if x not in self.links]
        self.exceptions += page.exceptions



    def get_info(self):
        data = {}
        data["url"] = self.url
        data["text"] = self.full_text
        data["links"] = self.direct_links
        data["external_links"] = self.links
        data["title"] = self.title
        data["description"] = self.description
        data["mails"] = self.mails
        data["tels"] = self.tels
        return data













#=============================================================================================================================
# SCRAPPING A LIST OF URLS
#=============================================================================================================================


def scrap_list_of_urls(urls,max_recursivity = 10):
    data = []
    faulty_urls = []
    print(">> Launching scrapping on {} urls".format(len(urls)))
    for url in tqdm(urls):
        try:
            website = Website(url,max_recursivity = max_recursivity,verbose = 0)
            data.append(website.get_info())
        except:
            faulty_urls.append(url)

    print("... {} ({}%) successfully scrapped".format(len(urls),np.round((len(urls)-len(faulty_urls))/float(len(urls)))))
    return data

