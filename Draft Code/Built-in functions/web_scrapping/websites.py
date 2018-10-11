#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
OCTOPUS

Started on the 16/05/2017



Requirements
- BeautifulSoup : bs4
- html5lib
- nltk

https://www.scrapehero.com/how-to-prevent-getting-blacklisted-while-scraping/


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""




import pandas as pd
import numpy as np
import time
import datetime

from collections import Counter

from ekimetrics.utils.time import Timer
from ekimetrics.web_scrapping.utils import *





#=============================================================================================================================
# GOOGLE SEARCH
#=============================================================================================================================


class GoogleScrappingError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)




class GoogleSearch(Page):
    def __init__(self,query,compute = True,domain = "fr",**kwargs):


        self.query = query
        self.domain = domain
        url = self.build_url(query)


        super().__init__(url = url,compute = False,**kwargs)

        self.check_robot()


        if compute:
            self.find_data()


        self._print(">> Scrapping finished with {} exceptions".format(len(self.exceptions)))



    def build_url(self,query):
        return "https://www.google.{}/search?q={}".format(self.domain,query.replace(" ","+"))


    def check_robot(self):
        if "CAPTCHA" in self.page.get_text():
            raise GoogleScrappingError("Google has blocked your IP address")



    def find_data(self):
        self.find_results()



    def find_results(self):
        try:
            self.results = self.page.findAll("div",attrs = {"class":"g"})

        except Exception as e:
            self.results = []
            self.exceptions.append("results")


        def parse_result(result):
            meta = {}
            
            # URL
            try:
                url = result.find("a").attrs["href"].split("&sa")[0].replace("/url?q=","")
                meta["url"] = url
                
                # TEXT
                meta["text"] = clean_html_text(result.get_text())
                return meta
            except Exception as e:
                return None


        self.results = [parse_result(x) for x in self.results]
        self.results = [x for x in self.results if x is not None]


















#=============================================================================================================================
# SOCIETE.COM
#=============================================================================================================================




class SocieteCom(Page):
    def __init__(self,page_id,compute = True,**kwargs):

        if page_id.startswith("http"):
            url = page_id
        else:
            base_url = "http://www.societe.com/societe/"
            url = base_url + page_id


        super().__init__(url = url,compute = False,**kwargs)

        if compute:
            self.find_data()


        self._print(">> Scrapping finished with {} exceptions".format(len(self.exceptions)))










    #-----------------------------------------------------------------------
    # FIND DATA


    def find_data(self):
        """
        Call multiple extraction function
        """

        self.find_description()
        self.find_legal_information()
        self.find_management()
        self.find_judgments()





    def find_legal_information(self):
        """
        Scrape, find and extract the table of legal information as a json of data
        Creates: self.legal_information
        """


        # EXTRACT TABLE
        try:
            table = self.page.find("div",attrs = {"id":"renseignement"}).findAll("tr")
        except Exception as e:
            table = None
            self.exceptions.append("legal information")


        # PARSE THE TABLE AS A DICTIONARY
        def extract_meta(table):
            if table is None: 
                return None
            else:
                meta = {}
                for x in table:
                    x = x.findAll("td")
                    key = x[0].get_text().lower()
                    if key not in ['n° de tva intracommunautaire','téléphone']:
                        meta[key] = x[1].get_text().replace("\n","").strip()
                return meta


        self.legal_information = extract_meta(table)











    def find_management(self):
        """
        Scrape, find and extract the management information as a list of people
        Creates: self.management
        """

        try:
            from ekimetrics.nlp import fuzzy_matching
            
            # MANAGEMENT EXTRACTION
            self.management = [x for x in [x.get_text().strip() for x in self.page.find("div",attrs = {"id":"dir"}).findAll("td")] if x.startswith("M ") or x.startswith("MME ")]

            # FUZZY REMOVING
            self.management = fuzzy_matching.remove_duplicates(self.management)
        except Exception as e:
            self.management = None
            self.exceptions.append("management")








    def find_description(self):
        """
        Scrape, find and extract the company presentation
        Creates: self.presentation
        """

        # HEADER
        try:
            self.header = clean_html_text(self.page.find("div",attrs = {"id":"identitetext"}).find("address").get_text())
        except Exception as e:
            self.header = None
            self.exceptions.append("header")


        # PRESENTATION
        try:
            self.presentation = clean_html_text(self.page.find("div",attrs = {"id":"synthese"}).get_text())
        except Exception as e:
            self.presentation = None
            self.exceptions.append("presentation")





    def find_judgments(self):
        """
        Scrape, find and extract the company judgments data
        Does not raise an exception if not found
        Creates: self.judgments
        """

        try:
            judgments = self.page.find("div",attrs = {"id":"jugement"})
            self.judgments = [{x.split(":")[0].strip():x.split(":")[1].strip() for x in judgment.get_text().strip().split("\n")} for judgment in judgments.findAll("th")]

        except Exception as e:
            self.judgments = []




    #-----------------------------------------------------------------------
    # GETTERS

    def get_in_meta(self,field):
        return self.legal_information[field] if field in self.legal_information else None

    def get_revenues(self):
        pass


    def get_number_of_employees(self):
        meta = self.get_in_meta("tranche d'effectif")
        if meta is not None and meta.startswith("0 salarié"):
            return 0
        else:
            return 1        


    def get_creation_date(self):
        creation_date = self.get_in_meta("date création entreprise")
        if creation_date is not None:
            return pd.to_datetime(creation_date,dayfirst = True)
        else:
            return None




    def get_age(self):
        creation_date = self.get_creation_date()
        if creation_date is not None:
            age = int((pd.to_datetime(datetime.datetime.now()) - creation_date).to_timedelta64().astype("timedelta64[Y]"))
            return age
        else:
            return None



    def get_address(self):
        address = self.get_in_meta("adresse")
        if address is None:
            return None
        else:
            address = address.split(", ")
            
            if len(address) > 1:
                return address[1]
            else:
                return address[0]





    def get_capital(self):
        capital = self.get_in_meta("capital social")
        if capital is not None:
            capital = capital.replace("EURO","").strip()
            try:
                capital = int(capital)
            except Exception as e:
                capital = None
        return capital



    def get_executive_manager(self):
        return self.management[0] if self.management is not None else None


    def get_management(self):
        return self.management if self.management is not None else []







    #-----------------------------------------------------------------------
    # FRAUD DETECTION




    def is_liquidated(self):

        # IN JUDGMENTS
        if len(self.judgments) > 0 and "liquidation" in [x["Type"].lower() for x in self.judgments if "Type" in x]:
            return True

        # IN HEADER
        elif "liquidation" in self.header.lower():
            return True

        else:
            return False




    def is_family_managed(self):
        if self.management is not None:
            last_names = [x for x in [x.split(" ") for x in self.management] if len(x) >= 3]
            if len(last_names) > 1:
                names = Counter([x[-1] for x in last_names])
                names = [x[0] for x in names.items() if x[1] > 1]

                if len(names) >= 1:
                    self.family_members = [x for x in self.management if x.split(" ")[2] in names]
                    return True
                else:
                    self.family_members = None
                    return False
            else:
                self.family_members = None
                return False
        else:
            self.family_members = None
            return False





    def has_few_employees(self):
        number = self.get_number_of_employees()
        if number < 1:
            return True
        else:
            return False



    def is_deregistered(self):

        # IN LEGAL INFORMATION
        if self.legal_information is not None:

            # Status
            if "statut" in self.legal_information:
                status = self.legal_information["statut"]

                if "radiée" in status.lower() or "fermée" in status.lower():
                    return True

            # Judgment
            if "jugement" in self.legal_information:
                judgment = self.legal_information["jugement"]

                if "clôturée" in judgment.lower():
                    return True


        # IN HEADER
        elif "radiée" in self.header.lower() or "clôturée" in self.header.lower() or "fermée" in status.lower():
            return True

        else:
            return False







#=============================================================================================================================
# EY NEWS
# Done for POC PwC Tax Luxembourg
# Started on the 2017/07/18
#=============================================================================================================================



class EY_Feed(Page):
    def __init__(self,html = None,**kwargs):

        # base_url = "http://www.ey.com/gl/en/services/tax/international-tax/tax-alert-library/"

        if html is not None:
            super().__init__(html = html,compute = False,**kwargs)
            self.get_all_links()


    def get_all_links(self):
        self.links = self.page.find("ul",attrs = {"class":"default-ul"}).findAll("li")
        self.links = [self.parse_link(link) for link in self.links]



    def parse_link(self,x):
        meta = {}
        meta["country"] = x.attrs["data-country"]
        meta["topic"] = x.attrs["data-topic"]
        meta["date"] = x.attrs["data-releasedate"]
        link = x.find("a")
        meta["url"] = link.attrs["href"]
        meta["title"] = link.text
        return meta







class EY_News(Page):
    def __init__(self,url = None,meta = None,**kwargs):

        self.verbose = False


        if meta is not None:
            self.meta = meta
            for key in meta:
                setattr(self,key,meta[key])
        elif url is not None:
            self.url = url
        else:
            raise ValueError("No idea provided")
        
        super().__init__(url = self.url,compute = False,verbose = 0,**kwargs)

        self.get_data()


    def get_data(self):
        self.text = self.page.find("div",attrs = {"class":"maincolumn"}).find("section").get_text().replace("\t","")


    def extract_meta(self):
        meta = {
            "text": self.text,
            **self.meta
        }
        return meta
