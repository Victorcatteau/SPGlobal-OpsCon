#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
Scrapping via Tor

Started on the 18/09/2017

- https://www.linkedin.com/pulse/python-how-scrape-websites-anonymously-afsheen-khosravian/
- https://stem.torproject.org/index.html
- https://db-ip.com/
- https://stackoverflow.com/questions/30286293/make-requests-using-python-over-tor
- https://dm295.blogspot.fr/2016/02/tor-ip-changing-and-web-scraping.html
- https://deshmukhsuraj.wordpress.com/2015/03/08/anonymous-web-scraping-using-python-and-tor/



theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""




import pandas as pd
import numpy as np
import time
import datetime
import requests

from stem import Signal
from stem.control import Controller

from ekimetrics.web_scrapping.utils import scrapping





def get_ip(session = None):
    """
    Get the currently used user IP address
    :param session: the Tor session, None to use the actual user session
    :returns: str -- the IP address
    """
    if session is not None:
        return session.get("http://icanhazip.com").text.strip()
    else:
        return requests.get("http://icanhazip.com").text.strip()






def get_tor_session(port = 9050):
    """
    Get the Tor session at a given proxy port
    :param int port: the Tor proxy port
    :returns: the Tor session
    """
    session = requests.session()
    # Tor uses the 9050 port as the default socks port
    session.proxies = {'http':  'socks5://127.0.0.1:{}'.format(port),
                       'https': 'socks5://127.0.0.1:{}'.format(port)}
    return session







 
def set_new_ip(password = "ekimetrics",controller_port = 9051):
    """
    Change the IP using the Tor session
    Use the stem library from Tor official project
    :param str password: the non-hashed password
    :param int controller_port: the port used by the controller, use the default port 9051 by default
    """

    with Controller.from_port(port=controller_port) as controller:
        controller.authenticate(password=password)
        controller.signal(Signal.NEWNYM)








def get_location(ip):
    """
    Get the location and information at a given IP using the website https://db-ip.com
    :param str ip: the user ip address
    :returns: dict -- the info at the user IP address
    """
    page = scrapping("https://db-ip.com/{}".format(ip))
    return {x.find("th").text.strip():x.find("td").text.strip() for x in page.find("tbody").findAll("tr")}