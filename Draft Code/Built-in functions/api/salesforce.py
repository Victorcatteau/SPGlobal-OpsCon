#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This modules contains helper functions to extract data from Salesforce
using the Salesforce REST API. To instantiate the :class:`Salesforce`
class, you'll need OAtuh credentials.

The official Salesforce REST API reference lists the most current tools
used in this module (``sobjects``, ``describe``, ``query``); visit
https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/resources_list.htm
if you want to implement new functionalities or struggle to understand
the query URI syntax fin :meth:`Salesforce.api_call`.

.. module:: salesforce
.. moduleauthor:: Jean-Baptiste Gette <jean-baptiste.gette@ekimetrics.com>

"""

import json
import numpy as np
import pandas as pd
import requests


#==============================================================================
# SALESFORCE WRAPPER
#==============================================================================


class Salesforce():
    """The :class:`Salesforce` class handles the connection to the REST
    API and provides methods to query data and return the results as raw
    JSON or a Pandas DataFrame.

    Args:
        consumer_key (str): The connected app's consumer key.
        consumer_secret (str): The connected app's consumer secret.
        username (str): The email address you use to sign in.
        password_token (str): The concatenation of your user password
            and your secret token.

    """

    def __init__(self, consumer_key, consumer_secret, username, password_token):

        # Run authentication method upon instantiation
        self.oauth_params = {
            "grant_type": "password",
            "client_id": consumer_key, 
            "client_secret": consumer_secret,
            "username": username,
            "password": password_token
        }
        self.access_token, self.instance_url = None, None
        self._authenticate()


    def _authenticate(self):
        """Authentication function. Tries to authenticate using the
        credentials provided at instantiation, and displays different
        messages in case of success or failure.

        """
        r = requests.post("https://login.salesforce.com/services/oauth2/token",
                          params=self.oauth_params)

        self.access_token = r.json().get("access_token")
        self.instance_url = r.json().get("instance_url")

        if self.access_token is not None and self.instance_url is not None:
            print("Authentication succeeded.")
            print("Access Token:", self.access_token)
            print("Instance URL:", self.instance_url)
        else:
            print("Authentication failed")



    #--------------------------------------------------------------------------
    # MAIN WRAPPER

    def api_call(self, action, parameters = {}, method = 'GET', data = {},
                 verbose=False):
        """Helper function to make calls to Salesforce REST API. Copied
        from  https://jereze.com/fr/snippets/authentification-salesforce-rest-api-python
        (link in French). Use this for very particular queries; the most
        basic queries have already been implemented as separate methods.

        Args:
            action (str): URI.
            parameters (dict): URL params.
            method (str): HTTP method (``'GET'``, ``'POST'`` or
                ``'PATCH'``).
            data (dict): data for POST/PATCH methods.
            verbose (bool): When True, displays debugging URLs at each
                call.

        """
        headers = {
            'Content-type': 'application/json',
            'Accept-Encoding': 'gzip',
            'Authorization': 'Bearer %s' % self.access_token
        }

        if method == 'GET':
            r = requests.request(method,self.instance_url+action, 
                                 headers=headers, params=parameters, 
                                 timeout=30)
        elif method in ['POST', 'PATCH']:
            r = requests.request(method, self.instance_url+action,
                                 headers=headers, json=data, params=parameters,
                                 timeout=10)
        else:
            # Other methods not implemented yet
            raise ValueError('Method should be GET or POST or PATCH.')

        if verbose:
            print('Debug: API %s call: %s' % (method, r.url) )

        if r.status_code < 300:
            if method=='PATCH':
                return None
            else:
                return r.json()
        else:
            raise Exception('API error when calling %s : %s' % (r.url, r.content))


    def list_objects(self):
        """Lists all objects (i.e. "tables" that can be called in a SOQL
        query's ``FROM`` clause).

        """
        response = self.api_call('/services/data/v42.0/sobjects/', {})
        return [obj['name'] for obj in response['sobjects']]


    def list_fields(self, obj, sort=True):
        """Lists all fileds in an object (i.e. things that can be called
        in a SOQL query's ``SELECT`` clause).

        Args:
            obj (str): Object name (see :meth:`Salesforce.list_objects`).

        """
        response = self.api_call(f'/services/data/v42.0/sobjects/{obj}/describe', {})
        return [f['name'] for f in response['fields']]


    def query(self, soql, as_dataframe=True):
        """Runs a SOQL query and returns the data as a Pandas DataFrame
        or raw JSON.

        Args:
            soql (str): The SOQL query.
            as_dataframe (bool): Change to False to return raw JSON.

        """
        response = self.api_call('/services/data/v42.0/query/', {'q': soql})

        if as_dataframe:
            return pd.DataFrame(response['records']).drop(columns='attributes')
        else:
            return response
