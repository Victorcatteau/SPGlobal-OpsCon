import numpy as np
import pandas as pd
from dateutil import parser
import datetime
import requests
import json
import pickle



class OECD_Data(object):
	""" scrapping of OECD adatabase
		To understand the json structure, see : https://github.com/sdmx-twg/sdmx-json/blob/master/data-message/docs/1-sdmx-json-field-guide.md
	"""
	def __init__(self,dataset_identifier,dimensions, startTime, endTime):
		"""
		dimensions corresponds to the dimensions : subject, country and frequency
		see https://data.oecd.org/api/sdmx-json-documentation/
		"""
		self.url = 'http://stats.oecd.org/SDMX-JSON/data/' + dataset_identifier + '/' + dimensions + '/' + 'all'
		print(self.url)
		self.params = {"startTime" : startTime, "endTime" : endTime}
		resp = requests.get(url=self.url, params=self.params)
		self.json = json.loads(resp.text)
		self.dates = self.get_dates_dict()
		self.dim_dicts = self.get_dimensions_dict()
		self.dim_codes = self.get_dimensions_codes()
		self.data = self.json_to_dataframes()

	def json_to_dataframe(self, index):
		"""
		transforms the observations in the json to a dataset and fetches dataset dimensions
		"""
		observations = self.json['dataSets'][0]['series'][index]['observations']
		observations = pd.DataFrame([observations.keys(), list(map(lambda x:x[0], observations.values()))]).transpose()
		observations.columns = ["index","value"]
		observations["date"] = observations["index"].apply(lambda x:parser.parse(self.dates[int(x)]))

		dataset_info = self.get_dataset_info(index)
		dataset_with_info = {'info':dataset_info, 'dataset':observations}

		return(dataset_with_info)

	def json_to_dataframes(self):
		""" 
	    returns dict of dataframes from json and includes the dimensions of the dataset
	    """
		indexes = list(self.json['dataSets'][0]['series'].keys())
		res = dict()
		for i in range(0,len(indexes)):
			index = indexes[i]
			dataset_with_info = self.json_to_dataframe(index)
			res[i] = dataset_with_info
		return(res)

	def get_dates_dict(self):
		""" 
		extracts the dictionary of dates used in the json
		"""
		dates = list(map(lambda x: x['id'],self.json['structure']['dimensions']['observation'][0]['values']))
		dates_dict = dict(zip(range(0,len(dates)),dates))
		return dates_dict

	def get_dimensions_dict(self):
		"""
		extracts the dictionary of dimensions used in the json
		"""
		dim_dicts = dict()
		for i in range(0,len(self.json['structure']['dimensions']['series'])):
			dat = self.json['structure']['dimensions']['series'][i]
			names = [x['name'] for x in dat['values']]
			dim_dicts[dat['keyPosition']] = {'name' : dat['name'], 'dictionary':dict(zip(range(0,len(names)),names))} 
		return(dim_dicts)

	def get_dimensions_codes(self):
		"""
		extracts the dictionary of dimensions used in the json
		"""
		dim_dicts = dict()
		for i in range(0,len(self.json['structure']['dimensions']['series'])):
			dat = self.json['structure']['dimensions']['series'][i]
			codes = [x['id'] for x in dat['values']]
			names = [x['name'] for x in dat['values']]
			dim_dicts[dat['keyPosition']] = {'name' : dat['name'], 'dictionary':dict(zip(codes,names))} 
		return(dim_dicts)


	def get_dataset_info(self, index):
		"""
		fetches the dimensions corresponding to a dataframe
		"""
		res = dict()
		indexes = index.split(':')
		for ind in range(0,len(indexes)):
		    index_value = indexes[ind]
		    dim_dict = self.dim_dicts[ind]
		    name_dim = dim_dict['name']
		    value_dim = dim_dict['dictionary'][int(index_value)]
		    res[name_dim] = value_dim
		return(res)
		
	def save_to_pickle(self, file, file_name):
		pickle_out = open("C:/Users/mmonges/Documents/Python/oecd/dimensions_dictionaries/" + file_name + ".pickle","wb")
		pickle.dump(file, pickle_out)
		pickle_out.close()

	def plot_data(self, index):
		dat = self.data[index]['dataset']

	

def get_dimension_dictionary(dataset_identifier):
		pickle_in = open("C:/Users/mmonges/Documents/Python/oecd/dimensions_dictionaries/dimension_dictionary_" + dataset_identifier + ".pickle","rb")
		return(pickle.load(pickle_in))