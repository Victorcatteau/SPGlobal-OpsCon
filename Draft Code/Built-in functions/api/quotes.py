#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#------- Basic imports -----------------
import requests
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------- Sklearn imports -----------------
from sklearn.preprocessing import MinMaxScaler

#------- Data API imports -----------------
import quandl as qd
# import pandas_datareader.data as web
# import pandas_datareader as pdr



#------- Plot imports -----------------
from bokeh.palettes import Spectral5
from bokeh.layouts import gridplot, layout, widgetbox,row
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import GlyphRenderer, Span
from bokeh.models import ColumnDataSource, HoverTool, Div, CustomJS, BoxAnnotation, NumeralTickFormatter, Legend,Label
from bokeh.models.widgets import Panel, Tabs, Slider, Select, TextInput, Button
from bokeh.charts import Histogram, Bar, BoxPlot
from bokeh.charts.attributes import cat
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, CrosshairTool, BoxSelectTool, WheelZoomTool
from bokeh.models.glyphs import Ray


import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

figsize = (12,4)
tools = 'reset,xbox_zoom,box_zoom,wheel_zoom,pan,hover'

import warnings
warnings.filterwarnings("ignore")








#==============================================================================================================
# Generic functions
#==============================================================================================================

def create_dataset(data, look_back = 1):
    dataset = pd.DataFrame({})
    dataset['Y'] = data

    for i in range(look_back):
        dataset['X(t-{0})'.format(look_back-i)] = data.shift(look_back-i)

    return dataset.iloc[look_back:]
    


#==============================================================================================================
# QUOTES DATA
#==============================================================================================================


class Quotes(object):
    def __init__(self,code = "AAPL", quandl = True, load_pkl = None):
        self.code = code
        
        if load_pkl is not None:
            self.data = pd.read_pickle(load_pkl)
        else:
            if quandl:
                self.data = qd.get(code, authtoken="pH6s25ZKmSsZjSVv7odm")
            else:
                self.data = pdr.get_data_yahoo(code)


        # DEFAULT PARAMETERS
        self.set_plot_size(900,300)


    #----------------------------------------------------------------------------------------
    # GETTERS

    def get_start_end_dates(self,start = None,end = None):
        if start is None: start = min(self.data.index)
        if end is None: end = max(self.data.index)
        return start,end

    def filter_on_date(self,start = None,end = None):
        start,end = self.get_start_end_dates(start,end)
        return self.data.loc[(self.data.index >= start) & (self.data.index <= end)]


    #----------------------------------------------------------------------------------------
    # SETTERS
    def set_plot_size(self,width,height):
        self.plot_width = width
        self.plot_height = height



    #----------------------------------------------------------------------------------------
    # DEFINE VARIABLE OF INTEREST
    def set_target(self, target = "Adj Close"):
        self.data['target'] = self.data[target]



    #----------------------------------------------------------------------------------------
    # CALCULATION

    def compute_deltas(self,field = "target", periods = 5):
        self.data["{0}_days_{1}_delta".format(periods, field)] = -(self.data[field].diff(- periods)/self.data[field])


    def compute_logreturn(self,field = "target", periods = 5):
        self.data["{0}_days_{1}_logreturn".format(periods, field)] = np.log(self.data[field]).diff(-periods)


    def compute_variance(self,field = "target", periods = 5):
        self.data["{0}_days_{1}_variance".format(periods, field)] = self.data[field].rolling(periods).var().shift(-periods+1)


    def compute_ma(self,field = "target", periods = 5, center = True):
        self.data["{0}_days_{1}_ma".format(periods, field)] = pd.rolling_mean(self.data[field], periods, center = center)


    def create_dataset(self, field = "target", look_back = 1):
        self.dataset = create_dataset(self.data[field], look_back = look_back)


    def get_clusters_metrics(self, reuters = None, start = None, end = None, field = "target", periods = 5):
        if '{0}_days_{1}_delta'.format(periods, field) not in self.data.columns:
            self.compute_deltas(field = field, periods = periods)
        if '{0}_days_{1}_variance'.format(periods, field) not in self.data.columns:
            self.compute_variance(field = field, periods = periods)

        data = self.filter_on_date(start,end)[[field, '{0}_days_{1}_delta'.format(periods, field), '{0}_days_{1}_variance'.format(periods, field)]]
        data = data.reset_index().rename(columns = {'Date': 'date'})

        reuters_data = reuters.get_count(start = start, end = end, by_cluster = True, groupby = False)
        clusters = reuters.clusters

        merged = pd.merge(reuters_data, data, on = 'date')

        results = merged.groupby("cluster").agg({'{0}_days_{1}_delta'.format(periods, field):{
                                                    "count":"count",
                                                    "average delta (%)":lambda x : np.round(x.mean() *100,3),
                                                    "std delta":lambda x : np.round(x.std(),3)},
                                                '{0}_days_{1}_variance'.format(periods, field):{
                                                    "average variance":lambda x : np.round(x.mean(),3),
                                                    "std variance":lambda x : np.round(x.std(),3)}}).reset_index()

        results.columns = [results.columns.get_level_values(0)[0]] + list(results.columns.get_level_values(1))[1:]

        results = pd.merge(results,clusters[["cluster","top_10_words"]],on = "cluster").sort_values("average delta (%)",ascending = False)

        return results




    #----------------------------------------------------------------------------------------
    # PLOTTING

    def plot(self,field = "target",reuters = None,start = None,end = None, bokeh = True,file = False, clusters = None):

        #--------------------------------------------------------------
        # PLOT WITH SEABORN
        if not bokeh:
            data = self.filter_on_date(start,end)
            plt.figure(figsize=figsize)
            data[field].plot(lw = 0.7)

            def create_palette(max_count):
                value = 1./max_count
                palette = [np.array([1.0,i*value,i*value,1.0]) for i in reversed(range(max_count))]
                return palette
    

            if reuters is not None:
                reuters_data = reuters.get_count(start,end)
                max_count = reuters_data["count"].max()
                palette = create_palette(max_count)
                for i in range(len(reuters_data)):
                    date = reuters_data.iloc[i].loc["date"]
                    count = reuters_data.iloc[i].loc["count"]
                    plt.axvline(x = date,lw = 0.6,c = palette[count-1])
            plt.show()

        #--------------------------------------------------------------
        # PLOT WITH BOKEH
        else:
            data = self.filter_on_date(start,end).copy()
            max_target = data[field].max()
            min_target = data[field].min()
            p = figure(plot_width=self.plot_width, plot_height=self.plot_height, x_axis_type="datetime", 
                        title = '{0} evolution {1}'.format(self.code + ' ' + field, '' if reuters is None else 'and associated Reuters news'), 
                        tools = tools, active_drag = "xbox_zoom")

            source = ColumnDataSource(data)
            line = p.line('Date', field, line_width=2, alpha=0.8, source = source) # legend=self.code


            if reuters is not None:
                reuters_data = reuters.get_count(start,end, by_cluster = True,groupby = True)
                reuters_clusters = reuters.clusters
                x_rays = {**{reuters_clusters.iloc[i]['cluster']: [] for i in range(len(reuters_clusters))}}
                v_lines = []
                reuters_clusters['color'] = [tuple(map(lambda y: int(y*255),x)) for x in sns.hls_palette(len(reuters_clusters), l=.4, s=1)]

                vlines = {k:[] for k in list(reuters_clusters["cluster"].unique())}

                if clusters:
                    reuters_clusters = reuters_clusters[reuters_clusters['cluster'].isin(clusters)]

                for i in range(len(reuters_clusters)):                    
                    for j in range(len(reuters_data)):
                        date = reuters_data.iloc[j]['date']
                        count = reuters_data.iloc[j]["count_cluster_{0}".format(reuters_clusters.iloc[i]['cluster'])]
                        color = reuters_clusters.iloc[i]['color']
                        cluster = reuters_clusters.iloc[i]['cluster']
                        top_words = reuters_clusters.iloc[i]["top_10_words"]
                        if count > 0:
                            vlines[cluster].append(p.segment(x0 = [date],x1 = [date],y0 = [min_target - 10],y1 = [max_target],legend = str(cluster) +" : " +top_words[:50] + " ..",color = color))

                p.legend.location = "top_left"
                p.legend.click_policy="hide"

            if file:
                output_file("{}.html".format(self.code).replace("/","-"), title=self.code, mode = inline)
            show(p)


    def plot_cluster_metrics(self,reuters,field = "target",start = None, end = None, periods = 5):

        if '{0}_days_{1}_delta'.format(periods, field) not in self.data.columns:
            self.compute_deltas(field = field, periods = periods)

        data = self.filter_on_date(start,end)[[field, '{0}_days_{1}_delta'.format(periods, field)]]
        data = data.reset_index().rename(columns = {'Date': 'date'})
        reuters_data = reuters.get_count(start = start, end = end, by_cluster = True,groupby = False)
        merged = pd.merge(reuters_data, data, on = 'date')

        plt.figure(figsize = (12,6))
        sns.stripplot(x = merged["cluster"], y = merged['{0}_days_{1}_delta'.format(periods, field)],jitter = False)
        plt.show()


    def boxplot_cluster_metrics(self,reuters,field = "target",start = None, end = None, periods = 5, top = None, file = False):

        if '{0}_days_{1}_delta'.format(periods, field) not in self.data.columns:
            self.compute_deltas(field = field, periods = periods)

        data = self.filter_on_date(start,end)[[field, '{0}_days_{1}_delta'.format(periods, field)]]
        data = data.reset_index().rename(columns = {'Date': 'date'})
        reuters_data = reuters.get_count(start = start, end = end, by_cluster = True,groupby = False)
        merged = pd.merge(reuters_data, data, on = 'date')

        results = merged.groupby("cluster").agg({'{0}_days_{1}_delta'.format(periods, field):{
                                                "count":"count",
                                                "average delta (%)":lambda x : np.round(x.mean() *100,3),
                                                "std delta":lambda x : np.round(x.std(),3)}}).reset_index()
        results.columns = [results.columns.get_level_values(0)[0]] + list(results.columns.get_level_values(1))[1:]
        # print(results)
        results = results.sort_values("average delta (%)",ascending = False)
        
        if top:
            top_clusters = results.iloc[:top].index.values
        else:
            top_clusters = results.index.values

        merged = merged[merged['cluster'].isin(top_clusters)]
        merged['cluster_rank'] = merged['cluster'].astype('str').astype('category')
        merged['cluster_rank'].cat.set_categories([str(x) for x in top_clusters], inplace = True)
        merged.sort_values(['cluster_rank'], inplace = True)

        p = BoxPlot(merged, values = '{0}_days_{1}_delta'.format(periods, field), label = 'cluster_rank', color = 'cluster_rank', legend = False)

        if file:
            output_file("{}.html".format(self.code).replace("/","-"), title=self.code, mode = inline)

        show(p)

        







