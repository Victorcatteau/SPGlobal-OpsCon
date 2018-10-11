#!/usr/bin/env python
# -*- coding: utf-8 -*- 


__author__ = "Theo"



"""--------------------------------------------------------------------
PLOTTING
Started on the 29/03/2017

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from bokeh.layouts import gridplot, layout, widgetbox,row
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import GlyphRenderer
from bokeh.models import ColumnDataSource, HoverTool, Div, CustomJS, BoxAnnotation, NumeralTickFormatter, Legend,Label
from bokeh.models.widgets import Panel, Tabs, Slider, Select, TextInput, Button
from bokeh.io import curdoc
from bokeh.charts import Histogram, Bar
from bokeh.charts.attributes import cat
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, CrosshairTool, BoxSelectTool, WheelZoomTool

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)




#==================================================================================================================================
# PLOTTING UTILS
#==================================================================================================================================



def is_numeric(series):
    for el in list(series):
        try:
            el = int(el)
        except Exception as e:
            return False
    return True






#==================================================================================================================================
# PLOTTING FUNCTIONS 
#==================================================================================================================================


def analyze_series(series = None):
    # HANDLING MISSING DATA
    na = pd.isnull(series)
    total_length = len(series)
    print("WARNING : {} missing values on {} total values ({}%)".format(na.sum(),total_length,np.round(na.sum()/total_length*100,1)))

    series = series.loc[~na]
    counts = series.value_counts()
    occurences = list(counts.index)

    return series,counts,occurences



def plot_distribution(series = None,target = None,data = None, title = None,maximum = 30,stacked = False,normalize = False,verbose = 1):
    
    if data is not None:
        series = data[series]
        if target is not None:
            target = data[target]
            
    series,counts,occurences = analyze_series(series = series)
    xlabel = ""
    if title is None:
        try:
            if target is None:
                title = "{} DISTRIBUTION".format(series.name.upper())
                xlabel = series.name
            else:
                title = "{} VS {} DISTRIBUTION".format(series.name.upper(),target.name.upper())
        except Exception as e:
            title = ""

    numeric_case = is_numeric(occurences) and len(occurences) > maximum

    if target is None:
        if numeric_case:
            if verbose: print("Inferred it is a numeric distribution with too many values : plotting a histogram")
            plt.figure(figsize = (12,3))
            plt.title(title)
            plt.xlabel(xlabel)
            min_limit = np.percentile(series.dropna().astype(float), 0)
            max_limit = np.percentile(series.dropna().astype(float), 95)
            series = series[(series > min_limit) & (series < max_limit)]
            plt.ylabel("count")

            series.plot(kind = "hist",bins = 30)
            plt.show()

        else:
            if verbose: print("Inferred it is a categorical distribution : plotting a bar chart")

            if len(counts) > maximum:
                if verbose: print("WARNING : Taking the first {} categories on the total {} different categories".format(maximum,len(counts)))
                counts = counts.head(maximum)


            plt.figure(figsize = (12,3))
            plt.title(title)
            plt.ylabel("count")
            plt.xlabel(xlabel)
            counts.plot(kind = "bar")
            plt.show()


    else:
        if stacked:
            if is_numeric(target):
                d = pd.DataFrame()
                d[series.name] = series
                d[target.name] = target
                binned_name = "{}_binned".format(target.name)
                d[binned_name] = find_percentiles(target).astype(str)
                d_stacked = d.groupby([series.name,binned_name])[series.name].count().unstack(binned_name).fillna(0)
                d_stacked = d_stacked[sorted(list(d_stacked.columns),key = lambda x : float(x.split(",")[1].replace("]","")),reverse = False)]
                categories = list(d_stacked.columns)
                d_stacked["total"] = d_stacked.apply(np.sum,axis = 1)
                d_stacked.sort_values("total",ascending = False)
                
                if normalize:
                    for cat in categories:
                        d_stacked[cat] = 100*d_stacked[cat]/d_stacked["total"]
                    
                d_stacked.drop("total",axis = 1,inplace = True)
                d_stacked.plot(kind = "bar",stacked = True,title = title,figsize = (12,3))
                plt.xticks(rotation = 45)
                plt.show()

                

        else:
            plt.figure(figsize = (12,3))
            plt.title(title)
            sns.stripplot(series,target,jitter=True) #,order=order)
            plt.xticks(rotation = 45)
            plt.show()
            
            
            
def find_percentiles(series,percentiles = [0, .25, .5, .75, 1.]):
    series_transformed = pd.qcut(series + pd.Series(np.random.random(len(series))/10000), percentiles, labels=None, retbins=False, precision=3)
    return series_transformed





#==================================================================================================================================
# DEPRECATED 
#==================================================================================================================================


# tools = 'reset,xbox_zoom,wheel_zoom,pan'


# params = {
#     "plot_height":400,
#     "plot_width":900,
#     "active_drag":"xbox_zoom",
#     "toolbar_location":"right",
#     "tools":tools,
# }


# def return_figure(plot_figure):
#     def wrapper(*args,**kwargs):
#         p,fig = plot_figure(*args,**kwargs)
#         if fig:
#             return p
#         else:
#             show(p)
#     return wrapper



# @return_figure
# def plot_bar(series = None,fig = False,title = "",maximum = None):
#     # HANDLING MISSING DATA
#     # na = (series == "" & pd.isnull(series))
#     na = pd.isnull(series)
#     total_length = len(series)
#     print("WARNING : {} missing values on {} total values ({}%)".format(na.sum(),total_length,np.round(na.sum()/total_length*100,1)))

#     # PLOTTING BAR CHART FOR THE CATEGORIESz
#     series = series.loc[~na]
#     counts = series.value_counts()

#     categories = counts.sort_values(ascending = False).reset_index()
#     categories.columns = [categories.columns[-1],"count"]
#     tooltips = [(x,"@{}".format(x)) for x in categories.columns[0:1]]
    
#     # FILTER ON THE FIRST RESULTS
#     def group_rest(data,maximum):
#         first = data.iloc[:maximum]
#         rest = data.iloc[maximum:]
#         return first

#     if maximum is not None and maximum < len(categories):
#         print("WARNING : Taking the first {} categories on the total {} different categories".format(maximum,len(categories)))
#         x_label = "Top {} categories".format(maximum)
#     else:
#         x_label = None

#     categories = group_rest(categories,maximum = maximum)

#     p = Bar(data = categories,label = cat(categories.columns[0],sort = False),values = 'count',tooltips = tooltips,**params,legend = None,title = title,xlabel = x_label,ylabel = "{} COUNT".format(title).lower())
#     return p,fig




