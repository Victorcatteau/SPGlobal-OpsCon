#!/usr/bin/env python
# -*- coding: utf-8 -*- 


__author__ = "Theo"



"""--------------------------------------------------------------------
CHARTS
Started on the 05/09/2017

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


# USUAL LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# SEABORN
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)


# PLOTLY
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

# Matplotlib default configuration
plt.style.use('ggplot')
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = "#d4d4d4"
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 2





def init_plotly_offline():
    """
    Initialize the offline notebook mode for plotly plotting
    """
    from plotly.offline import init_notebook_mode
    init_notebook_mode(connected = True)










def plot_bar_chart(data,plotly = True,dash = True,figsize = (20,7),title = "",xlabel = "",ylabel = "",**kwargs):
    """ 
    Plot a simple bar chart with one variable

    :param pandas.DataFrame data: DataFrame with the value in the first column and the labels in the index
    :param bool plotly: ``False`` plots with matplotlib, ``True`` with plotly 
    :param bool dash: to either plot on the notebook or returns the fig for a Dash App
    :returns: plotly figure -- if dash is False
    
    """
    if plotly or dash:
        data = [go.Bar(x = list(data.index),y = list(data.loc[:,column]),name = column) for column in data.columns]
        layout = {
            "title":title,
            "xaxis":{"title":xlabel},
            "yaxis":{"title":ylabel},
            **kwargs,
        }

        fig = {"data":data,"layout":layout}

        if not dash:
            init_plotly_offline()
            iplot(fig)
        else:
            return fig
    else:
        legend = len(data.columns)>1
        data.plot(kind = "bar",figsize = figsize,legend = legend)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()






def plot_histogram(data,plotly = True,dash = False,figsize = (20,7),title = "",xlabel = "",ylabel = "",**kwargs):
    """ Plot histogram
    """
    if plotly or dash:
        data = [go.Histogram(x = data)]
        layout = {
            "title":title,
            "xaxis":{"title":xlabel},
            "yaxis":{"title":ylabel},
            **kwargs,
        }

        fig = {"data":data,"layout":layout}
        if not dash:
            init_plotly_offline()
            iplot(fig)
        else:
            return fig
    else:
        plt.hist(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()









def plot_line_chart(data,with_plotly = True,on_notebook = True,figsize = (20,7),title = "",xlabel = "",ylabel = "",**kwargs):
    """ 
    Plot a simple line chart with one variable

    :param pandas.DataFrame data: DataFrame with the value in the columns and the labels in the index
    :param bool with_plotly: ``False`` plots with matplotlib, ``True`` with plotly 
    :param bool on_notebook: to either plot on the notebook or returns the fig for a Dash App
    :returns: plotly figure -- if on_notebook is False
    
    """

    # With plotly
    if with_plotly:
        x = data.index
        data = [go.Scatter(x = x,y = data[column],name = column) for column in data.columns]

        # Safe arguments
        if "lw" in kwargs:
            kwargs.pop("lw")

        # Define the layout
        layout = {
            "title":title,
            "xaxis":{"title":xlabel},
            "yaxis":{"title":ylabel},
            **kwargs,
        }

        # Define the figure
        fig = {"data":data,"layout":layout}

        # Return the figure for Dash or plot the figure
        if on_notebook:
            init_plotly_offline()
            iplot(fig)
        else:
            return fig
    
    
    # With matplotlib
    else:
        legend = len(data.columns) > 1

        # Safe arguments
        if "height" in kwargs: kwargs.pop('height')
        if "width" in kwargs: kwargs.pop('width')

        # Plot
        data.plot(figsize = figsize,title = title,legend = legend,**kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()









def plot_pie_chart(data,plotly = True,dash = False,figsize = (8,8),title = "",**kwargs):
    """ 
    Plot a simple pie chart with one variable

    :param pandas.DataFrame data: DataFrame with the value in the columns and the labels in the index
    :param bool with_plotly: ``False`` plots with matplotlib, ``True`` with plotly 
    :param bool on_notebook: to either plot on the notebook or returns the fig for a Dash App
    :returns: plotly figure -- if on_notebook is False
    
    """

    data = data.iloc[:,:1]

    if plotly:
        trace = go.Pie(labels = list(data.index),values = list(data.iloc[:,0]))
        fig = {"data":[trace],"layout":{}}

        if dash:
            return fig
        else:
            init_plotly_offline()
            iplot(fig)
    else:
        data.iloc[:,0].plot(figsize = figsize,kind = "pie",subplots = True,autopct='%1.1f%%',startangle = 90,**kwargs)
        plt.title(title)
        plt.xlabel("")
        plt.ylabel("")
        plt.axis("equal")
        plt.show()






def plot_area_chart(data,with_plotly = True,on_notebook = True,figsize = (10,7),title = "",xlabel = "",ylabel = "",**kwargs):
    """ 
    Plot a filled area chart

    :param pandas.DataFrame data: DataFrame with the value in the columns and the labels in the index
    :param bool with_plotly: ``False`` plots with matplotlib, ``True`` with plotly 
    :param bool on_notebook: to either plot on the notebook or returns the fig for a Dash App
    :returns: plotly figure -- if on_notebook is False
    
    """

    if with_plotly:
        fig = [go.Scatter(x = data.index,y = data[column],name = column,fill = 'tonexty') for column in data.columns]
        if on_notebook:
            init_plotly_offline()
            iplot(fig)
        else:
            return fig
    else:
        for i in range(1,len(data.columns)):
            data.iloc[:,i] = data.iloc[:,i] - data.iloc[:,i-1]
        data.plot(kind = "area",figsize = figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()









def plot_heatmap(data,with_plotly = True,on_notebook = True,**kwargs):
    """ 
    Plot a heatmap

    :param pandas.DataFrame data: DataFrame of a matrix format
    :param bool with_plotly: ``False`` plots with matplotlib, ``True`` with plotly 
    :param bool on_notebook: to either plot on the notebook or returns the fig for a Dash App
    :returns: plotly figure -- if on_notebook is False
    
    """
    if with_plotly:
        trace = go.Heatmap(
                   z=data.as_matrix(),
                   x=data.columns,
                   y=data.index,
                   colorscale = "Picnic")
        fig = [trace]
        if on_notebook:
            init_plotly_offline()
            iplot(fig)
        else:
            return fig
    else:
        figsize = kwargs["figsize"] if "figsize" in kwargs else (15,15)
        plt.figure(figsize = figsize)
        sns.heatmap(data)
        plt.show()











def plot_scatter_chart(data,x,y,z = None,color = None,with_plotly = True,on_notebook = True,title = "",**kwargs):
    """ 
    Plot a filled area chart

    :param pandas.DataFrame data: DataFrame with the value in the columns and the labels in the index
    :param bool with_plotly: ``False`` plots with matplotlib, ``True`` with plotly 
    :param bool on_notebook: to either plot on the notebook or returns the fig for a Dash App
    :returns: plotly figure -- if on_notebook is False
    
    """

    if with_plotly:

        # Getting the color mapping
        if color is not None:
            color = data[color]


        # Defining the trace
        trace = [go.Scatter(x = data[x],y = data[y],text = data[z] if z is not None else None,mode = 'text+markers',textposition='middle center',marker = {"color":color})]

        # Layout
        layout = dict(
            title = title,
            hovermode = 'closest',
            height = 800,
            width = 800,
            xaxis = dict(title = x),
            yaxis = dict(title = y),
            )

        # Supercharging the default configuration
        for key in kwargs:
            layout[key] = kwargs[key]


        fig = {"data":trace,"layout":layout}

        if on_notebook:
            init_plotly_offline()
            iplot(fig)
        else:
            return fig
    else:
        # NOT IMPLEMENTED YET
        pass






def plot_map(points,dash = True,zoom = 5,height = 500,width = 600,
             api_token = "pk.eyJ1IjoidGhlb2x2cyIsImEiOiJjamM5NWgwaGoyZHJ1MnFwY3Exa2dmYjFuIn0.Qh-n3QNPf9_TZbqkBP0Qhg",full_france = True,**kwargs):
    """ 
    Plot a map with markers

    :param list points: a list of dictionaries {"lat","lng","label"}
    :param str api_token: mapbox public token
    :param bool dash: to either plot on the notebook or returns the fig for a Dash App
    :returns: plotly figure -- if dash is False
    
    """

    if type(points) != list: points = [points]


    lats = []
    lngs = []
    labels = []
    for point in points:
        lats.append(point.get("lat"))
        lngs.append(point.get("lng"))
        labels.append(point.get("label"))

    mean_lat = np.mean(lats)
    mean_lng = np.mean(lngs)


    if full_france:
        zoom = 4
        mean_lat = 46.786746
        mean_lng = 2.698687


    data = go.Data([go.Scattermapbox(
                lat=lats,
                lon=lngs,
                mode='markers',
                marker=go.Marker(size=14),
                text=labels)])

    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        height = height,
        width = width,
        margin=dict(t=0, b=0, l=0, r=0),
        mapbox=dict(
            accesstoken=api_token,
            bearing=0,
            center=dict(
                lat=mean_lat,
                lon=mean_lng
            ),
            pitch=0,
            zoom=zoom,
        ),
    )

    fig = {"data":data,"layout":layout}

    if not dash:
        init_plotly_offline()
        iplot(fig)
    else:
        return fig




def plot_network(G,on_notebook = True,dim = 2,layout = "spring",edge_width = 0.5,height = 500,width = 500,**kwargs):
    """Plot network
    """
    import networkx as nx
    assert dim in [2,3]
    assert layout in ["spring","kamada_kawai"]

    # Get layout
    pos = nx.spring_layout(G,dim = dim) if layout == "spring" else nx.kamada_kawai_layout(G,dim = dim)
    nx.set_node_attributes(G,pos,"pos")

    # Edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=go.Line(width=edge_width,color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    # Nodes
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        textposition='middle right',
        textfont=dict(size='0.5em'),
        mode='markers+text',
        hoverinfo='text',
    )

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        node_trace['text'].append(node)



    # Create figure

    data = [node_trace,edge_trace]
    layout = dict(
        autosize=True,
        showlegend=False,
        hovermode='closest',
        height=height,
        width=width,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig = {"data":data,"layout":layout}


    if on_notebook:
        init_plotly_offline()
        iplot(fig)
    else:
        return fig







def plot_network_3D(G,on_notebook = True,layout = "spring",edge_width = 0.5,height = 500,width = 500,**kwargs):
    """Plot network
    """
    import networkx as nx
    assert layout in ["spring","kamada_kawai"]

    # Get layout
    pos = nx.spring_layout(G,dim = 3) if layout == "spring" else nx.kamada_kawai_layout(G,dim = 3)
    nx.set_node_attributes(G,pos,"pos")

    # Edges
    edge_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        line=go.Line(width=edge_width,color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0,z0 = G.node[edge[0]]['pos']
        x1, y1,z1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
        edge_trace['z'] += [z0, z1, None]

    # Nodes
    node_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        textposition='middle right',
        textfont=dict(size='0.5em'),
        mode='markers+text',
        hoverinfo='text',
    )

    for node in G.nodes():
        x, y,z = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        node_trace['z'].append(z)
        node_trace['text'].append(node)



    # Create figure
    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )
    fig = go.Figure(
                data=go.Data([node_trace,edge_trace]),
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    height=height,
                    width=width,
                    margin=dict(t=0, b=0, l=0, r=0),
                    scene = go.Scene(
                        xaxis=go.XAxis(axis),
                        yaxis=go.YAxis(axis),
                        zaxis=go.ZAxis(axis)
                    )))

    if on_notebook:
        init_plotly_offline()
        iplot(fig)
    else:
        return fig