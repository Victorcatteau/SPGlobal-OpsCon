#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
NETWORK ANALYSIS
Started on the 15/05/2017


https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python


theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



# Usual
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import networkx as nx
import itertools


# Utils
from tqdm import tqdm




#------------------------------------------------------------------------------------------------------------------
# EFFICIENT MODELLING FOR NETWORKS 



def get_combinations(x):
    """
    Helper function that returns a list of 2-combinations from a list
    Using itertools optimized iterators
    """
    return list(itertools.combinations(x,2))







def build_network_from_list_of_occurences(list_of_occurences,min_count = 2,min_degree = 1):
    """
    The nodes of the networks are the unique occurences
    Arguments : 
        - a list of occurences (list of list) 
            Ex : 
            l = [["hello","bonjour","ça"],
                 ["hello","bonjour","up"],
                 ["bonjour","je","veux","dire","hello"]]
            The nodes would be "hello","bonjour", ... 

    Returns :
        - g , a networkx graph
    """

    g = nx.Graph()

    for occurences in tqdm(list_of_occurences):
        combinations = get_combinations(occurences)
        g.add_nodes_from(occurences)

        for combination in combinations:
            s,t = combination
            if g.has_edge(s,t):
                g[s][t]["weight"] += 1
            else:
                g.add_edge(s,t,weight = 1)


    # REMOVING MIN COUNT EDGES
    edges_to_remove = [x for x in g.edges(data = True) if x[2]["weight"] < min_count]
    g.remove_edges_from(edges_to_remove)

    # REMOVING MIN DEGREE NODES
    nodes_to_remove = [x[0] for x in g.degree().items() if x[1] < min_degree]
    g.remove_nodes_from(nodes_to_remove)

    return g







def build_network_on_similarity(nodes,similarity_function,threshold = None):
    """
    Arguments : 
        - nodes (list) must be a list of dictionaries [{"name":"node_name","data":X},{"name":"node_name","data":Y}]
        - similarity_function (function) (X,Y) is measuring the similarity between 2 nodes data

    Returns : 
        g a networkx Graph

    """

    g = nx.Graph()
    all_nodes = [x["name"] for x in nodes]
    if len(all_nodes) > 1000:
        print("WARNING : the computation will take {} steps".format(0.5*len(all_nodes)*(len(all_nodes)-1)))
    g.add_nodes_from(all_nodes)
    combinations = get_combinations(nodes)


    for combination in tqdm(combinations):
        nodes = [x["name"] for x in combination]
        similarity = similarity_function(*[x["data"] for x in combination])
        if threshold is None or similarity > threshold:
            g.add_edge(*nodes,weight = similarity)

    return g

    






def plot_graph(graph,with_labels = True,with_plotly = False,pos = None):
    """
    Plot a network graph
    """
    if with_plotly:
        pass

    else:
        nx.draw(graph,cmap = plt.get_cmap('jet'),with_labels = with_labels,pos = pos)






def export_graph_to_gml(graph,file_path):
    """
    Export a networkx graph to a .gml file readable with Gephi
    """
    nx.write_gml(graph,file_path)





#------------------------------------------------------------------------------------------------------------------
# SIMILARITIES FUNCTION

def cooccurrences_similarity(X,Y,norm = True):
    """
    Function computing the co-occurrences similarity between two vectors
    """

    if len(X) > 0 and len(Y) > 0:
        intersection = set(X).intersection(set(Y))
        if norm:
            return (len(intersection)*len(intersection))/(len(set(X))*len(set(Y)))
        else:
            return len(intersection)
    else:
        return 0.0


def cosine_similarity(X,Y):
    """
    Function computing the cosine similarity between two vectors
    """
    cosine = np.dot(X,Y)/(np.linalg.norm(X,2)*np.linalg.norm(Y,2))
    cosine = 0.0 if cosine <= 0.0 else cosine
    return cosine

