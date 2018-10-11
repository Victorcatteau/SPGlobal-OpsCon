#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
UTILS
time utils

Started on the 02/02/2017


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



import time
import numpy as np


try:
    import winsound
except:
    pass






#=============================================================================================================================
# MISC
#=============================================================================================================================


def sec_to_hms(seconds):
    if seconds < 60:
        return "{}s".format(int(seconds))
    elif seconds < 60*60:
        m,s = divmod(int(seconds),60)
        return "{}m{}s".format(m,s)
    else:
        m,s = divmod(int(seconds),60)
        h,m = divmod(m,60)
        return "{}h{}m{}s".format(h,m,s)




def play_alarm(file = "C:/sound/alarm.wav"):
    winsound.PlaySound(file, winsound.SND_FILENAME)

























#=============================================================================================================================
# DEPRECATED
#=============================================================================================================================


class Timer():
    def __init__(self):
        self.start = time.time()


    def sec_to_minutes(self,seconds):
        minutes = int(int(seconds)/60)
        left_seconds = int(int(seconds)%60)
        output = str(minutes)+"m"+str(left_seconds)+"s"
        return output

    def end(self,name = ""):
        if name != "":
            name = '"{}" '.format(name)
        end = time.time()
        print('... Process {}finished in {}'.format(name,self.sec_to_minutes(end-self.start)))


    def reset(self):
        self.__init__()




def get_colors():

    colors = ["blue","red","green","magenta","cyan","purple",'brown','orange','lawngreen','darkmagenta','deepskyblue','dimgray','darkorchid','darkred']*4
    # self.colors = ['aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond','blue','blueviolet','brown',,,'chartreuse','chocolate','coral','cornflowerblue','cornsilk','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen',,'darksalmon','darkseagreen','darkslateblue','darkslategray','darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','floralwhite','forestgreen','fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgreen','lightgray','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen']
    return colors


""""---------------------------------------------------------------------------------
LINEAR ALGEBRA
    ---------------------------------------------------------------------------------
"""


def normalize_l2(x):
    """Normalize a vector along the L2 norm"""
    norm = np.sqrt(np.sum(x**2))
    return x/norm



def distance(x,y = None,metric = "euclidean"):
    """
    Possible distances are :
    - ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']. 
    - ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    """
    x = np.array(x)
    if len(x.shape)<=1:
        x = x.reshape(1,-1)
        y = np.array(y).reshape(1,-1) if y is not None else None
    return pairwise_distances(x,y,metric = metric)




def get_distances():
    return ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan','braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
   
