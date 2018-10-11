#!/usr/bin/env python
# -*- coding: utf-8 -*- 



__author__ = "Theo"


"""--------------------------------------------------------------------
NATURAL LANGUAGE PROCESSING UTILS
Grouping various scripts and functions for nlp 


Started on the 16/01/2017

https://github.com/facebookresearch/fastText
http://textminingonline.com/dive-into-nltk-part-i-getting-started-with-nltk
http://textminingonline.com/getting-started-with-word2vec-and-glove
https://spacy.io/
http://cs224d.stanford.edu/
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

https://stackoverflow.com/questions/25566426/correcting-repeated-letters-in-user-messages

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pandas import read_hdf
from pandas import HDFStore,DataFrame
from tqdm import tqdm
from PIL import Image

# NLTK
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords


# SKLEARN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
# GENSIM
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence




# BOKEH
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, value

# PLOTLY
import plotly.plotly as py
import plotly.graph_objs as go
import plotly



# SpaCy
try:
    import spacy
except:
    pass










#=============================================================================================================================
# FILTER TOKEN FUNCTIONS
#=============================================================================================================================



def filter_token(token,method = "nltk",**kwargs):
    if method == "nltk":
        return filter_token_nltk(token,**kwargs)
    elif method == "spacy":
        return filter_token_spacy(token,**kwargs)
    else:
        raise ValueError("Method must be 'nltk' or 'spacy'")



def filter_token_nltk(token,
                      stop_words = False,punctuation = False,number = False,
                      stop_vocab = None,strict_number = True,
                      min_length = None,max_length = None,
                      language = None,**kwargs):


    if stop_words and is_stop_word(token,language):
        return True
    if punctuation and is_punct(token):
        return True
    if number and is_number(token,strict = strict_number):
        return True
    if stop_vocab is not None and is_in_stop_vocab(token,stop_vocab):
        return True
    if min_length is not None and is_too_short(token,min_length):
        return True
    if max_length is not None and is_too_long(token,max_length):
        return True

    
    return False  




def filter_token_spacy(token,
                       stop_words = False,punctuation = False,number = False,
                       stop_vocab = None,strict_number = True,
                       min_length = None,max_length = None,
                       out_of_vocabulary_spacy = False,**kwargs):
    if token.is_space:
        return True
    if stop_words and token.is_stop:
        return True
    if punctuation and (token.is_punct or is_punct(token.orth_)):
        return True
    if number and (token.like_num or is_number(token.orth_,strict = strict_number)):
        return True
    if out_of_vocabulary_spacy and token.is_oov:
        return True
    if stop_vocab is not None and is_in_stop_vocab(token.orth_,stop_vocab):
        return True
    if min_length is not None and is_too_short(token.orth_,min_length):
        return True
    if max_length is not None and is_too_long(token.orth_,max_length):
        return True
    
    return False




#=============================================================================================================================
# IS TOKEN RESPECTING A CONDITION
#=============================================================================================================================


def is_stop_word(token,language = None):
    return token in get_stop_words(language)

def is_too_long(token,max_length):
    return len(token) > max_length

def is_too_short(token,min_length):
    return len(token) < min_length

def is_punct(token):
    return len(set(stop_characters_punctuation).intersection(set(token))) > 0

def is_number(token,strict = False):
    if not strict:
        if len(set([str(x) for x in range(10)]).intersection(str(token)))>0:
            return True
        else:
            return False
    else:
        try:
            token = int(token)
            return True
        except Exception as e:
            return False


def is_in_stop_vocab(token,stop_vocab):
    return token in stop_vocab







#=============================================================================================================================
# SPACY UTILS
#=============================================================================================================================


def load_spacy_engine():
    print(">> Loading spaCy NLP engine")
    nlp = spacy.load('en')
    return nlp






#=============================================================================================================================
# TOKENIZING
#=============================================================================================================================





def tokenize(text):
    tokens = nltk.wordpunct_tokenize(text)
    return tokens




#=============================================================================================================================
# LEMMATIZING
#=============================================================================================================================




def lemmatize_nltk(word):
    pos_category = find_syntactic_categories(word)
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word,pos = pos_category)


def find_syntactic_categories(word):
    category = pos_tag([word])[0][1][0].lower()
    if category == "j":
        category = "a"
    elif category not in ["v","a","n","r"]:
        category = "n"
    return category



def lemmatize_spacy(document):
    lemmatized_tokens = []
    for token in document:
        if is_punct_or_space(token):
            lemmatized_tokens.append(token.orth_)
        else:
            lemmatized_tokens.append(token.lemma_)
    return " ".join(lemmatized_tokens)







#=============================================================================================================================
# IO FUNCTIONS
#=============================================================================================================================




def read_file(file_path):
    if file_path.endswith(".txt"):
        text = open(file_path,encoding = "utf8").read()

    elif file_path.endswith(".pdf"):
        text = convert_pdf_to_txt(file_path)

    else:
        print('File type not recognized')


    return text







#=============================================================================================================================
# PLOTTING
#=============================================================================================================================



def plot_word_cloud(text,title = "",width = 600,height = 300,max_words = 50,figsize = (18,10),max_font_size = 50,mask_file = None,**kwargs):
    from wordcloud import WordCloud
    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=figsize)
        if mask_file is not None:
            mask = np.array(Image.open(mask_file))
        else:
            mask = None
        wordcloud = WordCloud(background_color='white', width=width, height=height,
                        max_font_size=max_font_size, max_words=max_words,
                        mask = mask,**kwargs).generate(text)
        wordcloud.recolor(random_state=0)
        plt.imshow(wordcloud)
        plt.title(title, fontsize=20)
        plt.axis("off")
        plt.show()
    return fig






#=============================================================================================================================
# ANALYSIS
#=============================================================================================================================


def count_words(words):
    return pd.DataFrame(pd.Series(words).value_counts(),columns = ["count"])








#=============================================================================================================================
# LANGUAGE DETECTION
#=============================================================================================================================


def detect_language(text = None,words = None,details = False,default = "english"):
    '''Detect a language by using the number of stopwords from differents languages contained in the text sample
       Use NLTK library
    '''
    ratios = {}
    if text is not None:
        words = tokenize(text)
    else:
        words = words
    for language in nltk.corpus.stopwords.fileids():
        stopwords_set = set(nltk.corpus.stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        ratios[language] = len(common_elements)

    try:
        ratios = {k:v for k,v in sorted(ratios.items(),key = lambda x : x[1],reverse = True)[:3]}
        ratios = {k:float(v)/sum(ratios.values()) for k,v in ratios.items()}
        if details:
            return list(max(ratios.items(),key=lambda x : x[1]))+[ratios]
        else:
            return max(ratios.items(),key=lambda x : x[1])
    except Exception as e:
        return default






#=============================================================================================================================
# GENSIM
#=============================================================================================================================

def token_collocation_model(sentences):
    """sentences must be a list of sentences, which being list of independent words"""
    model = Phrases(sentences)
    return model


def count_token_collocation_model(model):
    """Returns a Counter object that can be used to detect the best collocations"""
    bigram_counter = Counter()
    for key in tqdm(model.vocab.keys(),desc = "Token collocation"):
        if str(key) not in stopwords.words("english"):
            if len(str(key).split("_")) > 1:
                bigram_counter[key] += model.vocab[key]
    return bigram_counter


def bow_generator(sentences,dictionary):
    for sentence in sentences:
        yield dictionary.doc2bow(sentence)







#=============================================================================================================================
# SOCIAL NETWORK NLP UTILS
#=============================================================================================================================


    
def extract_emojis(text):
    try:
        import emoji
    except:
        pass
    list_emojis = []
    for c in text:
        if c in emoji.UNICODE_EMOJI:
            list_emojis.append(c)
    smileys = [ '😘', '😍' , '😢' , '😋' , '❤' , '🤤' , '😂' , '😩' , '😡' , '😭' , '😔' , '😀' , '😥' , '😁' , '😒' , '🤢' , '😅' , '😕' ]
    output = []
    for i in list_emojis:
        if i in smileys:
            output.append(i)
    return [''.join( c for c in text if c not in emoji.UNICODE_EMOJI), output ]






#=============================================================================================================================
# GETTERS
#=============================================================================================================================


def get_stop_words(language = None):
    if language is None:
        language = list(nltk.corpus.stopwords.fileids())

    if type(language) != list:
        return list(set(nltk.corpus.stopwords.words(language)))

    else:
        return [x for lang in language for x in get_stop_words(lang)]




#=============================================================================================================================
# VOCABS
#=============================================================================================================================




stop_characters_punctuation = ["\'",".",";",",",":","?","!","'","\"","(",")","&","”","“","‘","’"," ","<",">","►"
                               ,"☰","-","/","://","...","€","$","£","%","*","¨","+","-","*","{","}","[","]","=","|","→"
                               ,"©","@","®","…","🕰","🙌","🇩🇰"]

stop_vocabulary = [
    "http","www","’s","privacyterms",
]




#=============================================================================================================================
# ENTITIES
#=============================================================================================================================
