#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""---------------------------------------------------------------------------------------------------------------------------
NATURAL LANGUAGE PROCESSING MODELS WITH Spacy
Grouping various scripts and functions for nlp 


Started on the 06/02/2017

http://textminingonline.com/getting-started-with-spacy
http://blog.sharepointexperience.com/2016/01/nlp-and-sharepoint-part-1/
http://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb#topic=10&lambda=1&term=
http://iamaaditya.github.io/2016/04/visual_question_answering_demo_notebook
https://radimrehurek.com/gensim/tutorial.html
http://stackoverflow.com/questions/195010/how-can-i-split-multiple-joined-words
http://www.markhneedham.com/blog/2015/02/12/pythongensim-creating-bigrams-over-how-i-met-your-mother-transcripts/
https://explosion.ai/blog/sense2vec-with-spacy
https://explosion.ai/blog/chatbot-node-js-spacy


About Latent Dirichlet Allocation : 
Common LDA limitations (http://stats.stackexchange.com/questions/164621/limitation-of-lda-latent-dirichlet-allocation/208630):

- Fixed K (the number of topics is fixed and must be known ahead of time)
- Uncorrelated topics (Dirichlet topic distribution cannot capture correlations)
- Non-hierarchical (in data-limited regimes hierarchical models allow sharing of data)
- Static (no evolution of topics over time)
- Bag of words (assumes words are exchangeable, sentence structure is not modeled)
- Unsupervised (sometimes weak supervision is desirable, e.g. in sentiment analysis)
 A number of these limitations have been addressed in papers that followed the original LDA work. Despite its limitations, LDA is central to topic modeling and has really revolutionized the field.

theo.alves.da.costa@gmail.com
https://github.com/theolvs
---------------------------------------------------------------------------------------------------------------------------
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import json
import spacy
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from ekimetrics.nlp import utils as nlp_utils
from ekimetrics.nlp import models
from ekimetrics.utils.time import Timer


import warnings
warnings.filterwarnings("ignore")





def filter_token(token,white_space = True,stop_words = False,punctuation = False,number = False,out_of_vocabulary_spacy = False,out_of_vocabulary = False,min_length = None,max_length = None):
    if white_space and token.is_space:
        return True
    if stop_words and token.is_stop:
        return True
    if punctuation and (token.is_punct or nlp_utils.is_punct(token.orth_)):
        return True
    if number and (token.like_num or nlp_utils.is_number(token.orth_,strict = True)):
        return True
    if out_of_vocabulary_spacy and token.is_oov:
        return True
    if out_of_vocabulary and nlp_utils.out_of_vocabulary(token.orth_):
        return True
    if min_length is not None and len(token.orth_) <= min_length:
        return True
    if max_length is not None and len(token) >= max_length:
        return True
    
    return False


def filter_document(document,white_space = True,stop_words = False,punctuation = False,number = False,out_of_vocabulary_spacy = False,out_of_vocabulary = False,min_length = None,max_length = None):
    tokens = []
    for token in document:
        if not filter_token(token,white_space,stop_words,punctuation,number,out_of_vocabulary_spacy,out_of_vocabulary,min_length,max_length):
            tokens.append(token.orth_)

    return " ".join(tokens)



def is_punct_or_space(token):
    return token.is_punct or token.is_space





#=============================================================================================================================
# SPACY DOCUMENT
#=============================================================================================================================







class Spacy_Document(models.Document):
    def __init__(self,document,nlp):
        self.title = document.title
        self.nlp = nlp
        self.set_model_from_text(document.text)


    def __repr__(self):
        return self.title

    def __str__(self):
        return self.__repr__()

    def get_text(self):
        return self.model.text

    def get_tokens(self):
        return [token.orth_ for token in self.model]

    def get_entities(self):
        return self.model.ents

    def set_model_from_text(self,text):
        self.model = self.nlp(text)
        self.tokens = self.get_tokens()

    def find_entities(self,persons = True,organizations = True):
        entity_type = []
        if persons:
            entity_type.append("PERSON")
        if organizations:
            entity_type.append("ORG")

        entities = [{"entity":entity.orth_,"type":entity.label_} for entity in self.get_entities() if entity.label_ in ['ORG','PERSON']]

        return pd.DataFrame(entities)


    def filter(self,white_space = True,stop_words = False,punctuation = False,number = False,out_of_vocabulary_spacy = False,out_of_vocabulary = False,min_length = None,max_length = None):
        filtered_text = filter_document(self.model,white_space,stop_words,punctuation,number,out_of_vocabulary_spacy,out_of_vocabulary,min_length,max_length)
        self.set_model_from_text(filtered_text)


    def lemmatize(self):
        self.set_model_from_text(lemmatize(self.model))


















#=============================================================================================================================
# SPACY CORPUS
#=============================================================================================================================





class Spacy_Corpus(models.Corpus):
    def __init__(self,corpus = None,nlp = None,json_path = None,max_documents = None):
        if nlp is None:
            nlp = spacy_brain()
        self.nlp = nlp
        self.documents = []
        self.max_documents = max_documents

        self.timer = Timer()

        if corpus is not None:
            self.load_corpus(corpus)
        elif json_path is not None:
            self.load_json_of_texts(json_path)
        else:
            pass

        self.entities = self.find_entities()





    def load_corpus(self,corpus):
        self.start_timer()
        for i,document in enumerate(corpus.documents[:self.max_documents]):
            length = len(corpus.documents) if self.max_documents is None else min([len(corpus.documents),self.max_documents])
            print('\r[{}/{}] Spacy NLP analysis '.format(i+1,length),end = "")
            self.documents.append(Spacy_Document(document,self.nlp))
        self.end_timer()


    def load_json_of_texts(self,json_path):
        corpus = models.Corpus()
        corpus.load_json_of_texts(json_path,process = False)
        self.__init__(corpus = corpus,nlp = self.nlp,max_documents = self.max_documents)

    def save_as_json_of_texts(self,json_path):
        dictionary = {document.title:document.get_text() for document in self.documents}
        with open(json_path,'w') as file:
            json.dump(dictionary,file)


    def show(self,n):
        print(self.documents[n].model)


    def get_tokens(self,cluster = None,flatten = False):
        if flatten:
            tokens = self.get_tokens(cluster = cluster,flatten = False)
            return [token for y in tokens for token in y]
        else:
            if cluster is None:
                return [document.get_tokens() for document in self.documents]
            else:
                return [document.get_tokens() for document in self.documents if document.cluster == cluster]


    def find_entities(self,persons = True,organizations = True):
        entities = pd.DataFrame(columns = ["document","entity","type"])
        for document in self.documents:
            entities_document = document.find_entities(persons = persons,organizations = organizations)
            entities_document["document"] = document.title
            entities = entities.append(entities_document)
        return entities.reset_index(drop = True)


    def filter(self,white_space = True,stop_words = False,punctuation = False,number = False,out_of_vocabulary_spacy = False,out_of_vocabulary = False,min_length = None,max_length = None):
        self.start_timer()
        for i,document in enumerate(self.documents):
            print('\r[{}/{}] Filtering unwanted tokens '.format(i+1,len(self.documents)),end = "")
            document.filter(white_space,stop_words,punctuation,number,out_of_vocabulary_spacy,out_of_vocabulary,min_length,max_length)
        self.end_timer()


    def lemmatize(self):
        self.start_timer()
        for i,document in enumerate(self.documents):
            print('\r[{}/{}] Lemmatizing tokens '.format(i+1,len(self.documents)),end = "")
            document.lemmatize()
        self.end_timer()

        


    def clean(self):
        print(">> Cleaning the corpus")
        self.filter(punctuation = True,number = True)
        self.bigrams = self.token_collocation()
        self.trigrams = self.token_collocation()
        self.lemmatize()
        self.filter(stop_words = True,min_length = 2,max_length = 30,out_of_vocabulary = True,out_of_vocabulary_spacy = True)



